from __future__ import annotations
from copy import deepcopy
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModel


class NVEmbedV2Embedder:
    """
    Minimal embedder wrapper around SentenceTransformer NV-Embed-v2.
    Provides:
      - encode_queries(queries)  : uses prompt prefix
      - encode_passages(passages): no prompt
    """

    def __init__(self, cfg, device: Optional[str] = None):
        self.cfg = cfg.model
        self.device = (
            torch.device(device) if device is not None else torch.device("cuda")
        )
        self.model_init_params = {
            "pretrained_model_name_or_path": self.cfg.embeddingName,
            "trust_remote_code": True,
            "torch_dtype": "float16",  # Use half-precision (16-bit) to reduce memory usage
        }
        self.encode_params = {
            "max_length": self.cfg.embedding_max_seq_length,
            "batch_size": self.cfg.batch_size,
            "normalize": self.cfg.normalize,
            "add_eos": self.cfg.add_eos,
            "num_workers": 32,
        }
        self.model = AutoModel.from_pretrained(**self.model_init_params)
        self.model.to(self.device)
        self.model.eval()

        # NV-Embed model card examples set these:
        self.model.tokenizer.padding_side = "right"

        self._query_instruction = (
            "Given a question, retrieve passages that answer the question"
        )

    def _maybe_add_eos(self, texts: Sequence[str]) -> List[str]:
        if not self.cfg.add_eos:
            return list(texts)
        eos = self.model.tokenizer.eos_token or ""
        if not eos:
            return list(texts)
        return [t + eos for t in texts]

    def batch_encode(
        self,
        texts: Sequence[str],
        instruction: str = "",
        **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = self._maybe_add_eos(texts)

        params = deepcopy(self.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = params.pop("batch_size", 16)
        do_norm = bool(params.pop("normalize", self.cfg.normalize))

        instruct_prefix = ""
        if instruction:
            instruct_prefix = f"Instruct: {instruction}\nQuery: "

        results_chunks = []
        it = range(0, len(texts), batch_size)
        pbar = (
            tqdm(total=len(texts), desc="Batch Encoding")
            if len(texts) > batch_size
            else None
        )

        for i in it:
            chunk = texts[i : i + batch_size]
            try:
                out = self.model.encode(
                    prompts=chunk,
                    instruction=instruct_prefix,
                    batch_size=len(chunk),
                    **params,
                )
            except TypeError:
                out = self.model.encode(
                    chunk,
                    prompt=instruct_prefix,
                    batch_size=len(chunk),
                    **params,
                )

            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
            results_chunks.append(np.asarray(out, dtype=np.float32))

            if pbar:
                pbar.update(len(chunk))

        if pbar:
            pbar.close()

        results = np.concatenate(results_chunks, axis=0)

        if do_norm:
            denom = np.linalg.norm(results, axis=1, keepdims=True) + 1e-12
            results = results / denom

        return results

    def encode_queries(
        self, queries: Sequence[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        return self.batch_encode(
            queries,
            instruction=self._query_instruction,
            batch_size=(batch_size or self.cfg.batch_size),
        )

    def encode_passages(
        self, passages: Sequence[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        return self.batch_encode(
            passages,
            instruction="",
            batch_size=(batch_size or self.cfg.batch_size),
        )
