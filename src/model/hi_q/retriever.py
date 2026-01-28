import os
import sys
from typing import List, Optional, Dict, Any, Tuple
import json
import numpy as np
import hashlib

from src.model.baseline.embedding_model.NVEmbedV2 import NVEmbedV2Embedder


class DenseRetriever:
    """
    Keeps:
      - passages: List[str]
      - passage_embeddings: np.ndarray (N, D) (normalized if cfg.normalize=True)

    Methods:
      - build_index(passages): precompute passage embeddings ONCE
      - search(queries, topk): return topk (passage, score) per query
    """

    def __init__(self, embedder: NVEmbedV2Embedder):
        self.embedder = embedder
        self.passages: List[str] = []
        self.passage_emb: Optional[np.ndarray] = None  # (N, D)

    # -----------------------------
    # NEW: cache helpers
    # -----------------------------
    @staticmethod
    def _sha1_of_passages(passages: List[str]) -> str:
        h = hashlib.sha1()
        for p in passages:
            h.update(p.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _save_cache_npz(
        self,
        cache_path: str,
        passages: List[str],
        emb: np.ndarray,
        meta: Dict[str, Any],
    ) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        passages_arr = np.array(passages, dtype=object)
        meta_json = json.dumps(meta, ensure_ascii=False)

        np.savez_compressed(
            cache_path,
            passages=passages_arr,
            passage_emb=emb.astype(np.float32, copy=False),
            meta=np.array([meta_json], dtype=object),
        )

    def _load_cache_npz(self, cache_path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(cache_path):
            return None

        try:
            data = np.load(cache_path, allow_pickle=True)
            passages = data["passages"].tolist()
            emb = data["passage_emb"]
            meta_json = data["meta"].tolist()[0]
            meta = json.loads(meta_json)

            return {"passages": passages, "passage_emb": emb, "meta": meta}
        except Exception:
            return None

    def build_or_load_index(
        self,
        passages: List[str],
        cache_path: str,
        batch_size: Optional[int] = None,
        force_rebuild: bool = False,
        validate_corpus: bool = True,
        validate_model: bool = True,
    ) -> None:

        passages = list(passages)
        corpus_hash = self._sha1_of_passages(passages)
        model_name = getattr(self.embedder.cfg, "model_name", None)
        normalize_flag = bool(getattr(self.embedder.cfg, "normalize", True))

        if (not force_rebuild) and os.path.exists(cache_path):
            cached = self._load_cache_npz(cache_path)
            if cached is not None:
                meta = cached["meta"]

                corpus_ok = (
                    (meta.get("corpus_hash") == corpus_hash)
                    if validate_corpus
                    else True
                )
                model_ok = (
                    (meta.get("embedding_model_name") == str(model_name))
                    if validate_model
                    else True
                )
                norm_ok = meta.get("normalize") == normalize_flag

                emb = cached["passage_emb"]
                passages_cached = cached["passages"]

                # shape check
                shape_ok = emb.ndim == 2 and emb.shape[0] == len(passages_cached)

                if corpus_ok and model_ok and norm_ok and shape_ok:
                    # load success
                    self.passages = passages_cached
                    self.passage_emb = emb

                    # guarantee normalize
                    if not self.embedder.cfg.normalize:
                        denom = (
                            np.linalg.norm(self.passage_emb, axis=1, keepdims=True)
                            + 1e-12
                        )
                        self.passage_emb = self.passage_emb / denom
                    return

        self.build_index(passages, batch_size=batch_size)
        meta = {
            "embedding_model_name": str(model_name),
            "normalize": normalize_flag,
            "corpus_hash": corpus_hash,
            "num_passages": len(self.passages),
            "dim": (
                int(self.passage_emb.shape[1]) if self.passage_emb is not None else None
            ),
        }
        self._save_cache_npz(cache_path, self.passages, self.passage_emb, meta)

    def build_index(
        self, passages: List[str], batch_size: Optional[int] = None
    ) -> None:

        self.passages = list(passages)
        self.passage_emb = self.embedder.encode_passages(
            self.passages, batch_size=batch_size
        )

        if self.passage_emb.ndim != 2 or self.passage_emb.shape[0] != len(
            self.passages
        ):
            raise ValueError("Passage embeddings shape mismatch.")

        # If normalize=False in config, you must normalize here for cosine:
        if not self.embedder.cfg.normalize:
            denom = np.linalg.norm(self.passage_emb, axis=1, keepdims=True) + 1e-12
            self.passage_emb = self.passage_emb / denom

    def search(self, queries: List[str], topk: int) -> List[List[Tuple[str, float]]]:
        if self.passage_emb is None:
            raise RuntimeError("Index not built. Call build_index(passages) first.")
        k = min(topk, self.passage_emb.shape[0])

        q_emb = self.embedder.encode_queries(queries)  # (Q, D)
        if not self.embedder.cfg.normalize:
            denom = np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12
            q_emb = q_emb / denom

        # cosine similarity via dot product since normalized
        sims = q_emb @ self.passage_emb.T  # (Q, N)

        # top-k selection per query
        topk_idx_unsorted = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]  # (Q, k)
        topk_scores_unsorted = np.take_along_axis(sims, topk_idx_unsorted, axis=1)

        order = np.argsort(-topk_scores_unsorted, axis=1)
        topk_idx = np.take_along_axis(topk_idx_unsorted, order, axis=1)
        topk_scores = np.take_along_axis(topk_scores_unsorted, order, axis=1)

        results: List[List[Tuple[str, float]]] = []
        for row_idx, row in enumerate(topk_idx):
            scores = topk_scores[row_idx].tolist()
            results.append(
                [
                    (self.passages[i], float(score))
                    for i, score in zip(row.tolist(), scores)
                ]
            )
        return results
