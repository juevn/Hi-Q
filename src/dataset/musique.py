from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Union
from functools import cached_property
from collections import defaultdict
import os
import json

from .base_dataset import BaseDataset
from .registry import register_dataset

PathLike = Union[str, Path]
JsonDict = Dict[str, Any]


@register_dataset("musique_full")
@register_dataset("musique")
class MuSiQue(BaseDataset):
    """MuSiQue dataset class
    data format:
    [
        {
            "id": "2hop__13548_13529",
            "paragraphs": [
                {
                    "idx": 0,
                    "title": ,
                    "paragraph_text": ,
                    "is_supporting": true
                },
                ...
                {
                    "idx": 19,
                    "title":
                    "paragraph_text": ,
                    "is_supporting": false
                }
            ],
            "question":
            "question_decomposition": [
                {
                    "id": 13548,
                    "question": "To whom was Messi's goal in the first leg of the Copa del Rey compared?",
                    "answer": "Diego Maradona",
                    "paragraph_support_idx": 1
                },
                {
                    "id": 13529,
                    "question": "When was #1 signed by Barcelona?",
                    "answer": "June 1982",
                    "paragraph_support_idx": 2
                }
            ],
            "answer": "June 1982",
            "answer_aliases": [],
            "answerable": true
        },
        ...
    ]
    """

    def __init__(self, cfg):
        benchmark_conf = cfg.benchmark

        self.corpus_path = Path(benchmark_conf.corpus_path)
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        self.doc_corpus = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

        self.dataset_path = Path(benchmark_conf.dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            datas = json.load(f)
        super().__init__(datas)
