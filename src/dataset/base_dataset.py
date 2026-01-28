from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Union
from abc import ABC
from functools import cached_property

PathLike = Union[str, Path]
JsonDict = Dict[str, Any]


class BaseDataset(ABC):
    def __init__(self, data: List[Dict[str, Any]]):
        self.remove_text_before_save = False
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
