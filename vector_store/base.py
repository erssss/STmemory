from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class OutputData:
    id: str
    score: float
    payload: Optional[Dict[str, Any]] = None


class VectorStoreBase(ABC):
    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[np.ndarray], payloads: Optional[List[Dict[str, Any]]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[OutputData]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        raise NotImplementedError

