from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from vector_index import NumpyVectorIndex
from .base import OutputData, VectorStoreBase


class NumpyVectorStore(VectorStoreBase):
    def __init__(self):
        self._index = NumpyVectorIndex()

    def __len__(self) -> int:
        return len(self._index)

    def upsert(self, ids: List[str], vectors: List[np.ndarray], payloads: Optional[List[Dict[str, Any]]] = None) -> None:
        for vid, vec in zip(ids, vectors):
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            self._index.add(str(vid), arr)

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[OutputData]:
        q = np.asarray(query_vector, dtype=np.float32)
        hits = self._index.search(q, top_k=int(limit))
        return [OutputData(id=str(h.id), score=float(h.score), payload=None) for h in hits]

    def delete(self, ids: List[str]) -> None:
        for vid in ids:
            self._index.delete(str(vid))
