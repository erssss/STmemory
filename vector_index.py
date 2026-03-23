from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np


@dataclass
class VectorSearchResult:
    id: str
    score: float


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    if vecs.size == 0:
        return vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


class NumpyVectorIndex:
    def __init__(self, dim: Optional[int] = None):
        self._dim = dim
        self._ids: List[str] = []
        self._id_to_pos: Dict[str, int] = {}
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        self._dirty = False

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, item_id: str, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if self._dim is None:
            self._dim = int(vec.shape[1])
            self._matrix = np.zeros((0, self._dim), dtype=np.float32)
        if int(vec.shape[1]) != int(self._dim):
            raise ValueError(f"vector dim mismatch: {vec.shape[1]} != {self._dim}")
        vec = _l2_normalize(vec)

        if item_id in self._id_to_pos:
            pos = self._id_to_pos[item_id]
            self._matrix[pos] = vec[0]
            self._dirty = True
            return

        self._id_to_pos[item_id] = len(self._ids)
        self._ids.append(item_id)
        self._matrix = np.vstack([self._matrix, vec])
        self._dirty = True

    def delete(self, item_id: str) -> None:
        pos = self._id_to_pos.pop(item_id, None)
        if pos is None:
            return
        last_id = self._ids[-1]
        if pos != len(self._ids) - 1:
            self._ids[pos] = last_id
            self._id_to_pos[last_id] = pos
            self._matrix[pos] = self._matrix[-1]
        self._ids.pop()
        self._matrix = self._matrix[:-1]
        self._dirty = True

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[VectorSearchResult]:
        if not self._ids:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if self._dim is None:
            return []
        if int(q.shape[1]) != int(self._dim):
            raise ValueError(f"query dim mismatch: {q.shape[1]} != {self._dim}")
        q = _l2_normalize(q)
        sims = (self._matrix @ q.T).reshape(-1)
        k = int(max(1, min(top_k, len(self._ids))))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx_sorted = idx[np.argsort(-sims[idx])]
        return [VectorSearchResult(id=self._ids[i], score=float(sims[i])) for i in idx_sorted]

