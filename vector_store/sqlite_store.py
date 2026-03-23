from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import OutputData, VectorStoreBase


class SQLiteMemoriesVectorStore(VectorStoreBase):
    def __init__(self, connection, table_name: str = "memories"):
        self._conn = connection
        self._table = str(table_name or "memories")

    def upsert(self, ids: List[str], vectors: List[np.ndarray], payloads: Optional[List[Dict[str, Any]]] = None) -> None:
        if not ids or not vectors:
            return
        payloads = payloads or [None] * len(ids)
        rows: List[Tuple[Any, ...]] = []
        for vid, vec, payload in zip(ids, vectors, payloads):
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            dim = int(arr.size)
            blob = arr.tobytes()
            payload_json = json.dumps(payload, ensure_ascii=False) if payload else None
            rows.append((blob, dim, payload_json, str(vid)))
        if not rows:
            return
        cur = self._conn.cursor()
        cur.executemany(
            f"UPDATE {self._table} SET embedding = ?, embedding_dim = ?, metadata = COALESCE(metadata, ?) WHERE id = ?",
            rows,
        )
        self._conn.commit()

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[OutputData]:
        q = np.asarray(query_vector, dtype=np.float32)
        if q.size == 0:
            return []
        q = _l2_normalize(q)

        sql = f"SELECT id, embedding, embedding_dim, metadata FROM {self._table} WHERE embedding IS NOT NULL"
        params: List[Any] = []
        if filters:
            conds = []
            for k, v in filters.items():
                conds.append("json_extract(metadata, '$.' || ?) = ?")
                params.extend([str(k), v])
            if conds:
                sql += " AND " + " AND ".join(conds)

        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() or []

        scored: List[OutputData] = []
        for mid, blob, dim, meta in rows:
            if blob is None:
                continue
            d = int(dim) if dim else int(len(blob) // 4)
            if d <= 0:
                continue
            v = np.frombuffer(blob, dtype=np.float32, count=d)
            if v.size != q.size:
                continue
            score = float(np.dot(v, q))
            payload = None
            if meta:
                try:
                    payload = json.loads(meta)
                except Exception:
                    payload = None
            scored.append(OutputData(id=str(mid), score=score, payload=payload))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: int(limit)]

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        cur = self._conn.cursor()
        cur.executemany(
            f"UPDATE {self._table} SET embedding = NULL, embedding_dim = NULL WHERE id = ?",
            [(str(i),) for i in ids],
        )
        self._conn.commit()


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    if n <= 0:
        return x
    return x / n

