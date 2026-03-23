from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging

import numpy as np

from .base import OutputData, VectorStoreBase


_logger = logging.getLogger("stmemory.vector_store.qdrant")


@dataclass
class VectorStoreStats:
    upsert_calls: int = 0
    delete_calls: int = 0
    search_calls: int = 0
    upsert_ms: float = 0.0
    delete_ms: float = 0.0
    search_ms: float = 0.0
    last_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "upsert_calls": int(self.upsert_calls),
            "delete_calls": int(self.delete_calls),
            "search_calls": int(self.search_calls),
            "upsert_ms": float(self.upsert_ms),
            "delete_ms": float(self.delete_ms),
            "search_ms": float(self.search_ms),
            "last_error": str(self.last_error or ""),
        }


def _as_vec(x: Any) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32)
    if v.ndim != 1:
        v = v.reshape(-1)
    return v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


def _neg_l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return -float(np.linalg.norm(d))


class QdrantVectorStore(VectorStoreBase):
    def __init__(
        self,
        *,
        location: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "stmemory_deep",
        distance: str = "cosine",
        timeout_s: int = 5,
        prefer_grpc: bool = False,
        hnsw_m: Optional[int] = None,
        hnsw_ef_construct: Optional[int] = None,
        full_scan_threshold: Optional[int] = None,
        indexing_threshold: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        search_hnsw_ef: Optional[int] = None,
        search_exact: Optional[bool] = None,
        rerank_prefetch: int = 50,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except Exception as e:
            raise ImportError("qdrant-client is required for provider=qdrant") from e

        self._models = models
        self._client = QdrantClient(
            location=location,
            url=url,
            api_key=api_key,
            prefer_grpc=bool(prefer_grpc),
            timeout=int(timeout_s) if timeout_s else None,
        )
        self._collection_name = str(collection_name or "stmemory_deep")
        self._distance_name = str(distance or "cosine").lower().strip()
        self._vector_size: Optional[int] = None
        self._stats = VectorStoreStats()
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        self._full_scan_threshold = full_scan_threshold
        self._indexing_threshold = indexing_threshold
        self._on_disk_payload = on_disk_payload
        self._search_hnsw_ef = search_hnsw_ef
        self._search_exact = search_exact
        self._rerank_prefetch = int(max(1, rerank_prefetch))

    @property
    def stats(self) -> VectorStoreStats:
        return self._stats

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def _to_distance(self, name: str):
        n = str(name or "cosine").lower().strip()
        if n in {"cos", "cosine"}:
            return self._models.Distance.COSINE
        if n in {"l2", "euclid", "euclidean"}:
            return self._models.Distance.EUCLID
        if n in {"dot", "inner", "ip"}:
            return self._models.Distance.DOT
        raise ValueError(f"Unsupported distance: {name}")

    def _ensure_collection(self, vector_size: int) -> None:
        if self._vector_size is None:
            self._vector_size = int(vector_size)

        try:
            info = self._client.get_collection(self._collection_name)
            size = None
            try:
                size = int(info.config.params.vectors.size)
            except Exception:
                size = None
            if size is not None and int(size) != int(self._vector_size):
                raise ValueError(
                    f"Qdrant collection vector size mismatch: expected {self._vector_size}, got {size}"
                )
            return
        except Exception:
            pass

        vectors_config = self._models.VectorParams(size=int(self._vector_size), distance=self._to_distance(self._distance_name))
        hnsw_config = None
        if any(v is not None for v in (self._hnsw_m, self._hnsw_ef_construct, self._full_scan_threshold)):
            hnsw_config = self._models.HnswConfigDiff(
                m=int(self._hnsw_m) if self._hnsw_m is not None else None,
                ef_construct=int(self._hnsw_ef_construct) if self._hnsw_ef_construct is not None else None,
                full_scan_threshold=int(self._full_scan_threshold) if self._full_scan_threshold is not None else None,
            )

        optimizers_config = None
        if self._indexing_threshold is not None:
            optimizers_config = self._models.OptimizersConfigDiff(indexing_threshold=int(self._indexing_threshold))

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=vectors_config,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            on_disk_payload=bool(self._on_disk_payload) if self._on_disk_payload is not None else None,
        )

    def _to_point_id(self, raw_id: str) -> uuid.UUID:
        try:
            return uuid.UUID(str(raw_id))
        except Exception:
            return uuid.uuid5(uuid.NAMESPACE_URL, str(raw_id))

    def upsert(self, ids: List[str], vectors: List[np.ndarray], payloads: Optional[List[Dict[str, Any]]] = None) -> None:
        t0 = time.time()
        self._stats.upsert_calls += 1
        try:
            if not ids:
                return
            if len(ids) != len(vectors):
                raise ValueError("ids and vectors length mismatch")

            vec0 = _as_vec(vectors[0])
            self._ensure_collection(int(vec0.size))

            points = []
            payloads = payloads or [None] * len(ids)
            for raw_id, v, p in zip(ids, vectors, payloads):
                vv = _as_vec(v)
                if int(vv.size) != int(self._vector_size or vv.size):
                    raise ValueError("Vector size mismatch")
                payload = dict(p or {})
                payload.setdefault("st_id", str(raw_id))
                points.append(self._models.PointStruct(id=self._to_point_id(str(raw_id)), vector=vv.tolist(), payload=payload))

            self._client.upsert(collection_name=self._collection_name, points=points)
        except Exception as e:
            self._stats.last_error = f"{type(e).__name__}: {e}"
            _logger.warning("qdrant upsert failed", extra={"collection": self._collection_name, "count": len(ids)})
            raise
        finally:
            dt = (time.time() - t0) * 1000.0
            self._stats.upsert_ms += dt
            _logger.debug(
                "qdrant upsert",
                extra={"collection": self._collection_name, "count": len(ids or []), "ms": float(dt)},
            )

    def _search_params(self):
        if self._search_hnsw_ef is None and self._search_exact is None:
            return None
        return self._models.SearchParams(
            hnsw_ef=int(self._search_hnsw_ef) if self._search_hnsw_ef is not None else None,
            exact=bool(self._search_exact) if self._search_exact is not None else None,
        )

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[OutputData]:
        t0 = time.time()
        self._stats.search_calls += 1
        try:
            qv = _as_vec(query_vector)
            if self._vector_size is not None and int(qv.size) != int(self._vector_size):
                raise ValueError("Query vector size mismatch")
            self._ensure_collection(int(qv.size))

            filt = filters or {}
            wanted_distance = str(filt.get("distance") or self._distance_name).lower().strip()
            prefetch = int(filt.get("prefetch") or self._rerank_prefetch)
            prefetch = int(max(int(limit), prefetch))

            q_filter = filt.get("qdrant_filter")
            with_vectors = bool(wanted_distance and wanted_distance != self._distance_name)
            hits = self._client.search(
                collection_name=self._collection_name,
                query_vector=qv.tolist(),
                limit=int(prefetch if with_vectors else limit),
                query_filter=q_filter,
                with_payload=True,
                with_vectors=with_vectors,
                search_params=self._search_params(),
            )

            out: List[OutputData] = []
            if not with_vectors:
                for h in hits:
                    payload = dict(getattr(h, "payload", None) or {})
                    st_id = str(payload.get("st_id") or getattr(h, "id", ""))
                    score = float(getattr(h, "score", 0.0))
                    if self._distance_name in {"l2", "euclid", "euclidean"}:
                        score = -score
                    out.append(OutputData(id=st_id, score=score, payload=payload))
                return out

            scored = []
            for h in hits:
                payload = dict(getattr(h, "payload", None) or {})
                st_id = str(payload.get("st_id") or getattr(h, "id", ""))
                vv = getattr(h, "vector", None)
                if vv is None:
                    continue
                vec = _as_vec(vv)
                if wanted_distance in {"cos", "cosine"}:
                    s = _cosine(qv, vec)
                elif wanted_distance in {"l2", "euclid", "euclidean"}:
                    s = _neg_l2(qv, vec)
                elif wanted_distance in {"dot", "inner", "ip"}:
                    s = float(np.dot(qv, vec))
                else:
                    raise ValueError(f"Unsupported distance: {wanted_distance}")
                scored.append(OutputData(id=st_id, score=float(s), payload=payload))

            scored.sort(key=lambda x: float(x.score), reverse=True)
            return scored[: int(limit)]
        except Exception as e:
            self._stats.last_error = f"{type(e).__name__}: {e}"
            _logger.warning("qdrant search failed", extra={"collection": self._collection_name})
            raise
        finally:
            dt = (time.time() - t0) * 1000.0
            self._stats.search_ms += dt
            _logger.debug(
                "qdrant search",
                extra={"collection": self._collection_name, "limit": int(limit), "ms": float(dt)},
            )

    def delete(self, ids: List[str]) -> None:
        t0 = time.time()
        self._stats.delete_calls += 1
        try:
            if not ids:
                return
            points = [self._to_point_id(str(i)) for i in ids]
            self._client.delete(collection_name=self._collection_name, points_selector=points)
        except Exception as e:
            self._stats.last_error = f"{type(e).__name__}: {e}"
            _logger.warning("qdrant delete failed", extra={"collection": self._collection_name, "count": len(ids)})
            raise
        finally:
            dt = (time.time() - t0) * 1000.0
            self._stats.delete_ms += dt
            _logger.debug(
                "qdrant delete",
                extra={"collection": self._collection_name, "count": len(ids or []), "ms": float(dt)},
            )
