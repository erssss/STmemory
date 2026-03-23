from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import VectorStoreBase
from .numpy_store import NumpyVectorStore
from .qdrant_store import QdrantVectorStore
from .sqlite_store import SQLiteMemoriesVectorStore


@dataclass
class VectorStoreSettings:
    provider: str = "numpy"
    table_name: str = "memories"


class VectorStoreFactory:
    @staticmethod
    def create(provider: str, *, connection=None, table_name: Optional[str] = None, **kwargs: Any) -> VectorStoreBase:
        p = str(provider or "numpy").lower().strip()
        if p in {"numpy", "in_memory", "memory"}:
            return NumpyVectorStore()
        if p in {"sqlite", "sqlite_memories"}:
            if connection is None:
                raise ValueError("SQLiteMemoriesVectorStore requires a sqlite connection")
            return SQLiteMemoriesVectorStore(connection=connection, table_name=table_name or "memories")
        if p in {"qdrant"}:
            return QdrantVectorStore(
                location=kwargs.get("qdrant_location"),
                url=kwargs.get("qdrant_url"),
                api_key=kwargs.get("qdrant_api_key"),
                collection_name=kwargs.get("qdrant_collection") or "stmemory_deep",
                distance=kwargs.get("distance") or "cosine",
                timeout_s=int(kwargs.get("qdrant_timeout_s") or 5),
                prefer_grpc=bool(kwargs.get("qdrant_prefer_grpc") or False),
                hnsw_m=kwargs.get("qdrant_hnsw_m"),
                hnsw_ef_construct=kwargs.get("qdrant_hnsw_ef_construct"),
                full_scan_threshold=kwargs.get("qdrant_full_scan_threshold"),
                indexing_threshold=kwargs.get("qdrant_indexing_threshold"),
                on_disk_payload=kwargs.get("qdrant_on_disk_payload"),
                search_hnsw_ef=kwargs.get("qdrant_search_hnsw_ef"),
                search_exact=kwargs.get("qdrant_search_exact"),
                rerank_prefetch=int(kwargs.get("deep_vector_rerank_prefetch") or 50),
            )
        raise ValueError(f"Unsupported vector store provider: {provider}")
