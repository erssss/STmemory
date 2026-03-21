from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import numpy as np
import json
import os
import uuid
from collections import OrderedDict

from compression import MemoryCompressor, estimate_tokens
from temporal_model import parse_time_range_from_query, temporal_match_score
from vector_store import VectorStoreFactory


@dataclass
class MemoryEntry:
    """记忆条目数据结构"""
    id: str
    query: str
    response: str
    timestamp: datetime
    layer: str
    token_count: int
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    shallow_capacity: int = 100 
    working_capacity: int = 500 
    shallow_ttl: int = 300  # 5分钟
    working_ttl: int = 1800  # 30分钟
    deep_persist_path: str = "deep_memory.db"
    max_shallow_entries: int = 100
    max_working_entries: int = 50
    lambda_decay: float = 0.1  # 时间衰减系数
    alpha_similarity: float = 0.6  # 语义相似度权重
    beta_time: float = 0.3  # 时间权重
    gamma_layer: float = 0.1  # 层级权重
    token_budget_ratio: float = 0.8  # token预算比例
    embedding_device: Optional[str] = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "ollama"
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen3.5-9b"
    local_llm_timeout_s: int = 60
    local_llm_check: bool = True
    use_llm_summary: bool = False
    summary_llm_model: str = "MiniMax-M2.7"
    summary_llm_max_tokens: int = 256
    summary_max_chars: int = 200
    enable_compression: bool = True
    compressed_max_chars: int = 180
    compression_min_ratio: float = 0.5
    compression_target_ratio: float = 0.5
    compression_min_chars: int = 60
    compression_adapt_rate: float = 0.05
    deep_return_compressed: bool = True
    deep_vector_candidates: int = 50
    deep_expand_temporal_neighbors: int = 1
    deep_enable_vector_index: bool = True
    deep_enable_time_filter: bool = True
    deep_time_weight: float = 0.25
    deep_lexical_weight: float = 0.25
    deep_time_candidates_limit: int = 300
    locomo_search_context_token_budget: int = 60
    deep_vector_store_provider: str = "numpy"
    deep_vector_store_table: str = "memories"
    deep_vector_distance: str = "cosine"
    deep_vector_rerank_prefetch: int = 50
    deep_vector_auto_reindex: bool = True
    deep_vector_auto_reindex_max_entries: int = 5000
    deep_store_embeddings_in_sqlite: bool = True
    qdrant_location: Optional[str] = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "stmemory_deep"
    qdrant_timeout_s: int = 5
    qdrant_prefer_grpc: bool = False
    qdrant_hnsw_m: Optional[int] = None
    qdrant_hnsw_ef_construct: Optional[int] = None
    qdrant_full_scan_threshold: Optional[int] = None
    qdrant_indexing_threshold: Optional[int] = None
    qdrant_on_disk_payload: Optional[bool] = None
    qdrant_search_hnsw_ef: Optional[int] = None
    qdrant_search_exact: Optional[bool] = None
    deep_optimizer_enable: bool = False
    deep_optimizer_similarity_threshold: float = 0.88
    deep_optimizer_min_cluster_size: int = 3
    deep_optimizer_max_clusters: int = 5
    deep_optimizer_delete_sources: bool = True


class MemoryLayer(ABC):
    """记忆层抽象基类"""
    
    def __init__(self, config: MemoryConfig, layer_name: str):
        self.config = config
        self.layer_name = layer_name
        self.access_count = 0
        self.last_access_time = datetime.now()
    
    @abstractmethod
    def add(self, entry: MemoryEntry) -> bool:
        """添加记忆条目"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, budget: int) -> List[MemoryEntry]:
        """检索记忆条目"""
        pass
    
    @abstractmethod
    def decay(self, now: datetime) -> int:
        """执行时间衰减，返回删除的条目数"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass
    
    def update_access_info(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_access_time = datetime.now()


class ShallowMemoryLayer(MemoryLayer):
    """浅层记忆：最近N轮对话的原始token缓存"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config, "shallow")
        self.memories: OrderedDict[str, MemoryEntry] = OrderedDict()
        self.total_tokens = 0
    
    def add(self, entry: MemoryEntry) -> bool:
        """添加记忆条目，使用LRU策略"""
        if entry.id in self.memories:
            # 更新现有条目
            old_entry = self.memories.pop(entry.id)
            self.total_tokens -= old_entry.token_count
        
        # 检查容量限制
        while (len(self.memories) >= self.config.max_shallow_entries or 
               self.total_tokens + entry.token_count > self.config.max_shallow_entries * 100):
            # 移除最老的条目
            oldest_id, oldest_entry = self.memories.popitem(last=False)
            self.total_tokens -= oldest_entry.token_count
        
        self.memories[entry.id] = entry
        self.total_tokens += entry.token_count
        self.update_access_info()
        return True
    
    def retrieve(self, query: str, budget: int) -> List[MemoryEntry]:
        """按时间顺序检索最近的记忆"""
        self.update_access_info()
        results = []
        current_tokens = 0
        
        # 按时间倒序遍历
        for entry in reversed(list(self.memories.values())):
            if current_tokens + entry.token_count <= budget:
                results.append(entry)
                current_tokens += entry.token_count
            else:
                break
        
        return list(reversed(results))  # 保持时间顺序
    
    def decay(self, now: datetime) -> int:
        """基于TTL的过期清理"""
        expired_ids = []
        
        for entry_id, entry in self.memories.items():
            age = (now - entry.timestamp).total_seconds()
            if age > self.config.shallow_ttl:
                expired_ids.append(entry_id)
        
        removed_count = 0
        for entry_id in expired_ids:
            if entry_id in self.memories:
                removed_entry = self.memories.pop(entry_id)
                self.total_tokens -= removed_entry.token_count
                removed_count += 1
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "layer": self.layer_name,
            "entries": len(self.memories),
            "total_tokens": self.total_tokens,
            "access_count": self.access_count,
            "last_access": self.last_access_time.isoformat()
        }


class WorkingMemoryLayer(MemoryLayer):
    """工作记忆：经过去重、摘要后的关键信息"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config, "working")
        self.memories: OrderedDict[str, MemoryEntry] = OrderedDict()
        self.total_tokens = 0
        self.summaries: Dict[str, str] = {}
    
    def _generate_summary_llm_minimax(self, query: str, response: str) -> str:
        try:
            import anthropic
        except Exception:
            return ""

        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("MINIMAX_API_KEY") or ""
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

        query_trimmed = (query or "").strip()
        response_trimmed = (response or "").strip()
        if len(query_trimmed) > 2000:
            query_trimmed = query_trimmed[:2000]
        if len(response_trimmed) > 4000:
            response_trimmed = response_trimmed[:4000]

        prompt = (
            "请将下面的对话压缩为一个简洁摘要，保留关键事实、结论与实体。"
            f"摘要长度不超过{self.config.summary_max_chars}个字符。"
            "仅输出摘要正文，不要输出多余说明。"
            "\n\n"
            f"Q: {query_trimmed}\n"
            f"A: {response_trimmed}\n"
        )

        message = client.messages.create(
            model=self.config.summary_llm_model,
            max_tokens=self.config.summary_llm_max_tokens,
            system="你是一个擅长提炼关键信息的摘要助手。",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        )

        content = getattr(message, "content", None)
        if not content:
            return ""

        parts: List[str] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type != "text":
                continue
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

        return "\n".join(parts).strip()

    def _generate_summary_simple(self, query: str, response: str) -> str:
        combined = f"Q: {query}\nA: {response}"
        max_chars = max(20, int(self.config.summary_max_chars))
        if len(combined) > max_chars:
            return combined[:max_chars] + "..."
        return combined

    def _generate_summary(self, query: str, response: str) -> str:
        """生成对话摘要"""
        use_llm = bool(self.config.use_llm_summary)
        if use_llm:
            summary = self._generate_summary_llm_minimax(query, response)
            if summary:
                max_chars = max(20, int(self.config.summary_max_chars))
                if len(summary) > max_chars:
                    return summary[:max_chars] + "..."
                return summary

        return self._generate_summary_simple(query, response)
    
    def add(self, entry: MemoryEntry) -> bool:
        """添加记忆条目，生成摘要"""
        if entry.id in self.memories:
            old_entry = self.memories.pop(entry.id)
            self.total_tokens -= old_entry.token_count
            self.summaries.pop(entry.id, None)
        
        # 生成摘要
        summary = self._generate_summary(entry.query, entry.response)
        entry.metadata = entry.metadata or {}
        entry.metadata["summary"] = summary
        
        # 容量管理
        while (len(self.memories) >= self.config.max_working_entries or 
               self.total_tokens + entry.token_count > self.config.max_working_entries * 150):
            oldest_id, oldest_entry = self.memories.popitem(last=False)
            self.total_tokens -= oldest_entry.token_count
            self.summaries.pop(oldest_id, None)
        
        self.memories[entry.id] = entry
        self.total_tokens += entry.token_count
        self.update_access_info()
        return True
    
    def retrieve(self, query: str, budget: int) -> List[MemoryEntry]:
        """基于关键词匹配检索"""
        self.update_access_info()
        results = []
        current_tokens = 0
        query_words = set(query.lower().split())
        
        # 按相关性排序
        scored_entries = []
        for entry in self.memories.values():
            # 简单的关键词匹配评分
            entry_text = f"{entry.query} {entry.response}".lower()
            match_count = sum(1 for word in query_words if word in entry_text)
            score = match_count / len(query_words) if query_words else 0
            scored_entries.append((score, entry))
        
        # 按分数降序排序
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        
        for score, entry in scored_entries:
            if current_tokens + entry.token_count <= budget and score > 0:
                results.append(entry)
                current_tokens += entry.token_count
        
        return results
    
    def decay(self, now: datetime) -> int:
        """基于TTL的过期清理"""
        expired_ids = []
        
        for entry_id, entry in self.memories.items():
            age = (now - entry.timestamp).total_seconds()
            if age > self.config.working_ttl:
                expired_ids.append(entry_id)
        
        removed_count = 0
        for entry_id in expired_ids:
            if entry_id in self.memories:
                removed_entry = self.memories.pop(entry_id)
                self.total_tokens -= removed_entry.token_count
                self.summaries.pop(entry_id, None)
                removed_count += 1
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "layer": self.layer_name,
            "entries": len(self.memories),
            "total_tokens": self.total_tokens,
            "access_count": self.access_count,
            "last_access": self.last_access_time.isoformat()
        }


class DeepMemoryLayer(MemoryLayer):
    """深层记忆：结构化知识图谱与长期摘要"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config, "deep")
        self.db_path = config.deep_persist_path
        self._conn = None
        self.memories: Dict[str, MemoryEntry] = {}
        self.knowledge_graph: Dict[str, List[str]] = {}
        self._time_sorted_ids: List[str] = []
        self._encode_texts = None
        self._vector_store = None
        self._compressor = MemoryCompressor(
            max_chars=config.compressed_max_chars,
            min_ratio=config.compression_min_ratio,
            target_ratio=config.compression_target_ratio,
            min_chars=config.compression_min_chars,
        )
        self._compression_ema: Optional[float] = None
        self._init_database()
        self._vector_store = VectorStoreFactory.create(
            config.deep_vector_store_provider,
            connection=self._conn,
            table_name=config.deep_vector_store_table,
            distance=getattr(config, "deep_vector_distance", "cosine"),
            deep_vector_rerank_prefetch=getattr(config, "deep_vector_rerank_prefetch", 50),
            qdrant_location=getattr(config, "qdrant_location", None),
            qdrant_url=getattr(config, "qdrant_url", None),
            qdrant_api_key=getattr(config, "qdrant_api_key", None),
            qdrant_collection=getattr(config, "qdrant_collection", "stmemory_deep"),
            qdrant_timeout_s=getattr(config, "qdrant_timeout_s", 5),
            qdrant_prefer_grpc=getattr(config, "qdrant_prefer_grpc", False),
            qdrant_hnsw_m=getattr(config, "qdrant_hnsw_m", None),
            qdrant_hnsw_ef_construct=getattr(config, "qdrant_hnsw_ef_construct", None),
            qdrant_full_scan_threshold=getattr(config, "qdrant_full_scan_threshold", None),
            qdrant_indexing_threshold=getattr(config, "qdrant_indexing_threshold", None),
            qdrant_on_disk_payload=getattr(config, "qdrant_on_disk_payload", None),
            qdrant_search_hnsw_ef=getattr(config, "qdrant_search_hnsw_ef", None),
            qdrant_search_exact=getattr(config, "qdrant_search_exact", None),
        )
        self._load_from_database()

    def set_encoder(self, encode_texts_fn) -> None:
        self._encode_texts = encode_texts_fn
        if not bool(getattr(self.config, "deep_vector_auto_reindex", True)):
            return
        if not bool(self.config.deep_enable_vector_index):
            return
        provider = str(getattr(self.config, "deep_vector_store_provider", "numpy") or "numpy").lower().strip()
        if provider != "qdrant":
            return
        try:
            self._reindex_missing_vectors(max_entries=int(getattr(self.config, "deep_vector_auto_reindex_max_entries", 5000)))
        except Exception:
            pass

    def _reindex_missing_vectors(self, *, max_entries: int = 5000) -> Dict[str, Any]:
        if self._encode_texts is None:
            return {"reindexed": 0, "skipped": 0}
        limit = int(max(0, max_entries))
        if limit <= 0:
            return {"reindexed": 0, "skipped": 0}
        reindexed = 0
        skipped = 0
        for mid in list(self._time_sorted_ids)[-limit:]:
            entry = self.memories.get(mid)
            if entry is None:
                continue
            if entry.embedding is not None:
                skipped += 1
                continue
            text = self._entry_text_for_embedding(entry)
            if not text:
                skipped += 1
                continue
            vec = self._encode_texts([text])[0]
            entry.embedding = _l2_normalize_vec(np.asarray(vec, dtype=np.float32))
            self._vector_store.upsert([entry.id], [entry.embedding], payloads=[entry.metadata])
            reindexed += 1
        return {"reindexed": int(reindexed), "skipped": int(skipped)}
    
    def _init_database(self):
        """初始化SQLite数据库"""
        import sqlite3
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self._conn.cursor()
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                query TEXT,
                response TEXT,
                compressed_query TEXT,
                compressed_response TEXT,
                timestamp TEXT,
                timestamp_epoch INTEGER,
                layer TEXT,
                token_count INTEGER,
                raw_token_count INTEGER,
                embedding BLOB,
                embedding_dim INTEGER,
                metadata TEXT
            )
            """
        )
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                entity TEXT PRIMARY KEY,
                relations TEXT
            )
            """
        )

        self._ensure_schema(cursor)

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_ts_epoch ON memories(timestamp_epoch)
            """
        )
        
        self._conn.commit()

    def _ensure_schema(self, cursor) -> None:
        cursor.execute("PRAGMA table_info(memories)")
        cols = {str(r[1]) for r in (cursor.fetchall() or [])}
        wanted = {
            "compressed_query": "TEXT",
            "compressed_response": "TEXT",
            "timestamp_epoch": "INTEGER",
            "raw_token_count": "INTEGER",
            "embedding_dim": "INTEGER",
        }
        for name, typ in wanted.items():
            if name in cols:
                continue
            cursor.execute(f"ALTER TABLE memories ADD COLUMN {name} {typ}")

    def _load_from_database(self) -> None:
        if self._conn is None:
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT id, query, response, compressed_query, compressed_response, timestamp, timestamp_epoch,
                   layer, token_count, raw_token_count, embedding, embedding_dim, metadata
            FROM memories
            ORDER BY COALESCE(timestamp_epoch, 0) ASC
            """
        )
        rows = cur.fetchall() or []
        for (
            mid,
            q,
            r,
            cq,
            cr,
            ts,
            ts_epoch,
            layer,
            token_count,
            raw_token_count,
            emb_blob,
            emb_dim,
            meta_json,
        ) in rows:
            metadata = None
            if meta_json:
                try:
                    metadata = json.loads(meta_json)
                except Exception:
                    metadata = None
            metadata = metadata or {}
            if cq:
                metadata.setdefault("compressed_query", cq)
            if cr:
                metadata.setdefault("compressed_response", cr)
            if raw_token_count is not None:
                metadata.setdefault("raw_token_count", raw_token_count)

            dt = None
            if ts:
                try:
                    dt = datetime.fromisoformat(str(ts))
                except Exception:
                    dt = None
            if dt is None and ts_epoch is not None:
                try:
                    dt = datetime.fromtimestamp(int(ts_epoch))
                except Exception:
                    dt = datetime.now()
            if dt is None:
                dt = datetime.now()

            entry = MemoryEntry(
                id=str(mid),
                query=str(q or ""),
                response=str(r or ""),
                timestamp=dt,
                layer=str(layer or "deep"),
                token_count=int(token_count or 0),
                embedding=None,
                metadata=metadata,
            )
            if emb_blob is not None:
                d = int(emb_dim) if emb_dim else int(len(emb_blob) // 4)
                if d > 0:
                    try:
                        entry.embedding = np.frombuffer(emb_blob, dtype=np.float32, count=d).copy()
                        if self.config.deep_enable_vector_index:
                            self._vector_store.upsert([entry.id], [entry.embedding], payloads=[entry.metadata])
                    except Exception:
                        entry.embedding = None

            self.memories[entry.id] = entry
            self._time_sorted_ids.append(entry.id)
            entities = self._extract_entities(f"{entry.query} {entry.response}")
            for entity in entities:
                self.knowledge_graph.setdefault(entity, []).append(entry.id)
    
    def _extract_entities(self, text: str) -> List[str]:
        """简单的实体提取"""
        import re
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", (text or ""))
        seen = set()
        out: List[str] = []
        for t in tokens:
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
            if len(out) >= 16:
                break
        return out

    def _maybe_compress_entry(self, entry: MemoryEntry) -> None:
        if not self.config.enable_compression:
            return
        entry.metadata = entry.metadata or {}
        raw_q = (entry.query or "").strip()
        raw_r = (entry.response or "").strip()
        if raw_q:
            cq = self._compressor.compress_to_target(raw_q)
            entry.metadata["compressed_query"] = cq.compressed_text
            entry.metadata["raw_query_tokens"] = cq.raw_tokens
            entry.metadata["compressed_query_tokens"] = cq.compressed_tokens
        if raw_r:
            cr = self._compressor.compress_to_target(raw_r)
            entry.metadata["compressed_response"] = cr.compressed_text
            entry.metadata["raw_response_tokens"] = cr.raw_tokens
            entry.metadata["compressed_response_tokens"] = cr.compressed_tokens

        raw_tokens = estimate_tokens(raw_q) + estimate_tokens(raw_r)
        comp_tokens = estimate_tokens(str(entry.metadata.get("compressed_query") or raw_q)) + estimate_tokens(
            str(entry.metadata.get("compressed_response") or raw_r)
        )
        entry.metadata["raw_token_count"] = raw_tokens
        entry.metadata["compressed_token_count"] = comp_tokens
        entry.token_count = min(entry.token_count, comp_tokens) if entry.token_count else comp_tokens

        if raw_tokens > 0:
            ratio = comp_tokens / max(1, raw_tokens)
            rate = float(max(0.0, min(float(self.config.compression_adapt_rate), 1.0)))
            if self._compression_ema is None:
                self._compression_ema = ratio
            else:
                self._compression_ema = (1.0 - rate) * float(self._compression_ema) + rate * ratio
            target = float(self.config.compression_target_ratio)
            if self._compression_ema > target and self._compressor.max_chars > int(self._compressor.min_chars):
                self._compressor.max_chars = max(self._compressor.min_chars, int(self._compressor.max_chars * 0.97))
            elif self._compression_ema < target * 0.8:
                self._compressor.max_chars = min(int(self.config.compressed_max_chars), int(self._compressor.max_chars * 1.03))

    def _entry_text_for_embedding(self, entry: MemoryEntry) -> str:
        if not entry:
            return ""
        if self.config.deep_return_compressed and entry.metadata:
            cq = str(entry.metadata.get("compressed_query") or "").strip()
            cr = str(entry.metadata.get("compressed_response") or "").strip()
            if cq or cr:
                if cq and cr:
                    return f"{cq}\n{cr}"
                return cq or cr
        if entry.response:
            return f"{entry.query}\n{entry.response}"
        return entry.query
    
    def add(self, entry: MemoryEntry) -> bool:
        """添加记忆条目，构建知识图谱"""
        self._maybe_compress_entry(entry)
        self.memories[entry.id] = entry

        self._time_sorted_ids.append(entry.id)
        self._time_sorted_ids.sort(key=lambda x: self.memories[x].timestamp)
        
        # 提取实体
        entities = self._extract_entities(f"{entry.query} {entry.response}")
        
        # 更新知识图谱
        for entity in entities:
            if entity not in self.knowledge_graph:
                self.knowledge_graph[entity] = []
            self.knowledge_graph[entity].append(entry.id)

        if self.config.deep_enable_vector_index and self._encode_texts is not None:
            try:
                text = self._entry_text_for_embedding(entry)
                vec = self._encode_texts([text])[0]
                entry.embedding = _l2_normalize_vec(np.asarray(vec, dtype=np.float32))
                self._vector_store.upsert([entry.id], [entry.embedding], payloads=[entry.metadata])
            except Exception:
                pass
        
        # 持久化到数据库
        self._persist_entry(entry)
        self.update_access_info()

        if bool(self.config.deep_optimizer_enable):
            try:
                self.optimize(
                    compress=True,
                    deduplicate_exact=False,
                    similarity_threshold=float(self.config.deep_optimizer_similarity_threshold),
                    min_cluster_size=int(self.config.deep_optimizer_min_cluster_size),
                    max_clusters=int(self.config.deep_optimizer_max_clusters),
                    delete_sources=bool(self.config.deep_optimizer_delete_sources),
                )
            except Exception:
                pass
        return True
    
    def _persist_entry(self, entry: MemoryEntry):
        """持久化条目到数据库"""
        import sqlite3
        conn = self._conn or sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if entry.embedding is not None:
            entry.embedding = _l2_normalize_vec(np.asarray(entry.embedding, dtype=np.float32))
        store_emb = bool(getattr(self.config, "deep_store_embeddings_in_sqlite", True))
        embedding_blob = entry.embedding.tobytes() if (store_emb and entry.embedding is not None) else None
        embedding_dim = int(entry.embedding.size) if (store_emb and entry.embedding is not None) else None
        metadata_json = json.dumps(entry.metadata, ensure_ascii=False) if entry.metadata else None
        ts_epoch = int(entry.timestamp.timestamp())
        cq = None
        cr = None
        raw_token_count = None
        if entry.metadata:
            cq = entry.metadata.get("compressed_query")
            cr = entry.metadata.get("compressed_response")
            raw_token_count = entry.metadata.get("raw_token_count")
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, query, response, compressed_query, compressed_response, timestamp, timestamp_epoch, layer, token_count, raw_token_count, embedding, embedding_dim, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.query,
                entry.response,
                cq,
                cr,
                entry.timestamp.isoformat(),
                ts_epoch,
                entry.layer,
                entry.token_count,
                raw_token_count,
                embedding_blob,
                embedding_dim,
                metadata_json,
            ),
        )
        
        conn.commit()
        if self._conn is None:
            conn.close()
    
    def retrieve(self, query: str, budget: int) -> List[MemoryEntry]:
        """基于实体、向量与时间约束检索"""
        self.update_access_info()

        now = datetime.now()
        time_range = parse_time_range_from_query(query, now=now) if self.config.deep_enable_time_filter else None

        query_entities = self._extract_entities(query)
        candidates: Dict[str, float] = {}

        for entity in query_entities:
            ids = self.knowledge_graph.get(entity) or []
            for entry_id in ids[-200:]:
                if entry_id in self.memories:
                    candidates[entry_id] = max(candidates.get(entry_id, 0.0), 0.2)

        if self.config.deep_enable_vector_index and self._encode_texts is not None:
            try:
                qv = self._encode_texts([query])[0]
                hits = self._vector_store.search(qv, limit=int(self.config.deep_vector_candidates))
                for hit in hits:
                    candidates[hit.id] = max(candidates.get(hit.id, 0.0), float(hit.score))
            except Exception:
                pass

        if not candidates:
            candidates = {}

        if time_range is not None and (time_range.start is not None or time_range.end is not None):
            start = time_range.start
            end = time_range.end
            added = 0
            for mid in reversed(self._time_sorted_ids):
                entry = self.memories.get(mid)
                if entry is None:
                    continue
                if start is not None and entry.timestamp < start:
                    break
                if end is not None and entry.timestamp > end:
                    continue
                candidates[mid] = max(candidates.get(mid, 0.0), 0.15)
                added += 1
                if added >= int(max(1, self.config.deep_time_candidates_limit)):
                    break

        if not candidates:
            return []

        neighbor_hops = int(max(0, self.config.deep_expand_temporal_neighbors))
        if neighbor_hops > 0 and self._time_sorted_ids:
            pos = {mid: i for i, mid in enumerate(self._time_sorted_ids)}
            base_ids = list(candidates.keys())
            for mid in base_ids:
                i = pos.get(mid)
                if i is None:
                    continue
                for d in range(1, neighbor_hops + 1):
                    for j in (i - d, i + d):
                        if 0 <= j < len(self._time_sorted_ids):
                            nid = self._time_sorted_ids[j]
                            if nid in self.memories:
                                candidates[nid] = max(candidates.get(nid, 0.0), candidates.get(mid, 0.0) * 0.6)

        scored = []
        import re
        q_terms = set(re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{1,}", (query or "").lower()))
        q_terms = {t for t in q_terms if len(t) > 1}
        for mid, base in candidates.items():
            entry = self.memories.get(mid)
            if entry is None:
                continue
            t_bonus = 0.0
            if time_range is not None:
                t_bonus = temporal_match_score(entry.timestamp, time_range, now=now)
            text = self._entry_text_for_embedding(entry)
            e_terms = set(re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{1,}", (text or "").lower()))
            e_terms = {t for t in e_terms if len(t) > 1}
            lex = 0.0
            if q_terms and e_terms:
                overlap = len(q_terms.intersection(e_terms))
                lex = overlap / max(1.0, (len(q_terms) ** 0.5))
                lex = float(min(1.0, lex))

            w_time = float(max(0.0, min(1.0, self.config.deep_time_weight)))
            w_lex = float(max(0.0, min(1.0, self.config.deep_lexical_weight)))
            w_base = float(max(0.0, 1.0 - w_time - w_lex))
            score = w_base * float(base) + w_time * float(t_bonus) + w_lex * float(lex)
            scored.append((score, entry))

        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        results: List[MemoryEntry] = []
        current_tokens = 0
        for _score, entry in scored:
            if current_tokens + entry.token_count <= budget:
                results.append(entry)
                current_tokens += entry.token_count
            if current_tokens >= budget:
                break
        return results
    
    def decay(self, now: datetime) -> int:
        """深层记忆不做自动衰减，但可以做归档"""
        # 这里可以实现基于时间的老化算法
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "layer": self.layer_name,
            "entries": len(self.memories),
            "entities": len(self.knowledge_graph),
            "access_count": self.access_count,
            "last_access": self.last_access_time.isoformat()
        }

    def vector_search(self, query: str, *, top_k: int = 10, distance: str = "cosine") -> List[Dict[str, Any]]:
        if not self.config.deep_enable_vector_index or self._encode_texts is None:
            return []
        qv = self._encode_texts([str(query or "")])[0]
        qv = _l2_normalize_vec(np.asarray(qv, dtype=np.float32))

        k = int(max(1, min(int(top_k), 100)))
        prefetch = int(max(k, int(getattr(self.config, "deep_vector_rerank_prefetch", 50))))
        dist = str(distance or "cosine").lower().strip()

        hits = []
        try:
            hits = self._vector_store.search(qv, limit=prefetch, filters={"distance": dist, "prefetch": prefetch})
        except Exception:
            hits = []

        candidates: List[MemoryEntry] = []
        seen = set()
        for h in hits:
            mid = str(getattr(h, "id", "") or "")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            entry = self.memories.get(mid)
            if entry is not None:
                candidates.append(entry)
            if len(candidates) >= prefetch:
                break

        def _score(entry: MemoryEntry) -> float:
            if entry.embedding is None:
                text = self._entry_text_for_embedding(entry)
                if text:
                    v = self._encode_texts([text])[0]
                    entry.embedding = _l2_normalize_vec(np.asarray(v, dtype=np.float32))
            if entry.embedding is None:
                return float("-inf")
            ev = np.asarray(entry.embedding, dtype=np.float32)
            if dist in {"cos", "cosine", "dot", "inner", "ip"}:
                return float(np.dot(qv, ev))
            if dist in {"l2", "euclid", "euclidean"}:
                d = qv - ev
                return -float(np.linalg.norm(d))
            return float(np.dot(qv, ev))

        scored = []
        for e in candidates:
            s = _score(e)
            if s == float("-inf"):
                continue
            scored.append((s, e))
        scored.sort(key=lambda x: (float(x[0]), x[1].timestamp), reverse=True)

        out: List[Dict[str, Any]] = []
        for s, e in scored[:k]:
            out.append(
                {
                    "id": str(e.id),
                    "score": float(s),
                    "memory": self._entry_text_for_embedding(e),
                    "timestamp": e.timestamp.isoformat() if isinstance(e.timestamp, datetime) else "",
                    "metadata": dict(e.metadata or {}),
                }
            )
        return out

    def _delete_entry(self, entry_id: str) -> bool:
        if entry_id not in self.memories:
            return False
        entry = self.memories.pop(entry_id, None)
        if entry is None:
            return False
        try:
            self._vector_store.delete([entry_id])
        except Exception:
            pass
        try:
            if entry_id in self._time_sorted_ids:
                self._time_sorted_ids.remove(entry_id)
        except Exception:
            pass
        for entity, ids in list(self.knowledge_graph.items()):
            if not ids:
                continue
            if entry_id in ids:
                self.knowledge_graph[entity] = [x for x in ids if x != entry_id]
            if not self.knowledge_graph[entity]:
                self.knowledge_graph.pop(entity, None)
        if self._conn is not None:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
            self._conn.commit()
        return True

    def optimize(
        self,
        *,
        deduplicate_exact: bool = True,
        compress: bool = True,
        similarity_threshold: float = 0.88,
        min_cluster_size: int = 3,
        max_clusters: int = 5,
        delete_sources: bool = True,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "dedup_checked": 0,
            "dedup_deleted": 0,
            "clusters_found": 0,
            "clusters_compressed": 0,
            "source_deleted": 0,
            "new_created": 0,
        }
        if deduplicate_exact:
            deleted = self._deduplicate_exact()
            stats["dedup_checked"] = len(self.memories)
            stats["dedup_deleted"] = deleted
        if compress:
            cstats = self._compress_similar(
                similarity_threshold=float(similarity_threshold),
                min_cluster_size=int(min_cluster_size),
                max_clusters=int(max_clusters),
                delete_sources=bool(delete_sources),
            )
            stats.update(cstats)
        return stats

    def _deduplicate_exact(self) -> int:
        if not self.memories:
            return 0
        groups: Dict[str, List[MemoryEntry]] = {}
        for e in self.memories.values():
            key = hashlib.md5((str(e.query or "").strip() + "\n" + str(e.response or "").strip()).encode("utf-8")).hexdigest()
            groups.setdefault(key, []).append(e)
        deleted = 0
        for _h, ents in groups.items():
            if len(ents) <= 1:
                continue
            ents.sort(key=lambda x: (x.timestamp, x.id))
            for dup in ents[1:]:
                if self._delete_entry(dup.id):
                    deleted += 1
        return deleted

    def _compress_similar(
        self,
        *,
        similarity_threshold: float,
        min_cluster_size: int,
        max_clusters: int,
        delete_sources: bool,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "clusters_found": 0,
            "clusters_compressed": 0,
            "source_deleted": 0,
            "new_created": 0,
        }
        candidates: List[MemoryEntry] = []
        for e in self.memories.values():
            if not e.query and not e.response:
                continue
            if e.metadata and e.metadata.get("type") == "compressed_summary":
                continue
            candidates.append(e)
        if not candidates:
            return stats
        candidates.sort(key=lambda x: x.id)

        for e in candidates:
            if e.embedding is not None:
                continue
            if self._encode_texts is None or not self.config.deep_enable_vector_index:
                continue
            try:
                text = self._entry_text_for_embedding(e)
                vec = self._encode_texts([text])[0]
                e.embedding = np.asarray(vec, dtype=np.float32)
                self._vector_store.upsert([e.id], [e.embedding], payloads=[e.metadata])
                self._persist_entry(e)
            except Exception:
                e.embedding = None

        embs = {e.id: e.embedding for e in candidates if e.embedding is not None}
        if len(embs) < max(2, int(min_cluster_size)):
            return stats

        processed = set()
        clusters: List[List[MemoryEntry]] = []
        for e in candidates:
            if e.id in processed:
                continue
            v = embs.get(e.id)
            if v is None:
                continue
            cluster = [e]
            processed.add(e.id)
            for other in candidates:
                if other.id in processed:
                    continue
                ov = embs.get(other.id)
                if ov is None:
                    continue
                sim = float(_cosine_similarity(v, ov))
                if sim >= float(similarity_threshold):
                    cluster.append(other)
                    processed.add(other.id)
            if len(cluster) >= int(min_cluster_size):
                clusters.append(cluster)
                if len(clusters) >= int(max_clusters):
                    break

        stats["clusters_found"] = len(clusters)
        if not clusters:
            return stats

        for cluster in clusters:
            source_ids = [e.id for e in cluster]
            summary_query, summary_resp = self._summarize_cluster(cluster)
            new_id = str(uuid.uuid4())
            ts = datetime.now()
            new_entry = MemoryEntry(
                id=new_id,
                query=summary_query,
                response=summary_resp,
                timestamp=ts,
                layer="deep",
                token_count=int(estimate_tokens(summary_query) + estimate_tokens(summary_resp)),
                metadata={
                    "created_at": ts.isoformat(),
                    "type": "compressed_summary",
                    "source_count": len(source_ids),
                    "source_ids": source_ids,
                },
            )
            self._maybe_compress_entry(new_entry)
            self.memories[new_entry.id] = new_entry
            self._time_sorted_ids.append(new_entry.id)
            self._time_sorted_ids.sort(key=lambda x: self.memories[x].timestamp)
            entities = self._extract_entities(f"{new_entry.query} {new_entry.response}")
            for entity in entities:
                self.knowledge_graph.setdefault(entity, []).append(new_entry.id)
            if self.config.deep_enable_vector_index and self._encode_texts is not None:
                try:
                    text = self._entry_text_for_embedding(new_entry)
                    vec = self._encode_texts([text])[0]
                    new_entry.embedding = np.asarray(vec, dtype=np.float32)
                    self._vector_store.upsert([new_entry.id], [new_entry.embedding], payloads=[new_entry.metadata])
                except Exception:
                    pass
            self._persist_entry(new_entry)
            stats["new_created"] += 1

            if delete_sources:
                for sid in source_ids:
                    if self._delete_entry(sid):
                        stats["source_deleted"] += 1
            stats["clusters_compressed"] += 1
        return stats

    def _summarize_cluster(self, cluster: List[MemoryEntry]) -> Tuple[str, str]:
        cluster_sorted = sorted(cluster, key=lambda x: (x.timestamp, x.id))
        texts = []
        for e in cluster_sorted:
            q = (e.query or "").strip()
            r = (e.response or "").strip()
            if not q and not r:
                continue
            texts.append(f"Q: {q}\nA: {r}".strip())
        combined = "\n\n".join(texts)
        comp = self._compressor.compress_text(combined)
        entities: List[str] = []
        for e in cluster_sorted:
            entities.extend(self._extract_entities(f"{e.query} {e.response}"))
            if len(entities) >= 8:
                break
        uniq = []
        seen = set()
        for t in entities:
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(t)
            if len(uniq) >= 6:
                break
        title = "压缩摘要"
        if uniq:
            title += ": " + ", ".join(uniq)
        return title, (comp.compressed_text or combined)


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n <= 0:
        return x
    return x / n


class MetaMemoryLayer(MemoryLayer):
    """元记忆：记录访问模式和层级调度"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config, "meta")
        self.access_log: List[Dict[str, Any]] = []
        self.layer_transitions: Dict[Tuple[str, str], int] = {}
        self.query_patterns: Dict[str, int] = {}
    
    def add(self, entry: MemoryEntry) -> bool:
        """记录访问日志"""
        log_entry = {
            "timestamp": entry.timestamp.isoformat(),
            "query": entry.query,
            "response_length": len(entry.response),
            "layer": entry.layer,
            "token_count": entry.token_count
        }
        self.access_log.append(log_entry)
        
        # 更新查询模式
        query_type = self._classify_query(entry.query)
        self.query_patterns[query_type] = self.query_patterns.get(query_type, 0) + 1
        
        self.update_access_info()
        return True
    
    def _classify_query(self, query: str) -> str:
        """简单的查询分类"""
        query_lower = query.lower()
        if "what" in query_lower or "how" in query_lower:
            return "informational"
        elif "why" in query_lower:
            return "explanatory"
        elif "help" in query_lower or "assist" in query_lower:
            return "assistance"
        else:
            return "general"
    
    def record_layer_transition(self, from_layer: str, to_layer: str):
        """记录层级转换"""
        key = (from_layer, to_layer)
        self.layer_transitions[key] = self.layer_transitions.get(key, 0) + 1
    
    def retrieve(self, query: str, budget: int) -> List[MemoryEntry]:
        """检索访问模式和统计信息"""
        self.update_access_info()
        
        # 这里可以返回一些统计信息作为记忆条目
        stats_entry = MemoryEntry(
            id=f"meta_{len(self.access_log)}",
            query=query,
            response=json.dumps(self.get_stats(), indent=2),
            timestamp=datetime.now(),
            layer=self.layer_name,
            token_count=100
        )
        
        return [stats_entry] if budget >= 100 else []
    
    def decay(self, now: datetime) -> int:
        """清理旧的访问日志"""
        cutoff_time = now.timestamp() - 86400  # 24小时前的日志
        original_count = len(self.access_log)
        
        self.access_log = [
            log for log in self.access_log 
            if datetime.fromisoformat(log["timestamp"]).timestamp() > cutoff_time
        ]
        
        return original_count - len(self.access_log)
    
    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """获取层级转移概率矩阵"""
        matrix = {}
        layers = ["shallow", "working", "deep", "meta"]
        
        for from_layer in layers:
            matrix[from_layer] = {}
            from_total = sum(self.layer_transitions.get((from_layer, to), 0) 
                           for to in layers)
            
            for to_layer in layers:
                count = self.layer_transitions.get((from_layer, to_layer), 0)
                matrix[from_layer][to_layer] = count / from_total if from_total > 0 else 0.0
        
        return matrix
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_tokens = sum(log.get("token_count", 0) for log in self.access_log)
        return {
            "layer": self.layer_name,
            "entries": len(self.access_log),
            "total_tokens": total_tokens,
            "total_accesses": len(self.access_log),
            "query_patterns": self.query_patterns,
            "layer_transitions": self.layer_transitions,
            "transition_matrix": self.get_transition_matrix(),
            "access_count": self.access_count,
            "last_access": self.last_access_time.isoformat()
        }
