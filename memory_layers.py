from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import json
import os
from collections import OrderedDict


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
    
    def _generate_summary(self, query: str, response: str) -> str:
        """生成对话摘要"""
        # 简单的摘要生成逻辑，实际可以使用更复杂的算法
        combined = f"Q: {query}\nA: {response}"
        if len(combined) > 200:
            return combined[:200] + "..."
        return combined
    
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
        self.memories: Dict[str, MemoryEntry] = {}
        self.knowledge_graph: Dict[str, List[str]] = {}
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        import sqlite3
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                query TEXT,
                response TEXT,
                timestamp TEXT,
                layer TEXT,
                token_count INTEGER,
                embedding BLOB,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                entity TEXT PRIMARY KEY,
                relations TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _extract_entities(self, text: str) -> List[str]:
        """简单的实体提取"""
        # 这里可以使用更复杂的NLP技术
        words = text.lower().split()
        entities = [word for word in words if len(word) > 3]
        return entities[:10]  # 限制实体数量
    
    def add(self, entry: MemoryEntry) -> bool:
        """添加记忆条目，构建知识图谱"""
        self.memories[entry.id] = entry
        
        # 提取实体
        entities = self._extract_entities(f"{entry.query} {entry.response}")
        
        # 更新知识图谱
        for entity in entities:
            if entity not in self.knowledge_graph:
                self.knowledge_graph[entity] = []
            self.knowledge_graph[entity].append(entry.id)
        
        # 持久化到数据库
        self._persist_entry(entry)
        self.update_access_info()
        return True
    
    def _persist_entry(self, entry: MemoryEntry):
        """持久化条目到数据库"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = entry.embedding.tobytes() if entry.embedding is not None else None
        metadata_json = json.dumps(entry.metadata) if entry.metadata else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO memories 
            (id, query, response, timestamp, layer, token_count, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (entry.id, entry.query, entry.response, entry.timestamp.isoformat(),
              entry.layer, entry.token_count, embedding_blob, metadata_json))
        
        conn.commit()
        conn.close()
    
    def retrieve(self, query: str, budget: int) -> List[MemoryEntry]:
        """基于实体和语义检索"""
        self.update_access_info()
        
        # 提取查询实体
        query_entities = self._extract_entities(query)
        
        # 基于实体检索
        relevant_entries = set()
        for entity in query_entities:
            if entity in self.knowledge_graph:
                for entry_id in self.knowledge_graph[entity]:
                    if entry_id in self.memories:
                        relevant_entries.add(self.memories[entry_id])
        
        # 按时间排序并限制预算
        sorted_entries = sorted(relevant_entries, key=lambda x: x.timestamp, reverse=True)
        
        results = []
        current_tokens = 0
        for entry in sorted_entries:
            if current_tokens + entry.token_count <= budget:
                results.append(entry)
                current_tokens += entry.token_count
            else:
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
        return {
            "layer": self.layer_name,
            "total_accesses": len(self.access_log),
            "query_patterns": self.query_patterns,
            "layer_transitions": self.layer_transitions,
            "transition_matrix": self.get_transition_matrix(),
            "access_count": self.access_count,
            "last_access": self.last_access_time.isoformat()
        }