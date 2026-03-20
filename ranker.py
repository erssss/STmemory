from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
import hashlib

from memory_layers import MemoryEntry, MemoryLayer, MemoryConfig


@dataclass
class ScoredMemory:
    """带评分的记忆条目"""
    entry: MemoryEntry
    score: float
    similarity_score: float
    time_score: float
    layer_score: float
    layer_transition_prob: float


class _HashEmbeddingModel:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                idx = int(digest[:8], 16) % self.dim
                vectors[i, idx] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class SpatioTemporalRanker:
    """时空记忆排序器"""
    
    def __init__(self, config: MemoryConfig, model_name: str = "all-MiniLM-L6-v2"):
        self.config = config
        self.model_name = model_name
        self.model = None
        self._load_model()
        self.layer_transition_matrix = self._init_transition_matrix()
    
    def _load_model(self):
        """加载句子嵌入模型"""
        try:
            device = os.getenv("STMEM_DEVICE")
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
        except Exception as e:
            print(f"Warning: Failed to load sentence transformer model: {e}")
            self.model = _HashEmbeddingModel()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            return _HashEmbeddingModel().encode(texts)
        try:
            vecs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return np.asarray(vecs, dtype=np.float32)
        except TypeError:
            vecs = self.model.encode(texts)
            return np.asarray(vecs, dtype=np.float32)
    
    def _init_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """初始化层级转移概率矩阵"""
        layers = ["shallow", "working", "deep", "meta"]
        matrix = {}
        
        for from_layer in layers:
            matrix[from_layer] = {}
            for to_layer in layers:
                # 基础转移概率：相邻层级更高概率
                if from_layer == to_layer:
                    matrix[from_layer][to_layer] = 0.6  # 自环概率
                elif self._are_adjacent_layers(from_layer, to_layer):
                    matrix[from_layer][to_layer] = 0.3
                else:
                    matrix[from_layer][to_layer] = 0.1
        
        return matrix
    
    def _are_adjacent_layers(self, layer1: str, layer2: str) -> bool:
        """检查两个层级是否相邻"""
        layer_order = ["shallow", "working", "deep", "meta"]
        idx1 = layer_order.index(layer1)
        idx2 = layer_order.index(layer2)
        return abs(idx1 - idx2) == 1
    
    def compute_time_decay(self, entry: MemoryEntry, current_time: datetime) -> float:
        """计算时间衰减权重"""
        time_diff = (current_time - entry.timestamp).total_seconds()
        # 使用指数衰减：w(t) = exp(-λ * Δt)
        decay_weight = math.exp(-self.config.lambda_decay * time_diff)
        return decay_weight
    
    def compute_semantic_similarity(self, query: str, entry: MemoryEntry) -> float:
        """计算语义相似度"""
        if self.model is None:
            # 回退到简单的关键词匹配
            return self._keyword_similarity(query, f"{entry.query} {entry.response}")
        
        try:
            # 生成嵌入向量
            query_embedding = self.model.encode([query])
            entry_text = f"{entry.query} {entry.response}"
            
            if entry.embedding is not None:
                entry_embedding = entry.embedding.reshape(1, -1)
            else:
                entry_embedding = self.model.encode([entry_text])
                entry.embedding = entry_embedding[0]  # 缓存嵌入
            
            # 计算余弦相似度
            similarity = cosine_similarity(query_embedding, entry_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Warning: Error computing semantic similarity: {e}")
            return self._keyword_similarity(query, f"{entry.query} {entry.response}")
    
    def _keyword_similarity(self, query: str, text: str) -> float:
        """简单的关键词相似度"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words)
    
    def get_layer_transition_probability(self, from_layer: str, to_layer: str) -> float:
        """获取层级转移概率"""
        return self.layer_transition_matrix.get(from_layer, {}).get(to_layer, 0.1)
    
    def compute_spatiotemporal_score(
        self, 
        query: str, 
        entry: MemoryEntry, 
        current_time: datetime,
        current_layer: str
    ) -> ScoredMemory:
        """计算时空综合评分"""
        # 计算各个分量
        similarity_score = self.compute_semantic_similarity(query, entry)
        time_score = self.compute_time_decay(entry, current_time)
        layer_score = self.get_layer_transition_probability(current_layer, entry.layer)
        
        # 综合评分：score = α·语义相似度 + β·时间衰减 + γ·层级转移
        total_score = (
            self.config.alpha_similarity * similarity_score +
            self.config.beta_time * time_score +
            self.config.gamma_layer * layer_score
        )
        
        return ScoredMemory(
            entry=entry,
            score=total_score,
            similarity_score=similarity_score,
            time_score=time_score,
            layer_score=layer_score,
            layer_transition_prob=layer_score
        )
    
    def rank_memories(
        self,
        query: str,
        memories: List[MemoryEntry],
        current_time: datetime,
        current_layer: str = "shallow"
    ) -> List[ScoredMemory]:
        """对记忆条目进行时空排序"""
        scored_memories = []
        
        for entry in memories:
            scored_memory = self.compute_spatiotemporal_score(
                query, entry, current_time, current_layer
            )
            scored_memories.append(scored_memory)
        
        # 按综合评分降序排序
        scored_memories.sort(key=lambda x: x.score, reverse=True)
        
        return scored_memories
    
    def select_memories_within_budget(
        self,
        query: str,
        all_memories: Dict[str, List[MemoryEntry]],
        budget: int,
        current_time: datetime
    ) -> List[MemoryEntry]:
        """在token预算内选择最优记忆组合"""
        selected_memories = []
        current_tokens = 0
        
        # 从所有层级收集记忆
        all_entries = []
        for layer_name, memories in all_memories.items():
            for memory in memories:
                all_entries.append((layer_name, memory))

        if not all_entries:
            return []
        
        if self.model is None:
            layer_scores: Dict[str, ScoredMemory] = {}
            for layer_name, entry in all_entries:
                scored_memory = self.compute_spatiotemporal_score(query, entry, current_time, layer_name)
                layer_scores[entry.id] = scored_memory
        else:
            query_vec = self._encode_texts([query]).reshape(1, -1)

            missing_entries: List[MemoryEntry] = []
            missing_texts: List[str] = []
            for _layer_name, entry in all_entries:
                if entry.embedding is None:
                    missing_entries.append(entry)
                    missing_texts.append(f"{entry.query} {entry.response}")

            if missing_texts:
                enc = self._encode_texts(missing_texts)
                for entry, vec in zip(missing_entries, enc):
                    entry.embedding = vec

            matrix = np.asarray([entry.embedding for _layer_name, entry in all_entries], dtype=np.float32)
            sims = cosine_similarity(query_vec, matrix)[0]

            layer_scores = {}
            for (layer_name, entry), sim in zip(all_entries, sims):
                similarity_score = float(sim)
                time_score = self.compute_time_decay(entry, current_time)
                layer_score = self.get_layer_transition_probability(layer_name, entry.layer)
                total_score = (
                    self.config.alpha_similarity * similarity_score +
                    self.config.beta_time * time_score +
                    self.config.gamma_layer * layer_score
                )
                layer_scores[entry.id] = ScoredMemory(
                    entry=entry,
                    score=total_score,
                    similarity_score=similarity_score,
                    time_score=time_score,
                    layer_score=layer_score,
                    layer_transition_prob=layer_score,
                )
        
        # 按评分排序
        sorted_entries = sorted(
            all_entries,
            key=lambda x: layer_scores[x[1].id].score,
            reverse=True
        )
        
        # 贪心选择，直到预算用完
        for layer_name, entry in sorted_entries:
            if current_tokens + entry.token_count <= budget:
                selected_memories.append(entry)
                current_tokens += entry.token_count
            else:
                break
        
        # 按时间排序以保持对话连贯性
        selected_memories.sort(key=lambda x: x.timestamp)
        
        return selected_memories
    
    def update_transition_matrix(self, meta_layer: 'MetaMemoryLayer'):
        """基于实际访问数据更新转移概率矩阵"""
        transition_matrix = meta_layer.get_transition_matrix()
        if transition_matrix:
            self.layer_transition_matrix = transition_matrix
    
    def get_stats(self) -> Dict[str, Any]:
        """获取排序器统计信息"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "lambda_decay": self.config.lambda_decay,
            "alpha_similarity": self.config.alpha_similarity,
            "beta_time": self.config.beta_time,
            "gamma_layer": self.config.gamma_layer,
            "transition_matrix": self.layer_transition_matrix
        }
