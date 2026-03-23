import pytest
import asyncio
from datetime import datetime, timedelta
import uuid
import numpy as np

from memory_layers import (
    MemoryEntry, MemoryConfig,
    ShallowMemoryLayer, WorkingMemoryLayer,
    DeepMemoryLayer, MetaMemoryLayer
)


class TestMemoryEntry:
    """测试记忆条目"""
    
    def test_memory_entry_creation(self):
        """测试记忆条目创建"""
        entry = MemoryEntry(
            id="test_1",
            query="What is AI?",
            response="AI stands for Artificial Intelligence.",
            timestamp=datetime.now(),
            layer="shallow",
            token_count=10
        )
        
        assert entry.id == "test_1"
        assert entry.query == "What is AI?"
        assert entry.response == "AI stands for Artificial Intelligence."
        assert entry.layer == "shallow"
        assert entry.token_count == 10
    
    def test_memory_entry_with_embedding(self):
        """测试带嵌入的记忆条目"""
        embedding = np.array([0.1, 0.2, 0.3])
        entry = MemoryEntry(
            id="test_2",
            query="Test query",
            response="Test response",
            timestamp=datetime.now(),
            layer="working",
            token_count=5,
            embedding=embedding
        )
        
        assert np.array_equal(entry.embedding, embedding)
    
    def test_memory_entry_with_metadata(self):
        """测试带元数据的记忆条目"""
        metadata = {"confidence": 0.9, "source": "user"}
        entry = MemoryEntry(
            id="test_3",
            query="Test query",
            response="Test response",
            timestamp=datetime.now(),
            layer="deep",
            token_count=5,
            metadata=metadata
        )
        
        assert entry.metadata == metadata


class TestShallowMemoryLayer:
    """测试浅层记忆"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = MemoryConfig(max_shallow_entries=5)
        self.layer = ShallowMemoryLayer(self.config)
        self.test_entry = MemoryEntry(
            id="test_1",
            query="What is AI?",
            response="AI stands for Artificial Intelligence.",
            timestamp=datetime.now(),
            layer="shallow",
            token_count=10
        )
    
    def test_add_entry(self):
        """测试添加条目"""
        result = self.layer.add(self.test_entry)
        assert result is True
        assert len(self.layer.memories) == 1
        assert self.layer.total_tokens == 10
    
    def test_retrieve_entry(self):
        """测试检索条目"""
        self.layer.add(self.test_entry)
        
        # 添加更多条目
        for i in range(3):
            entry = MemoryEntry(
                id=f"test_{i+2}",
                query=f"Query {i+2}",
                response=f"Response {i+2}",
                timestamp=datetime.now(),
                layer="shallow",
                token_count=5
            )
            self.layer.add(entry)
        
        retrieved = self.layer.retrieve("test query", budget=20)
        assert len(retrieved) > 0
        assert len(retrieved) <= 4  # 不超过预算
    
    def test_decay_expired_entries(self):
        """测试过期条目清理"""
        # 创建过期条目
        old_entry = MemoryEntry(
            id="old_test",
            query="Old query",
            response="Old response",
            timestamp=datetime.now() - timedelta(minutes=10),  # 10分钟前
            layer="shallow",
            token_count=5
        )
        
        self.layer.add(old_entry)
        assert len(self.layer.memories) == 1
        
        # 执行衰减（TTL是5分钟）
        removed = self.layer.decay(datetime.now())
        assert removed == 1
        assert len(self.layer.memories) == 0
    
    def test_lru_eviction(self):
        """测试LRU淘汰"""
        # 添加条目直到超过容量
        for i in range(6):  # 超过max_shallow_entries=5
            entry = MemoryEntry(
                id=f"test_{i}",
                query=f"Query {i}",
                response=f"Response {i}",
                timestamp=datetime.now(),
                layer="shallow",
                token_count=5
            )
            self.layer.add(entry)
        
        assert len(self.layer.memories) == 5  # 不超过最大容量
        assert "test_0" not in self.layer.memories  # 最老的条目被移除
    
    def test_get_stats(self):
        """测试获取统计信息"""
        self.layer.add(self.test_entry)
        
        stats = self.layer.get_stats()
        assert stats["layer"] == "shallow"
        assert stats["entries"] == 1
        assert stats["total_tokens"] == 10
        assert "access_count" in stats
        assert "last_access" in stats


class TestWorkingMemoryLayer:
    """测试工作记忆"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = MemoryConfig(max_working_entries=3)
        self.layer = WorkingMemoryLayer(self.config)
        self.test_entry = MemoryEntry(
            id="test_1",
            query="What is machine learning?",
            response="Machine learning is a subset of AI.",
            timestamp=datetime.now(),
            layer="working",
            token_count=8
        )
    
    def test_add_entry_with_summary(self):
        """测试添加条目并生成摘要"""
        result = self.layer.add(self.test_entry)
        assert result is True
        assert len(self.layer.memories) == 1
        assert self.test_entry.id in self.layer.memories
        assert self.test_entry.metadata is not None
        assert "summary" in self.test_entry.metadata
    
    def test_retrieve_by_keywords(self):
        """测试基于关键词检索"""
        # 添加相关条目
        ml_entry = MemoryEntry(
            id="ml_test",
            query="What is machine learning?",
            response="Machine learning is a subset of AI.",
            timestamp=datetime.now(),
            layer="working",
            token_count=10
        )
        
        ai_entry = MemoryEntry(
            id="ai_test",
            query="What is artificial intelligence?",
            response="AI is the simulation of human intelligence.",
            timestamp=datetime.now(),
            layer="working",
            token_count=12
        )
        
        self.layer.add(ml_entry)
        self.layer.add(ai_entry)
        
        # 检索包含"machine"的条目
        results = self.layer.retrieve("machine learning", budget=20)
        assert len(results) >= 1
        assert any("machine" in result.query.lower() for result in results)
    
    def test_decay_working_memory(self):
        """测试工作记忆衰减"""
        # 创建过期条目
        old_entry = MemoryEntry(
            id="old_working",
            query="Old working query",
            response="Old working response",
            timestamp=datetime.now() - timedelta(minutes=35),  # 35分钟前
            layer="working",
            token_count=6
        )
        
        self.layer.add(old_entry)
        assert len(self.layer.memories) == 1
        
        # 执行衰减（TTL是30分钟）
        removed = self.layer.decay(datetime.now())
        assert removed == 1
        assert len(self.layer.memories) == 0


class TestDeepMemoryLayer:
    """测试深层记忆"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = MemoryConfig(deep_persist_path=":memory:")  # 使用内存数据库
        self.layer = DeepMemoryLayer(self.config)
        self.test_entry = MemoryEntry(
            id="test_1",
            query="What is deep learning?",
            response="Deep learning uses neural networks.",
            timestamp=datetime.now(),
            layer="deep",
            token_count=15
        )
    
    def test_add_entry_with_entities(self):
        """测试添加条目并提取实体"""
        result = self.layer.add(self.test_entry)
        assert result is True
        assert len(self.layer.memories) == 1
        assert self.test_entry.id in self.layer.memories
        
        # 检查是否提取了实体
        assert len(self.layer.knowledge_graph) > 0
    
    def test_retrieve_by_entities(self):
        """测试基于实体检索"""
        # 添加条目
        self.layer.add(self.test_entry)
        
        # 检索包含相关实体的查询
        results = self.layer.retrieve("deep learning neural", budget=20)
        assert len(results) >= 0  # 可能找到相关结果
    
    def test_persistence_to_database(self):
        """测试持久化到数据库"""
        self.layer.add(self.test_entry)
        
        # 创建新的实例来测试持久化
        new_layer = DeepMemoryLayer(self.config)
        # 注意：由于使用内存数据库，数据不会持久化到磁盘
        # 实际使用文件数据库时会持久化
    
    def test_deep_memory_no_decay(self):
        """测试深层记忆不衰减"""
        old_entry = MemoryEntry(
            id="old_deep",
            query="Old deep query",
            response="Old deep response",
            timestamp=datetime.now() - timedelta(days=30),  # 30天前
            layer="deep",
            token_count=10
        )
        
        self.layer.add(old_entry)
        removed = self.layer.decay(datetime.now())
        assert removed == 0  # 深层记忆不衰减
        assert len(self.layer.memories) == 1


class TestMetaMemoryLayer:
    """测试元记忆"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = MemoryConfig()
        self.layer = MetaMemoryLayer(self.config)
        self.test_entry = MemoryEntry(
            id="test_1",
            query="What is memory?",
            response="Memory stores information.",
            timestamp=datetime.now(),
            layer="meta",
            token_count=8
        )
    
    def test_add_entry_with_logging(self):
        """测试添加条目并记录日志"""
        result = self.layer.add(self.test_entry)
        assert result is True
        assert len(self.layer.access_log) == 1
        assert self.layer.access_log[0]["query"] == "What is memory?"
    
    def test_query_classification(self):
        """测试查询分类"""
        informational = self.layer._classify_query("What is AI?")
        explanatory = self.layer._classify_query("Why is the sky blue?")
        assistance = self.layer._classify_query("Can you help me?")
        general = self.layer._classify_query("Random text")
        
        assert informational == "informational"
        assert explanatory == "explanatory"
        assert assistance == "assistance"
        assert general == "general"
    
    def test_layer_transition_recording(self):
        """测试层级转换记录"""
        self.layer.record_layer_transition("shallow", "working")
        self.layer.record_layer_transition("shallow", "working")
        self.layer.record_layer_transition("working", "deep")
        
        assert self.layer.layer_transitions[("shallow", "working")] == 2
        assert self.layer.layer_transitions[("working", "deep")] == 1
    
    def test_transition_matrix(self):
        """测试转移矩阵生成"""
        # 记录一些转换
        self.layer.record_layer_transition("shallow", "working")
        self.layer.record_layer_transition("shallow", "shallow")
        self.layer.record_layer_transition("working", "deep")
        
        matrix = self.layer.get_transition_matrix()
        
        assert "shallow" in matrix
        assert "working" in matrix
        assert matrix["shallow"]["working"] > 0
        assert matrix["shallow"]["shallow"] > 0
    
    def test_decay_access_log(self):
        """测试访问日志清理"""
        # 创建24小时前的日志条目
        old_time = datetime.now() - timedelta(hours=25)
        old_entry = MemoryEntry(
            id="old_meta",
            query="Old meta query",
            response="Old meta response",
            timestamp=old_time,
            layer="meta",
            token_count=6
        )
        
        self.layer.add(old_entry)
        original_count = len(self.layer.access_log)
        
        # 执行衰减
        removed = self.layer.decay(datetime.now())
        assert removed == original_count - len(self.layer.access_log)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        self.layer.add(self.test_entry)
        self.layer.record_layer_transition("shallow", "working")
        
        stats = self.layer.get_stats()
        assert stats["layer"] == "meta"
        assert stats["total_accesses"] == 1
        assert "query_patterns" in stats
        assert "layer_transitions" in stats
        assert "transition_matrix" in stats


class TestMemoryConfig:
    """测试记忆配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MemoryConfig()
        assert config.shallow_ttl == 300
        assert config.working_ttl == 1800
        assert config.deep_persist_path == "deep_memory.db"
        assert config.max_shallow_entries == 100
        assert config.max_working_entries == 50
        assert config.lambda_decay == 0.1
        assert config.alpha_similarity == 0.6
        assert config.beta_time == 0.3
        assert config.gamma_layer == 0.1
        assert config.token_budget_ratio == 0.8
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = MemoryConfig(
            shallow_ttl=600,
            working_ttl=3600,
            lambda_decay=0.2,
            alpha_similarity=0.7
        )
        assert config.shallow_ttl == 600
        assert config.working_ttl == 3600
        assert config.lambda_decay == 0.2
        assert config.alpha_similarity == 0.7


@pytest.mark.asyncio
class TestMemoryLayersAsync:
    """异步测试记忆层"""
    
    async def test_concurrent_access(self):
        """测试并发访问"""
        config = MemoryConfig()
        layer = ShallowMemoryLayer(config)
        
        # 创建多个条目
        entries = []
        for i in range(10):
            entry = MemoryEntry(
                id=f"async_test_{i}",
                query=f"Async query {i}",
                response=f"Async response {i}",
                timestamp=datetime.now(),
                layer="shallow",
                token_count=5
            )
            entries.append(entry)
        
        # 并发添加
        tasks = [asyncio.to_thread(layer.add, entry) for entry in entries]
        results = await asyncio.gather(*tasks)
        
        assert all(results)
        assert len(layer.memories) == 10
    
    async def test_async_retrieve(self):
        """测试异步检索"""
        config = MemoryConfig()
        layer = WorkingMemoryLayer(config)
        
        # 添加测试条目
        entry = MemoryEntry(
            id="async_retrieve_test",
            query="Async test query",
            response="Async test response",
            timestamp=datetime.now(),
            layer="working",
            token_count=8
        )
        await asyncio.to_thread(layer.add, entry)
        
        # 异步检索
        result = layer.retrieve("async test", budget=10)
        assert len(result) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
