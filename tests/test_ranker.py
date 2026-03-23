import pytest
import numpy as np
from datetime import datetime, timedelta
import math

from ranker import SpatioTemporalRanker, ScoredMemory
from memory_layers import MemoryEntry, MemoryConfig


class TestSpatioTemporalRanker:
    """测试时空排序器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = MemoryConfig(
            lambda_decay=0.1,
            alpha_similarity=0.6,
            beta_time=0.3,
            gamma_layer=0.1
        )
        self.ranker = SpatioTemporalRanker(self.config)
        
        # 创建测试记忆条目
        self.test_entry = MemoryEntry(
            id="test_1",
            query="What is artificial intelligence?",
            response="AI is the simulation of human intelligence.",
            timestamp=datetime.now(),
            layer="shallow",
            token_count=15
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.ranker.config == self.config
        assert self.ranker.model_name == "all-MiniLM-L6-v2"
        assert self.ranker.model is not None  # 模型应该被加载
        assert "shallow" in self.ranker.layer_transition_matrix
        assert "working" in self.ranker.layer_transition_matrix
    
    def test_time_decay_calculation(self):
        """测试时间衰减计算"""
        current_time = datetime.now()
        entry_time = current_time - timedelta(seconds=10)  # 10秒前
        
        # 修改条目时间
        self.test_entry.timestamp = entry_time
        
        decay_weight = self.ranker.compute_time_decay(self.test_entry, current_time)
        
        # 检查衰减权重是否在合理范围内
        assert 0 < decay_weight <= 1
        
        # 检查指数衰减公式: w(t) = exp(-λ * Δt)
        expected_decay = math.exp(-self.config.lambda_decay * 10)
        assert abs(decay_weight - expected_decay) < 0.01
    
    def test_semantic_similarity_with_model(self):
        """测试语义相似度计算（使用模型）"""
        query = "What is AI?"
        similarity = self.ranker.compute_semantic_similarity(query, self.test_entry)
        
        # 相似度应该在0到1之间
        assert 0 <= similarity <= 1
        
        # 相似查询应该有较高的相似度
        high_similarity_query = "What is artificial intelligence exactly?"
        high_similarity = self.ranker.compute_semantic_similarity(high_similarity_query, self.test_entry)
        
        # 不相似查询应该有较低的相似度
        low_similarity_query = "What is the weather today?"
        low_similarity = self.ranker.compute_semantic_similarity(low_similarity_query, self.test_entry)
        
        assert high_similarity > low_similarity
    
    def test_keyword_similarity_fallback(self):
        """测试关键词相似度回退"""
        # 临时禁用模型以测试回退
        original_model = self.ranker.model
        self.ranker.model = None
        
        try:
            query = "What is AI?"
            similarity = self.ranker.compute_semantic_similarity(query, self.test_entry)
            
            # 关键词匹配应该返回合理的相似度
            assert 0 <= similarity <= 1
            
            # 包含关键词的查询应该有更高的相似度
            keyword_query = "artificial intelligence machine learning"
            keyword_similarity = self.ranker.compute_semantic_similarity(keyword_query, self.test_entry)
            
            assert keyword_similarity > 0
            
        finally:
            # 恢复模型
            self.ranker.model = original_model
    
    def test_layer_transition_probability(self):
        """测试层级转移概率"""
        # 测试自环概率
        self_loop_prob = self.ranker.get_layer_transition_probability("shallow", "shallow")
        assert self_loop_prob == 0.6
        
        # 测试相邻层级概率
        adjacent_prob = self.ranker.get_layer_transition_probability("shallow", "working")
        assert adjacent_prob == 0.3
        
        # 测试非相邻层级概率
        non_adjacent_prob = self.ranker.get_layer_transition_probability("shallow", "deep")
        assert non_adjacent_prob == 0.1
    
    def test_spatiotemporal_score_computation(self):
        """测试时空综合评分计算"""
        query = "What is artificial intelligence?"
        current_time = datetime.now()
        current_layer = "shallow"
        
        scored_memory = self.ranker.compute_spatiotemporal_score(
            query, self.test_entry, current_time, current_layer
        )
        
        assert isinstance(scored_memory, ScoredMemory)
        assert scored_memory.entry == self.test_entry
        assert 0 <= scored_memory.similarity_score <= 1
        assert 0 <= scored_memory.time_score <= 1
        assert 0 <= scored_memory.layer_score <= 1
        assert scored_memory.score > 0  # 综合评分应该为正
        
        # 验证评分公式: score = α·similarity + β·time + γ·layer
        expected_score = (
            self.config.alpha_similarity * scored_memory.similarity_score +
            self.config.beta_time * scored_memory.time_score +
            self.config.gamma_layer * scored_memory.layer_score
        )
        assert abs(scored_memory.score - expected_score) < 0.01
    
    def test_rank_memories(self):
        """测试记忆排序"""
        # 创建多个测试条目
        entries = [
            MemoryEntry(
                id=f"test_{i}",
                query=f"Query about topic {i}",
                response=f"Response about topic {i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                layer="shallow" if i % 2 == 0 else "working",
                token_count=10
            )
            for i in range(5)
        ]
        
        query = "Query about topic 0"
        current_time = datetime.now()
        
        scored_memories = self.ranker.rank_memories(query, entries, current_time)
        
        assert len(scored_memories) == 5
        
        # 检查是否按评分降序排序
        for i in range(1, len(scored_memories)):
            assert scored_memories[i-1].score >= scored_memories[i].score
    
    def test_select_memories_within_budget(self):
        """测试预算内记忆选择"""
        # 创建测试记忆数据
        all_memories = {
            "shallow": [
                MemoryEntry(
                    id="shallow_1",
                    query="Shallow query 1",
                    response="Shallow response 1",
                    timestamp=datetime.now(),
                    layer="shallow",
                    token_count=10
                ),
                MemoryEntry(
                    id="shallow_2",
                    query="Shallow query 2",
                    response="Shallow response 2",
                    timestamp=datetime.now() - timedelta(minutes=1),
                    layer="shallow",
                    token_count=15
                )
            ],
            "working": [
                MemoryEntry(
                    id="working_1",
                    query="Working query 1",
                    response="Working response 1",
                    timestamp=datetime.now() - timedelta(minutes=2),
                    layer="working",
                    token_count=20
                )
            ]
        }
        
        query = "test query"
        budget = 25  # 足够容纳一些条目
        current_time = datetime.now()
        
        selected_memories = self.ranker.select_memories_within_budget(
            query, all_memories, budget, current_time
        )
        
        # 检查是否在预算内
        total_tokens = sum(memory.token_count for memory in selected_memories)
        assert total_tokens <= budget
        
        # 检查是否按时间排序
        for i in range(1, len(selected_memories)):
            assert selected_memories[i-1].timestamp <= selected_memories[i].timestamp
    
    def test_empty_memories_selection(self):
        """测试空记忆选择"""
        all_memories = {"shallow": [], "working": [], "deep": []}
        query = "test query"
        budget = 100
        current_time = datetime.now()
        
        selected_memories = self.ranker.select_memories_within_budget(
            query, all_memories, budget, current_time
        )
        
        assert len(selected_memories) == 0
    
    def test_very_small_budget_selection(self):
        """测试极小预算选择"""
        all_memories = {
            "shallow": [
                MemoryEntry(
                    id="test_1",
                    query="Test query",
                    response="Test response",
                    timestamp=datetime.now(),
                    layer="shallow",
                    token_count=50  # 大于预算
                )
            ]
        }
        
        query = "test query"
        budget = 10  # 小于单个条目
        current_time = datetime.now()
        
        selected_memories = self.ranker.select_memories_within_budget(
            query, all_memories, budget, current_time
        )
        
        assert len(selected_memories) == 0  # 预算太小，无法选择任何条目
    
    def test_update_transition_matrix(self):
        """测试转移矩阵更新"""
        # 创建模拟的元记忆层数据
        class MockMetaLayer:
            def get_transition_matrix(self):
                return {
                    "shallow": {"shallow": 0.8, "working": 0.15, "deep": 0.05},
                    "working": {"shallow": 0.2, "working": 0.7, "deep": 0.1}
                }
        
        mock_meta = MockMetaLayer()
        original_matrix = self.ranker.layer_transition_matrix.copy()
        
        self.ranker.update_transition_matrix(mock_meta)
        
        # 检查矩阵是否被更新
        updated_matrix = self.ranker.layer_transition_matrix
        assert updated_matrix != original_matrix
        assert updated_matrix["shallow"]["shallow"] == 0.8
    
    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.ranker.get_stats()
        
        assert "model_name" in stats
        assert "model_loaded" in stats
        assert "lambda_decay" in stats
        assert "alpha_similarity" in stats
        assert "beta_time" in stats
        assert "gamma_layer" in stats
        assert "transition_matrix" in stats
        
        assert stats["model_name"] == "all-MiniLM-L6-v2"
        assert stats["lambda_decay"] == self.config.lambda_decay
        assert stats["alpha_similarity"] == self.config.alpha_similarity
        assert isinstance(stats["transition_matrix"], dict)
    
    def test_adjacent_layer_detection(self):
        """测试相邻层级检测"""
        # 测试相邻层级
        assert self.ranker._are_adjacent_layers("shallow", "working") is True
        assert self.ranker._are_adjacent_layers("working", "deep") is True
        
        # 测试非相邻层级
        assert self.ranker._are_adjacent_layers("shallow", "deep") is False
        assert self.ranker._are_adjacent_layers("shallow", "meta") is False
        
        # 测试相同层级
        assert self.ranker._are_adjacent_layers("shallow", "shallow") is False


class TestScoredMemory:
    """测试评分记忆条目"""
    
    def test_scored_memory_creation(self):
        """测试评分记忆条目创建"""
        entry = MemoryEntry(
            id="test_1",
            query="Test query",
            response="Test response",
            timestamp=datetime.now(),
            layer="shallow",
            token_count=10
        )
        
        scored_memory = ScoredMemory(
            entry=entry,
            score=0.85,
            similarity_score=0.9,
            time_score=0.8,
            layer_score=0.7,
            layer_transition_prob=0.7
        )
        
        assert scored_memory.entry == entry
        assert scored_memory.score == 0.85
        assert scored_memory.similarity_score == 0.9
        assert scored_memory.time_score == 0.8
        assert scored_memory.layer_score == 0.7
        assert scored_memory.layer_transition_prob == 0.7
    
    def test_scored_memory_comparison(self):
        """测试评分记忆条目比较"""
        entry1 = MemoryEntry(
            id="test_1",
            query="Query 1",
            response="Response 1",
            timestamp=datetime.now(),
            layer="shallow",
            token_count=10
        )
        
        entry2 = MemoryEntry(
            id="test_2",
            query="Query 2",
            response="Response 2",
            timestamp=datetime.now(),
            layer="shallow",
            token_count=10
        )
        
        scored_memory1 = ScoredMemory(
            entry=entry1,
            score=0.9,
            similarity_score=0.9,
            time_score=0.8,
            layer_score=0.7,
            layer_transition_prob=0.7
        )
        
        scored_memory2 = ScoredMemory(
            entry=entry2,
            score=0.7,
            similarity_score=0.7,
            time_score=0.6,
            layer_score=0.5,
            layer_transition_prob=0.5
        )
        
        # 评分高的应该排在前面
        memories = [scored_memory2, scored_memory1]
        memories.sort(key=lambda x: x.score, reverse=True)
        
        assert memories[0] == scored_memory1
        assert memories[1] == scored_memory2


@pytest.mark.integration
class TestRankerIntegration:
    """集成测试时空排序器"""
    
    def test_full_ranking_pipeline(self):
        """测试完整排序管道"""
        config = MemoryConfig(
            lambda_decay=0.1,
            alpha_similarity=0.6,
            beta_time=0.3,
            gamma_layer=0.1
        )
        ranker = SpatioTemporalRanker(config)
        
        # 创建多样化的测试数据
        memories = [
            MemoryEntry(
                id="recent_relevant",
                query="What is machine learning?",
                response="Machine learning is a subset of AI.",
                timestamp=datetime.now() - timedelta(minutes=1),
                layer="shallow",
                token_count=15
            ),
            MemoryEntry(
                id="old_relevant",
                query="What is artificial intelligence?",
                response="AI simulates human intelligence.",
                timestamp=datetime.now() - timedelta(hours=2),
                layer="working",
                token_count=15
            ),
            MemoryEntry(
                id="recent_irrelevant",
                query="What is the weather?",
                response="The weather is sunny today.",
                timestamp=datetime.now() - timedelta(minutes=2),
                layer="shallow",
                token_count=10
            ),
            MemoryEntry(
                id="deep_relevant",
                query="How does machine learning work?",
                response="ML uses algorithms to learn from data.",
                timestamp=datetime.now() - timedelta(minutes=30),
                layer="deep",
                token_count=20
            )
        ]
        
        query = "Tell me about machine learning"
        current_time = datetime.now()
        
        # 执行排序
        scored_memories = ranker.rank_memories(query, memories, current_time)
        
        # 验证结果
        assert len(scored_memories) == 4
        
        # 最相关的记忆应该排在前面（综合考虑时间、相关性和层级）
        top_memory = scored_memories[0]
        assert top_memory.score > 0
        
        # 输出排序结果用于调试
        print("\nRanking results:")
        for i, scored in enumerate(scored_memories):
            print(f"{i+1}. {scored.entry.id}: score={scored.score:.3f}, "
                  f"similarity={scored.similarity_score:.3f}, "
                  f"time={scored.time_score:.3f}, layer={scored.entry.layer}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])