import pytest
from datetime import datetime

from budget import BudgetController, ModelConfig


class TestModelConfig:
    """测试模型配置"""
    
    def test_model_config_creation(self):
        """测试模型配置创建"""
        config = ModelConfig(
            name="test-model",
            max_tokens=4096,
            context_window=4096,
            cost_per_1k_tokens=0.001,
            optimal_budget_ratio=0.8
        )
        
        assert config.name == "test-model"
        assert config.max_tokens == 4096
        assert config.context_window == 4096
        assert config.cost_per_1k_tokens == 0.001
        assert config.optimal_budget_ratio == 0.8


class TestBudgetController:
    """测试预算控制器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.controller = BudgetController(
            model_name="openclaw-medium",
            system_prompt_ratio=0.1
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.controller.model_name == "openclaw-medium"
        assert self.controller.system_prompt_ratio == 0.1
        assert self.controller.model_config.name == "openclaw-medium"
        assert self.controller.model_config.max_tokens == 8192
        assert self.controller.model_config.optimal_budget_ratio == 0.75
    
    def test_get_model_config_existing(self):
        """测试获取现有模型配置"""
        config = self.controller._get_model_config("gpt-4")
        assert config.name == "gpt-4"
        assert config.max_tokens == 8192
        assert config.cost_per_1k_tokens == 0.03
    
    def test_get_model_config_unknown(self):
        """测试获取未知模型配置"""
        config = self.controller._get_model_config("unknown-model")
        assert config.name == "unknown-model"
        assert config.max_tokens == 4096  # 默认值
        assert config.optimal_budget_ratio == 0.8  # 默认值
    
    def test_calculate_memory_budget_basic(self):
        """测试基础记忆预算计算"""
        query = "What is artificial intelligence?"
        budget = self.controller.calculate_memory_budget(query)
        
        # 基础预算应该是模型最大token数的配置比例
        expected_base = int(8192 * 0.75)  # openclaw-medium的optimal_budget_ratio
        expected_system = int(8192 * 0.1)  # system_prompt_ratio
        expected_response = int(expected_base * 0.3)  # 默认响应预算
        expected_memory = expected_base - expected_system - expected_response
        
        assert budget >= 100  # 最小预算
        assert budget <= expected_base
    
    def test_calculate_memory_budget_with_estimate(self):
        """测试带响应长度估算的记忆预算"""
        query = "List all the benefits of machine learning"
        response_estimate = 200
        budget = self.controller.calculate_memory_budget(query, response_estimate)
        
        # 应该为响应预留足够的空间
        expected_response_buffer = int(response_estimate * 1.2)
        
        assert budget >= 100
        # 预算应该考虑到响应长度
        assert budget < int(8192 * 0.75)  # 小于总预算
    
    def test_calculate_total_budget(self):
        """测试总预算计算"""
        total_budget_with_memory = self.controller.calculate_total_budget(include_memory=True)
        total_budget_without_memory = self.controller.calculate_total_budget(include_memory=False)
        
        expected_with_memory = int(8192 * 0.75)
        expected_without_memory = int(8192 * 0.65)  # 减去0.1的内存预算
        
        assert total_budget_with_memory == expected_with_memory
        assert total_budget_without_memory == expected_without_memory
    
    def test_estimate_response_length_list_query(self):
        """测试列表类查询响应长度估算"""
        query = "List all machine learning algorithms"
        estimated_length = self.controller.estimate_response_length(query)
        
        # 列表类查询应该有更长的响应
        query_words = len(query.split())
        expected_length = min(query_words * 3, 500)
        
        assert estimated_length == expected_length
    
    def test_estimate_response_length_explanatory_query(self):
        """测试解释类查询响应长度估算"""
        query = "Explain how neural networks work"
        estimated_length = self.controller.estimate_response_length(query)
        
        # 解释类查询应该有中等长度的响应
        query_words = len(query.split())
        expected_length = min(query_words * 2, 300)
        
        assert estimated_length == expected_length
    
    def test_estimate_response_length_factual_query(self):
        """测试事实类查询响应长度估算"""
        query = "What is the capital of France?"
        estimated_length = self.controller.estimate_response_length(query)
        
        # 事实类查询应该有较短的响应
        query_words = len(query.split())
        expected_length = min(query_words * 1.5, 200)
        
        assert estimated_length == expected_length
    
    def test_estimate_response_length_general_query(self):
        """测试一般查询响应长度估算"""
        query = "Tell me something interesting"
        estimated_length = self.controller.estimate_response_length(query)
        
        # 一般查询应该有默认长度的响应
        query_words = len(query.split())
        expected_length = min(query_words * 2, 250)
        
        assert estimated_length == expected_length
    
    def test_update_usage_stats(self):
        """测试使用统计更新"""
        initial_queries = self.controller.usage_stats["total_queries"]
        initial_tokens = self.controller.usage_stats["total_tokens_used"]
        
        tokens_used = 150
        memory_tokens = 50
        
        self.controller.update_usage_stats(tokens_used, memory_tokens)
        
        assert self.controller.usage_stats["total_queries"] == initial_queries + 1
        assert self.controller.usage_stats["total_tokens_used"] == initial_tokens + tokens_used
        assert self.controller.usage_stats["average_tokens_per_query"] > 0
        assert self.controller.usage_stats["total_cost"] > 0
    
    def test_check_budget_constraints_within_limits(self):
        """测试预算约束检查（在限制内）"""
        memory_tokens = 500
        response_tokens = 1000
        
        constraints = self.controller.check_budget_constraints(memory_tokens, response_tokens)
        
        assert constraints["within_limits"] is True
        assert constraints["total_tokens"] == 1500
        assert constraints["max_tokens"] == 8192
        assert constraints["memory_tokens"] == memory_tokens
        assert constraints["response_tokens"] == response_tokens
        assert constraints["memory_ratio"] == 500 / 1500
        assert constraints["response_ratio"] == 1000 / 1500
    
    def test_check_budget_constraints_exceeded(self):
        """测试预算约束检查（超出限制）"""
        memory_tokens = 5000
        response_tokens = 4000  # 总计9000，超过8192
        
        constraints = self.controller.check_budget_constraints(memory_tokens, response_tokens)
        
        assert constraints["within_limits"] is False
        assert constraints["total_tokens"] == 9000
        assert constraints["max_tokens"] == 8192
        assert self.controller.usage_stats["budget_exceeded_count"] == 1
    
    def test_optimize_budget_allocation_no_optimization_needed(self):
        """测试预算分配优化（无需优化）"""
        memory_tokens = 300
        response_tokens = 500
        
        optimization = self.controller.optimize_budget_allocation(memory_tokens, response_tokens)
        
        assert optimization["optimized"] is False
        assert optimization["memory_tokens"] == memory_tokens
        assert optimization["response_tokens"] == response_tokens
    
    def test_optimize_budget_allocation_optimization_needed(self):
        """测试预算分配优化（需要优化）"""
        memory_tokens = 4000
        response_tokens = 5000  # 总计9000，超过限制
        
        optimization = self.controller.optimize_budget_allocation(memory_tokens, response_tokens)
        
        assert optimization["optimized"] is True
        assert optimization["memory_tokens"] < memory_tokens
        assert optimization["response_tokens"] <= 5000
        assert optimization["response_tokens"] >= 50  # 最小响应预算
        assert (optimization["memory_tokens"] + optimization["response_tokens"]) <= 8192
    
    def test_optimize_budget_allocation_extreme_case(self):
        """测试预算分配优化（极端情况）"""
        memory_tokens = 8000
        response_tokens = 5000  # 严重超出限制
        
        optimization = self.controller.optimize_budget_allocation(memory_tokens, response_tokens)
        
        assert optimization["optimized"] is True
        assert optimization["memory_tokens"] < 8192
        assert optimization["response_tokens"] == 50  # 应该设置为最小值
        assert (optimization["memory_tokens"] + optimization["response_tokens"]) <= 8192
    
    def test_get_cost_estimate(self):
        """测试成本估算"""
        tokens = 1000
        cost = self.controller.get_cost_estimate(tokens)
        
        expected_cost = (tokens / 1000) * self.controller.model_config.cost_per_1k_tokens
        assert cost == expected_cost
    
    def test_get_savings_estimate(self):
        """测试节省估算"""
        original_tokens = 1000
        optimized_tokens = 700
        
        savings = self.controller.get_savings_estimate(original_tokens, optimized_tokens)
        
        assert savings["tokens_saved"] == 300
        assert savings["savings_ratio"] == 0.3
        assert savings["percentage_saved"] == 30.0
        assert savings["cost_saved"] > 0
    
    def test_get_stats(self):
        """测试获取统计信息"""
        # 先更新一些使用统计
        self.controller.update_usage_stats(1000, 300)
        self.controller.update_usage_stats(1500, 500)
        
        stats = self.controller.get_stats()
        
        assert "model_name" in stats
        assert "model_config" in stats
        assert "system_prompt_ratio" in stats
        assert "usage_stats" in stats
        assert "estimated_monthly_cost" in stats
        
        assert stats["model_name"] == "openclaw-medium"
        assert stats["system_prompt_ratio"] == 0.1
        assert stats["usage_stats"]["total_queries"] == 2
        assert stats["usage_stats"]["total_tokens_used"] == 2500
        assert stats["estimated_monthly_cost"] > 0
    
    def test_estimate_monthly_cost_no_usage(self):
        """测试无使用情况下的月度成本估算"""
        controller = BudgetController("openclaw-medium")
        monthly_cost = controller._estimate_monthly_cost()
        
        assert monthly_cost == 0.0
    
    def test_estimate_monthly_cost_with_usage(self):
        """测试有使用情况下的月度成本估算"""
        # 模拟一些使用
        self.controller.update_usage_stats(1000, 300)
        self.controller.update_usage_stats(2000, 600)
        
        monthly_cost = self.controller._estimate_monthly_cost()
        
        # 月度成本应该基于当前使用模式计算
        assert monthly_cost > 0
        # 月度成本应该大于日成本
        daily_cost = self.controller.usage_stats["total_cost"]
        assert monthly_cost > daily_cost
    
    def test_export_config(self):
        """测试配置导出"""
        config = self.controller.export_config()
        
        assert "model_name" in config
        assert "system_prompt_ratio" in config
        assert "model_config" in config
        
        assert config["model_name"] == "openclaw-medium"
        assert config["system_prompt_ratio"] == 0.1
        assert config["model_config"]["name"] == "openclaw-medium"
        assert config["model_config"]["max_tokens"] == 8192
    
    def test_load_config(self):
        """测试配置加载"""
        new_config = {
            "model_name": "gpt-4",
            "system_prompt_ratio": 0.15,
            "model_config": {
                "name": "gpt-4",
                "max_tokens": 8192,
                "context_window": 8192,
                "cost_per_1k_tokens": 0.03,
                "optimal_budget_ratio": 0.75
            }
        }
        
        self.controller.load_config(new_config)
        
        assert self.controller.model_name == "gpt-4"
        assert self.controller.system_prompt_ratio == 0.15
        assert self.controller.model_config.name == "gpt-4"
        assert self.controller.model_config.cost_per_1k_tokens == 0.03


class TestBudgetControllerEdgeCases:
    """测试预算控制器的边界情况"""
    
    def test_zero_tokens(self):
        """测试零token情况"""
        controller = BudgetController("openclaw-medium")
        
        budget = controller.calculate_memory_budget("test", 0)
        assert budget >= 100  # 应该返回最小预算
        
        cost = controller.get_cost_estimate(0)
        assert cost == 0.0
        
        savings = controller.get_savings_estimate(0, 0)
        assert savings["tokens_saved"] == 0
        assert savings["savings_ratio"] == 0
    
    def test_negative_tokens(self):
        """测试负token情况"""
        controller = BudgetController("openclaw-medium")
        
        # 负token应该被处理为0
        cost = controller.get_cost_estimate(-100)
        assert cost == 0.0
    
    def test_very_large_token_count(self):
        """测试极大token数量"""
        controller = BudgetController("openclaw-medium")
        
        large_tokens = 1000000
        cost = controller.get_cost_estimate(large_tokens)
        assert cost > 0
        assert cost == (large_tokens / 1000) * controller.model_config.cost_per_1k_tokens
    
    def test_empty_query(self):
        """测试空查询"""
        controller = BudgetController("openclaw-medium")
        
        budget = controller.calculate_memory_budget("")
        assert budget >= 100
        
        response_length = controller.estimate_response_length("")
        assert response_length >= 0
    
    def test_very_long_query(self):
        """测试极长查询"""
        controller = BudgetController("openclaw-medium")
        
        long_query = "What is " + "artificial intelligence " * 100 + "?"
        budget = controller.calculate_memory_budget(long_query)
        assert budget >= 100
        
        response_length = controller.estimate_response_length(long_query)
        assert response_length > 0
        # 响应长度应该有上限
        assert response_length <= 500  # 列表查询的最大值


if __name__ == "__main__":
    pytest.main([__file__, "-v"])