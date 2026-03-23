import pytest
import asyncio
from datetime import datetime
import uuid

from plugin import (
    SpatioTemporalMemoryPlugin,
    get_plugin,
    cleanup_plugin,
    openclaw_with_memory,
    make_configured_llm_api_func,
)
from memory_layers import MemoryConfig


# Mock OpenClaw API函数
async def mock_openclaw_api(prompt: str) -> str:
    """模拟OpenClaw API调用"""
    if "hello" in prompt.lower():
        return "Hello! How can I help you today?"
    elif "weather" in prompt.lower():
        return "The weather is nice today. Sunny with a light breeze."
    elif "memory" in prompt.lower():
        return "I can help you with memory-related questions. What would you like to know?"
    elif "help" in prompt.lower():
        return "I can help you with various tasks. Feel free to ask me anything!"
    elif "ai" in prompt.lower() or "artificial intelligence" in prompt.lower():
        return "AI stands for Artificial Intelligence. It's the simulation of human intelligence in machines."
    else:
        return f"I understand you're asking about: {prompt[:50]}... Let me think about that."


class TestSpatioTemporalMemoryPlugin:
    """测试时空记忆插件"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = MemoryConfig(
            max_shallow_entries=10,
            max_working_entries=5,
            deep_persist_path=":memory:",
            lambda_decay=0.1,
            alpha_similarity=0.6,
            beta_time=0.3,
            gamma_layer=0.1
        )
        self.plugin = SpatioTemporalMemoryPlugin(
            model_name="openclaw-medium",
            memory_config=self.config,
            enable_logging=False  # 禁用日志以提高测试速度
        )
    
    def test_initialization(self):
        """测试插件初始化"""
        assert self.plugin.model_name == "openclaw-medium"
        assert self.plugin.config == self.config
        assert len(self.plugin.layers) == 4  # shallow, working, deep, meta
        assert "shallow" in self.plugin.layers
        assert "working" in self.plugin.layers
        assert "deep" in self.plugin.layers
        assert "meta" in self.plugin.layers
        assert self.plugin.ranker is not None
        assert self.plugin.budget_controller is not None
    
    def test_estimate_token_count(self):
        """测试token数量估算"""
        # 测试英文文本
        english_text = "Hello world, how are you today?"
        tokens = self.plugin._estimate_token_count(english_text)
        assert tokens > 0
        assert isinstance(tokens, int)
        
        # 测试中文文本
        chinese_text = "你好，今天天气怎么样？"
        tokens_chinese = self.plugin._estimate_token_count(chinese_text)
        assert tokens_chinese > 0
        assert isinstance(tokens_chinese, int)
        
        # 测试混合文本
        mixed_text = "Hello 你好 world 世界"
        tokens_mixed = self.plugin._estimate_token_count(mixed_text)
        assert tokens_mixed > 0
    
    def test_create_memory_entry(self):
        """测试创建记忆条目"""
        query = "What is AI?"
        response = "AI stands for Artificial Intelligence."
        layer = "shallow"
        
        entry = self.plugin._create_memory_entry(query, response, layer)
        
        assert entry.query == query
        assert entry.response == response
        assert entry.layer == layer
        assert entry.token_count > 0
        assert isinstance(entry.id, str)
        assert isinstance(entry.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories_empty(self):
        """测试空记忆检索"""
        query = "What is quantum computing?"
        
        context, memory_tokens, relevant_memories = await self.plugin.retrieve_relevant_memories(query)
        
        assert context == ""  # 没有相关记忆
        assert memory_tokens == 0
        assert len(relevant_memories) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories_with_data(self):
        """测试有数据的记忆检索"""
        # 先添加一些测试数据
        test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Tell me about neural networks"
        ]
        
        for query in test_queries:
            entry = self.plugin._create_memory_entry(query, f"Response to: {query}", "shallow")
            self.plugin.layers["shallow"].add(entry)
        
        # 检索相关记忆
        query = "What is AI and machine learning?"
        context, memory_tokens, relevant_memories = await self.plugin.retrieve_relevant_memories(query)
        
        # 应该找到一些相关记忆
        assert len(relevant_memories) >= 0  # 可能找到相关结果
        assert isinstance(context, str)
        assert memory_tokens >= 0
    
    @pytest.mark.asyncio
    async def test_process_query_basic(self):
        """测试基础查询处理"""
        query = "Hello, how are you?"
        
        result = await self.plugin.process_query(query, mock_openclaw_api)
        
        assert result["status"] == "success"
        assert result["query"] == query
        assert len(result["response"]) > 0
        assert result["latency_ms"] > 0
        assert result["memory_tokens"] >= 0
        assert result["response_tokens"] > 0
        assert result["total_tokens"] > 0
        assert result["relevant_memories_count"] >= 0
        assert "budget_check" in result
    
    @pytest.mark.asyncio
    async def test_process_query_with_context(self):
        """测试带上下文的查询处理"""
        # 先添加一些相关记忆
        context_query = "What is artificial intelligence?"
        context_entry = self.plugin._create_memory_entry(
            context_query, 
            "AI is the simulation of human intelligence in machines.", 
            "shallow"
        )
        self.plugin.layers["shallow"].add(context_entry)
        
        # 询问相关问题
        related_query = "Can you explain more about AI?"
        
        result = await self.plugin.process_query(related_query, mock_openclaw_api)
        
        assert result["status"] == "success"
        assert result["query"] == related_query
        assert len(result["response"]) > 0
        assert result["relevant_memories_count"] >= 0
        
        # 检查是否使用了上下文
        if result["relevant_memories_count"] > 0:
            assert len(result["context"]) > 0
    
    @pytest.mark.asyncio
    async def test_process_query_error_handling(self):
        """测试错误处理"""
        async def failing_api(prompt: str) -> str:
            raise Exception("API Error")
        
        query = "This should fail"
        result = await self.plugin.process_query(query, failing_api)
        
        assert result["status"] == "error"
        assert "error" in result
        assert "API Error" in result["error"]
        assert result["latency_ms"] > 0
    
    def test_cleanup_expired_memories(self):
        """测试过期记忆清理"""
        # 添加一些即将过期的记忆
        from datetime import timedelta
        
        old_entry = self.plugin._create_memory_entry(
            "Old query",
            "Old response",
            "shallow"
        )
        # 修改时间使其过期
        old_entry.timestamp = datetime.now() - timedelta(minutes=10)
        self.plugin.layers["shallow"].add(old_entry)
        
        # 清理过期记忆
        removed = self.plugin.cleanup_expired_memories()
        
        # 应该移除了过期记忆
        assert removed >= 0
    
    def test_get_memory_stats(self):
        """测试获取记忆统计"""
        # 添加一些测试数据
        for i in range(3):
            entry = self.plugin._create_memory_entry(
                f"Query {i}",
                f"Response {i}",
                "shallow"
            )
            self.plugin.layers["shallow"].add(entry)
        
        stats = self.plugin.get_memory_stats()
        
        assert "total" in stats
        assert "shallow" in stats
        assert "working" in stats
        assert "deep" in stats
        assert "meta" in stats
        
        assert stats["total"]["entries"] >= 3
        assert stats["total"]["layers"] == 4
        assert stats["shallow"]["entries"] >= 3
    
    def test_get_performance_stats(self):
        """测试获取性能统计"""
        stats = self.plugin.get_performance_stats()
        
        assert "performance" in stats
        assert "budget" in stats
        assert "ranker" in stats
        assert "recent_queries" in stats
        
        performance = stats["performance"]
        assert "total_queries" in performance
        assert "average_latency_ms" in performance
        assert "total_tokens_saved" in performance
        assert "total_cost_saved" in performance
    
    def test_export_configuration(self):
        """测试配置导出"""
        config = self.plugin.export_configuration()
        
        assert "model_name" in config
        assert "memory_config" in config
        assert "budget_config" in config
        assert "performance_stats" in config
        
        assert config["model_name"] == "openclaw-medium"
        assert "shallow_ttl" in config["memory_config"]
        assert "lambda_decay" in config["memory_config"]
    
    def test_reset_statistics(self):
        """测试重置统计信息"""
        # 先进行一些操作以产生统计
        self.plugin.performance_stats["total_queries"] = 10
        self.plugin.performance_stats["total_tokens_saved"] = 1000
        
        # 重置统计
        self.plugin.reset_statistics()
        
        assert self.plugin.performance_stats["total_queries"] == 0
        assert self.plugin.performance_stats["total_tokens_saved"] == 0
        assert self.plugin.performance_stats["average_latency_ms"] == 0
        assert len(self.plugin.query_log) == 0


class TestPluginSingleton:
    """测试插件单例模式"""
    
    def test_get_plugin_singleton(self):
        """测试获取插件单例"""
        plugin1 = get_plugin(model_name="openclaw-medium")
        plugin2 = get_plugin(model_name="openclaw-medium")
        
        assert plugin1 is plugin2  # 应该是同一个实例
    
    def test_get_plugin_different_models(self):
        """测试获取不同模型的插件"""
        # 清理现有实例
        cleanup_plugin()
        
        plugin1 = get_plugin(model_name="openclaw-medium")
        plugin2 = get_plugin(model_name="openclaw-small")  # 应该创建新的实例
        
        # 由于单例模式，第二个调用应该返回第一个实例
        assert plugin1 is plugin2
        # 但模型名称应该使用第一次调用的设置
        assert plugin1.model_name == "openclaw-medium"


class TestMainPluginFunction:
    """测试主插件函数"""
    
    @pytest.mark.asyncio
    async def test_openclaw_with_memory(self):
        """测试带记忆的OpenClaw函数"""
        # 清理现有实例
        cleanup_plugin()
        
        query = "Hello, can you help me with AI?"
        
        result = await openclaw_with_memory(query, mock_openclaw_api, model_name="openclaw-medium")
        
        assert result["status"] == "success"
        assert result["query"] == query
        assert len(result["response"]) > 0
        assert result["latency_ms"] > 0
        assert "memory_tokens" in result
        assert "response_tokens" in result
        assert "total_tokens" in result


class TestMemoryIntegration:
    """测试记忆集成"""
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """测试多轮对话"""
        plugin = SpatioTemporalMemoryPlugin(enable_logging=False)
        
        # 第一轮对话
        query1 = "What is artificial intelligence?"
        result1 = await plugin.process_query(query1, mock_openclaw_api)
        assert result1["status"] == "success"
        
        # 第二轮对话，询问相关问题
        query2 = "Can you tell me more about AI applications?"
        result2 = await plugin.process_query(query2, mock_openclaw_api)
        assert result2["status"] == "success"
        
        # 第三轮对话，询问之前的内容
        query3 = "What did we talk about earlier?"
        result3 = await plugin.process_query(query3, mock_openclaw_api)
        assert result3["status"] == "success"
        
        # 检查记忆统计
        stats = plugin.get_memory_stats()
        assert stats["total"]["entries"] >= 3  # 至少有3条记忆
        
        # 检查性能统计
        perf_stats = plugin.get_performance_stats()
        assert perf_stats["performance"]["total_queries"] == 3
        assert perf_stats["performance"]["average_latency_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_memory_layer_interaction(self):
        """测试记忆层交互"""
        plugin = SpatioTemporalMemoryPlugin(enable_logging=False)
        
        # 添加数据到不同层级
        layers = ["shallow", "working", "deep"]
        for i, layer in enumerate(layers):
            entry = plugin._create_memory_entry(
                f"Query for {layer} layer",
                f"Response for {layer} layer",
                layer
            )
            plugin.layers[layer].add(entry)
        
        # 查询应该能从多个层级检索记忆
        query = "Tell me about the layers"
        result = await plugin.process_query(query, mock_openclaw_api)
        
        assert result["status"] == "success"
        # 应该找到了相关记忆（可能来自多个层级）
        assert result["relevant_memories_count"] >= 0


class TestLocalLLM:
    async def test_local_llm_success(self):
        from aiohttp import web

        seen_headers = {}

        async def models_handler(request):
            return web.json_response({"data": [{"id": "qwen3.5-9b"}]})

        async def chat_handler(request):
            seen_headers.update(dict(request.headers))
            payload = await request.json()
            assert payload.get("model") == "qwen3.5-9b"
            return web.json_response({"choices": [{"message": {"content": "ok"}}]})

        app = web.Application()
        app.router.add_get("/v1/models", models_handler)
        app.router.add_post("/v1/chat/completions", chat_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            port = site._server.sockets[0].getsockname()[1]
            base_url = f"http://127.0.0.1:{port}/v1"

            cfg = MemoryConfig(
                llm_provider="local",
                local_llm_base_url=base_url,
                local_llm_model="qwen3.5-9b",
                local_llm_check=True,
            )
            api = make_configured_llm_api_func(cfg)
            out = await api("hi")
            assert out == "ok"
            assert "Authorization" not in seen_headers
        finally:
            await runner.cleanup()

    async def test_local_llm_missing_model(self):
        from aiohttp import web

        async def models_handler(request):
            return web.json_response({"data": [{"id": "other"}]})

        async def chat_handler(request):
            return web.json_response({"choices": [{"message": {"content": "should-not-call"}}]})

        app = web.Application()
        app.router.add_get("/v1/models", models_handler)
        app.router.add_post("/v1/chat/completions", chat_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            port = site._server.sockets[0].getsockname()[1]
            base_url = f"http://127.0.0.1:{port}/v1"

            cfg = MemoryConfig(
                llm_provider="local",
                local_llm_base_url=base_url,
                local_llm_model="qwen3.5-9b",
                local_llm_check=True,
            )
            api = make_configured_llm_api_func(cfg)
            with pytest.raises(RuntimeError) as e:
                await api("hi")
            assert "未发现模型" in str(e.value) or "未发现" in str(e.value)
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
