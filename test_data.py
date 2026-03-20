"""
测试数据集生成器
生成用于评估时空记忆系统性能的对话数据
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any


def generate_conversation_dataset(num_conversations: int = 50) -> List[Dict[str, Any]]:
    """
    生成对话数据集
    
    Args:
        num_conversations: 对话数量
        
    Returns:
        对话数据集
    """
    
    # 预定义的主题和查询模板
    topics = {
        "ai": {
            "queries": [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "Can you explain deep learning?",
                "What are neural networks?",
                "How is AI used in real life?",
                "What is the difference between AI and ML?",
                "Tell me about computer vision",
                "What is natural language processing?"
            ],
            "responses": [
                "AI is the simulation of human intelligence in machines.",
                "Machine learning uses algorithms to learn from data.",
                "Deep learning uses neural networks with multiple layers.",
                "Neural networks are computing systems inspired by biological neural networks.",
                "AI is used in healthcare, finance, transportation, and many other fields.",
                "AI is the broader concept, while ML is a subset of AI.",
                "Computer vision enables machines to interpret and understand visual information.",
                "NLP allows computers to understand, interpret, and generate human language."
            ]
        },
        "weather": {
            "queries": [
                "What's the weather like today?",
                "Is it going to rain tomorrow?",
                "What's the temperature?",
                "How's the weather in your area?",
                "Should I bring an umbrella?",
                "Is it sunny outside?",
                "What's the forecast for this week?"
            ],
            "responses": [
                "The weather is nice today, sunny with a light breeze.",
                "There's a slight chance of rain tomorrow, about 30%.",
                "The temperature is around 22°C, quite comfortable.",
                "It's a beautiful day here, clear skies and mild temperature.",
                "It might be a good idea to bring an umbrella, just in case.",
                "Yes, it's quite sunny and pleasant outside.",
                "The forecast shows mostly sunny days with temperatures in the low 20s."
            ]
        },
        "general": {
            "queries": [
                "Hello, how are you?",
                "Can you help me?",
                "What can you do?",
                "Tell me something interesting",
                "What's new?",
                "How was your day?",
                "What should I know about you?"
            ],
            "responses": [
                "Hello! I'm doing well, thank you for asking.",
                "Of course! I'd be happy to help you with anything you need.",
                "I can help you with questions about AI, weather, and general topics.",
                "Here's something interesting: The human brain has about 86 billion neurons.",
                "Not much is new, but I'm always here to chat and help!",
                "My day has been productive, helping users with their questions.",
                "I'm an AI assistant designed to help answer your questions and have conversations."
            ]
        },
        "memory": {
            "queries": [
                "Do you remember what we talked about?",
                "What did I ask you earlier?",
                "Can you recall our previous conversation?",
                "What were we discussing before?",
                "Do you have memory of our chat?",
                "What topics have we covered?"
            ],
            "responses": [
                "Yes, I can recall our previous conversations using my memory system.",
                "You asked me about various topics, and I've stored that information.",
                "I can remember our previous discussions through my layered memory system.",
                "We were discussing various topics, and I've kept track of them.",
                "Yes, I use a sophisticated memory system to remember our interactions.",
                "We've covered topics like AI, weather, and general knowledge."
            ]
        }
    }
    
    conversations = []
    
    for conv_id in range(num_conversations):
        # 随机选择主题
        topic = random.choice(list(topics.keys()))
        topic_data = topics[topic]
        
        # 生成多轮对话
        num_turns = random.randint(3, 8)
        conversation = {
            "id": f"conv_{conv_id:03d}",
            "topic": topic,
            "turns": [],
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
        }
        
        # 生成对话轮次
        for turn_id in range(num_turns):
            # 随机选择查询和响应
            query = random.choice(topic_data["queries"])
            response = random.choice(topic_data["responses"])
            
            # 添加一些变化以避免完全重复
            if random.random() < 0.3:  # 30%概率添加个性化内容
                query += " I'm particularly interested in this topic."
                response += " Let me know if you'd like more details."
            
            turn = {
                "id": f"turn_{turn_id:02d}",
                "query": query,
                "response": response,
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat()
            }
            
            conversation["turns"].append(turn)
        
        conversations.append(conversation)
    
    return conversations


def generate_evaluation_queries() -> List[Dict[str, Any]]:
    """
    生成评估查询数据集
    
    Returns:
        评估查询列表
    """
    
    evaluation_queries = [
        {
            "id": "eval_001",
            "query": "What is artificial intelligence?",
            "expected_topics": ["ai", "technology"],
            "difficulty": "easy",
            "type": "factual"
        },
        {
            "id": "eval_002", 
            "query": "How does machine learning relate to AI?",
            "expected_topics": ["ai", "machine learning"],
            "difficulty": "medium",
            "type": "explanatory"
        },
        {
            "id": "eval_003",
            "query": "Can you explain what we discussed about neural networks?",
            "expected_topics": ["ai", "memory"],
            "difficulty": "hard",
            "type": "memory_recall"
        },
        {
            "id": "eval_004",
            "query": "What's the weather like today?",
            "expected_topics": ["weather"],
            "difficulty": "easy",
            "type": "factual"
        },
        {
            "id": "eval_005",
            "query": "Do you remember our previous conversations about AI?",
            "expected_topics": ["ai", "memory"],
            "difficulty": "medium",
            "type": "memory_recall"
        },
        {
            "id": "eval_006",
            "query": "Tell me something interesting about technology",
            "expected_topics": ["ai", "general"],
            "difficulty": "medium",
            "type": "open_ended"
        },
        {
            "id": "eval_007",
            "query": "How can machine learning be applied in real life?",
            "expected_topics": ["ai", "applications"],
            "difficulty": "hard",
            "type": "application"
        },
        {
            "id": "eval_008",
            "query": "What topics have we covered in our discussions?",
            "expected_topics": ["memory", "general"],
            "difficulty": "medium",
            "type": "memory_recall"
        }
    ]
    
    return evaluation_queries


def generate_stress_test_queries(num_queries: int = 100) -> List[str]:
    """
    生成压力测试查询
    
    Args:
        num_queries: 查询数量
        
    Returns:
        压力测试查询列表
    """
    
    base_queries = [
        "What is AI?",
        "How does machine learning work?",
        "Tell me about neural networks",
        "What's the weather like?",
        "Can you help me?",
        "Do you remember our previous conversation?",
        "What is deep learning?",
        "How is AI used in healthcare?",
        "What's new with you?",
        "Explain computer vision"
    ]
    
    # 生成变体
    stress_queries = []
    for i in range(num_queries):
        base_query = random.choice(base_queries)
        
        # 添加随机变体
        if random.random() < 0.5:
            # 添加额外词语
            extra_words = ["please", "exactly", "in detail", "briefly", "simply"]
            base_query += " " + random.choice(extra_words)
        
        if random.random() < 0.3:
            # 添加个性化内容
            base_query += " I'm really interested in this topic."
        
        stress_queries.append(base_query)
    
    return stress_queries


def generate_edge_case_queries() -> List[Dict[str, Any]]:
    """
    生成边界情况测试查询
    
    Returns:
        边界情况查询列表
    """
    
    edge_cases = [
        {
            "id": "edge_001",
            "query": "",  # 空查询
            "type": "empty",
            "expected_behavior": "graceful_handling"
        },
        {
            "id": "edge_002",
            "query": "a",  # 单个字符
            "type": "minimal",
            "expected_behavior": "minimal_response"
        },
        {
            "id": "edge_003",
            "query": "What is " + "artificial intelligence " * 50 + "?",  # 非常长的查询
            "type": "very_long",
            "expected_behavior": "truncated_handling"
        },
        {
            "id": "edge_004",
            "query": "!@#$%^&*()",  # 特殊字符
            "type": "special_chars",
            "expected_behavior": "special_handling"
        },
        {
            "id": "edge_005",
            "query": "123456789",  # 纯数字
            "type": "numeric",
            "expected_behavior": "numeric_handling"
        },
        {
            "id": "edge_006",
            "query": "AI ML DL NN CV NLP",  # 缩写密集
            "type": "abbreviations",
            "expected_behavior": "abbreviation_handling"
        },
        {
            "id": "edge_007",
            "query": "What is AI in English, 中文, Français, Español?",  # 多语言
            "type": "multilingual",
            "expected_behavior": "multilingual_handling"
        }
    ]
    
    return edge_cases


def save_test_data():
    """保存测试数据到文件"""
    
    # 生成数据
    conversations = generate_conversation_dataset(50)
    evaluation_queries = generate_evaluation_queries()
    stress_queries = generate_stress_test_queries(100)
    edge_cases = generate_edge_case_queries()
    
    # 保存到文件
    test_data = {
        "conversations": conversations,
        "evaluation_queries": evaluation_queries,
        "stress_queries": stress_queries,
        "edge_cases": edge_cases,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_conversations": len(conversations),
            "total_evaluation_queries": len(evaluation_queries),
            "total_stress_queries": len(stress_queries),
            "total_edge_cases": len(edge_cases)
        }
    }
    
    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"测试数据已保存到 test_data.json")
    print(f"对话数量: {len(conversations)}")
    print(f"评估查询数量: {len(evaluation_queries)}")
    print(f"压力测试查询数量: {len(stress_queries)}")
    print(f"边界情况数量: {len(edge_cases)}")


if __name__ == "__main__":
    save_test_data()