"""
SpatioTemporal Memory System for OpenClaw

一个面向OpenClaw的玩具级智能体插件，通过分层时空记忆系统降低调用成本并提升推理效率。

核心特性:
- 分层记忆架构（浅层、工作、深层、元记忆）
- 时空建模与智能检索
- Token预算管理与成本优化
- 高性能异步处理
- 轻量级设计，适合CPU-only环境

作者: AI Assistant
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "SpatioTemporal Memory System for OpenClaw"

from .memory_layers import (
    MemoryLayer,
    ShallowMemoryLayer,
    WorkingMemoryLayer,
    DeepMemoryLayer,
    MetaMemoryLayer,
    MemoryEntry,
    MemoryConfig
)

from .ranker import SpatioTemporalRanker
from .budget import BudgetController
from .plugin import SpatioTemporalMemoryPlugin

__all__ = [
    "MemoryLayer",
    "ShallowMemoryLayer", 
    "WorkingMemoryLayer",
    "DeepMemoryLayer",
    "MetaMemoryLayer",
    "MemoryEntry",
    "MemoryConfig",
    "SpatioTemporalRanker",
    "BudgetController",
    "SpatioTemporalMemoryPlugin"
]