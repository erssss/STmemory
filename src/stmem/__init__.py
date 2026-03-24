"""
stmem - Intelligent Memory System

An AI-powered intelligent memory management system that provides a persistent memory layer for LLM applications.
"""

import os
from typing import Any

__version__ = os.environ.get("POWERMEM_VERSION", "1.0.0")

# Import core classes
from .core.memory import Memory, _auto_convert_config
from .core.base import MemoryBase

# Import configuration loader
from .config_loader import load_config_from_env, create_config, validate_config, auto_config


def __getattr__(name: str):
    if name == "UserMemory":
        from .user_memory import UserMemory

        return UserMemory
    raise AttributeError(name)


def create_memory(
    config: Any = None,
    **kwargs
):
    if config is None:
        config = auto_config()
    
    return Memory(config=config, **kwargs)


__all__ = [
    "Memory",
    "MemoryBase",
    "UserMemory",
    "load_config_from_env",
    "create_config",
    "validate_config",
    "create_memory",
    "auto_config",
]
