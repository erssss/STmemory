from typing import Any, Dict, Optional, Union

try:
    import httpx
except Exception:
    class _HttpxClient:  # type: ignore
        pass

    class _HttpxModule:  # type: ignore
        Client = _HttpxClient

    httpx = _HttpxModule()
try:
    from pydantic import AliasChoices, Field
except Exception:
    from stmem.utils.pydantic_compat import AliasChoices, Field

from stmem.integrations.embeddings.config.base import BaseEmbedderConfig
from stmem.settings import settings_config


class QwenEmbeddingConfig(BaseEmbedderConfig):
    _provider_name = "qwen"
    _class_path = "stmem.integrations.embeddings.qwen.QwenEmbedding"

    model_config = settings_config("EMBEDDING_", extra="forbid", env_file=None)

    api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "api_key",
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
            "EMBEDDING_API_KEY",
        ),
    )
    model: Optional[str] = Field(default=None)
    dashscope_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "dashscope_base_url",
            "QWEN_EMBEDDING_BASE_URL",
        ),
    )
    memory_add_embedding_type: Optional[str] = Field(default=None)
    memory_update_embedding_type: Optional[str] = Field(default=None)
    memory_search_embedding_type: Optional[str] = Field(default=None)



class OllamaEmbeddingConfig(BaseEmbedderConfig):
    _provider_name = "ollama"
    _class_path = "stmem.integrations.embeddings.ollama.OllamaEmbedding"

    model_config = settings_config("EMBEDDING_", extra="forbid", env_file=None)

    model: Optional[str] = Field(default=None)
    ollama_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OLLAMA_EMBEDDING_BASE_URL"),
    )




class VertexAIEmbeddingConfig(BaseEmbedderConfig):
    _provider_name = "vertexai"
    _class_path = "stmem.integrations.embeddings.vertexai.VertexAIEmbedding"

    model_config = settings_config("EMBEDDING_", extra="forbid", env_file=None)

    model: Optional[str] = Field(default=None)
    vertex_credentials_json: Optional[str] = Field(default=None)
    memory_add_embedding_type: Optional[str] = Field(default=None)
    memory_update_embedding_type: Optional[str] = Field(default=None)
    memory_search_embedding_type: Optional[str] = Field(default=None)


class TogetherEmbeddingConfig(BaseEmbedderConfig):
    _provider_name = "together"
    _class_path = "stmem.integrations.embeddings.together.TogetherEmbedding"

    model_config = settings_config("EMBEDDING_", extra="forbid", env_file=None)

    model: Optional[str] = Field(default=None)



class MockEmbeddingConfig(BaseEmbedderConfig):
    _provider_name = "mock"
    _class_path = "stmem.integrations.embeddings.mock.MockEmbeddings"

    model_config = settings_config("EMBEDDING_", extra="allow", env_file=None)


class CustomEmbeddingConfig(BaseEmbedderConfig):
    model_config = settings_config("EMBEDDING_", extra="allow", env_file=None)
