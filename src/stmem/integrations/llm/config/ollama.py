from typing import Optional

try:
    from pydantic import AliasChoices, Field
except Exception:
    from stmem.utils.pydantic_compat import AliasChoices, Field

from stmem.integrations.llm.config.base import BaseLLMConfig
from stmem.settings import settings_config


class OllamaConfig(BaseLLMConfig):
    """
    Configuration class for Ollama-specific parameters.
    Inherits from BaseLLMConfig and adds Ollama-specific settings.
    """

    _provider_name = "ollama"
    _class_path = "stmem.integrations.llm.ollama.OllamaLLM"

    model_config = settings_config("LLM_", extra="forbid", env_file=None)

    # Ollama-specific fields
    ollama_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "ollama_base_url",
            "OLLAMA_LLM_BASE_URL",
        ),
        description="Ollama base URL"
    )
