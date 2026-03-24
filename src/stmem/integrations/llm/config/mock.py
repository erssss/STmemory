from typing import Optional

try:
    from pydantic import Field
except Exception:
    from stmem.utils.pydantic_compat import Field

from stmem.integrations.llm.config.base import BaseLLMConfig
from stmem.settings import settings_config


class MockLLMConfig(BaseLLMConfig):
    _provider_name = "mock"
    _class_path = "stmem.integrations.llm.mock.MockLLM"

    model_config = settings_config("LLM_", extra="allow", env_file=None)

    model: Optional[str] = Field(default="mock")
