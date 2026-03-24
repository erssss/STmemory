import json
from typing import Dict, List, Optional, Union

from stmem.integrations.llm.base import LLMBase
from stmem.integrations.llm.config.base import BaseLLMConfig


class MockLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLLMConfig, Dict]] = None):
        super().__init__(config)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        if tools:
            return {"content": "", "tool_calls": []}

        wants_json = False
        if isinstance(response_format, dict):
            wants_json = response_format.get("type") == "json_object"

        last_content = ""
        if messages:
            last = messages[-1] if isinstance(messages[-1], dict) else {}
            last_content = str(last.get("content") or "")

        if wants_json:
            lower = last_content.lower()
            if '"facts"' in lower or "facts" in lower:
                return json.dumps({"facts": []})
            if '"memory"' in lower or "event" in lower or "add|update|delete" in lower:
                return json.dumps({"memory": []})
            return json.dumps({})

        return ""
