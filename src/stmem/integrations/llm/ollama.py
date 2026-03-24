from typing import Dict, List, Optional, Union

from stmem.integrations.llm import LLMBase
from stmem.integrations.llm.config.base import BaseLLMConfig
from stmem.integrations.llm.config.ollama import OllamaConfig

try:
    from ollama import Client
except ImportError:
    Client = None


class OllamaLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLLMConfig, OllamaConfig, Dict]] = None):
        # Convert to OllamaConfig if needed
        if config is None:
            config = OllamaConfig()
        elif isinstance(config, dict):
            config = OllamaConfig(**config)
        elif isinstance(config, BaseLLMConfig) and not isinstance(config, OllamaConfig):
            # Convert BaseLLMConfig to OllamaConfig
            config = OllamaConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "qwen3.5:9b"

        self.client = None
        if Client is not None:
            self.client = Client(host=getattr(self.config, "ollama_base_url", "http://localhost:11434"))

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API (may be stream, list, or dict).
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        import json
        def try_parse_json(content):
            try:
                return json.loads(content)
            except Exception:
                return content

        # 兼容 Ollama 流式/多段响应
        def extract_content(resp):
            # Ollama Python SDK 可能返回生成器、列表或单个 dict
            if hasattr(resp, '__iter__') and not isinstance(resp, (dict, str, bytes)):
                # 生成器或列表
                contents = []
                for chunk in resp:
                    # chunk 可能是 dict 或对象
                    if isinstance(chunk, dict):
                        c = chunk.get("message", {}).get("content", "")
                    else:
                        c = getattr(getattr(chunk, "message", None), "content", "")
                    if c:
                        contents.append(c)
                return "".join(contents)
            elif isinstance(resp, dict):
                return resp.get("message", {}).get("content", "")
            else:
                return getattr(getattr(resp, "message", None), "content", "")

        if tools:
            content = extract_content(response)
            processed_response = {
                "content": content,
                "tool_calls": [],
            }
            return processed_response
        else:
            content = extract_content(response)
            return try_parse_json(content)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using Ollama (requests, not SDK).
        """
        import copy
        import requests
        url = self.config.ollama_base_url or "http://localhost:11434"
        api_url = f"{url}/api/chat"
        data = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }
        fields_set = getattr(self.config, "model_fields_set", set()) or set()
        explicit = set(fields_set)
        provided = set(kwargs.keys())
        temperature = kwargs.pop("temperature", self.config.temperature)
        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
        top_p = kwargs.pop("top_p", self.config.top_p)
        top_k = kwargs.pop("top_k", self.config.top_k)
        think = kwargs.pop("think", None)

        options = {}
        if "temperature" in explicit or "temperature" in provided:
            if temperature is not None:
                options["temperature"] = temperature
        if "max_tokens" in explicit or "max_tokens" in provided:
            if max_tokens is not None and int(max_tokens) > 0:
                options["num_predict"] = int(max_tokens)
        if "top_p" in explicit or "top_p" in provided:
            if top_p is not None:
                options["top_p"] = top_p
        if "top_k" in explicit or "top_k" in provided:
            if top_k is not None:
                options["top_k"] = top_k

        if options:
            data["options"] = options
        if think is None:
            data["think"] = False
        else:
            data["think"] = bool(think)
        # 兼容 json_object 格式
        wants_json = False
        if isinstance(response_format, dict):
            wants_json = response_format.get("type") == "json_object"
        if wants_json and not tools:
            data["format"] = "json"
        if tools:
            data["tools"] = tools
            if tool_choice:
                data["tool_choice"] = tool_choice

        def extract_content(result: Dict):
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result.get("error")))
            message = result.get("message")
            if isinstance(message, dict):
                content = message.get("content", "")
                if content is None:
                    return ""
                return content
            response = result.get("response")
            if response is None:
                return ""
            return response

        def call(payload: Dict):
            resp = requests.post(api_url, json=payload, timeout=600)
            resp.raise_for_status()
            return resp.json()

        result = call(data)
        content = extract_content(result)
        if isinstance(content, str) and not content.strip():
            retry_payload = copy.deepcopy(data)
            retry_payload.pop("options", None)
            retry_payload.pop("format", None)
            result2 = call(retry_payload)
            content2 = extract_content(result2)
            if not (isinstance(content2, str) and not content2.strip()):
                return content2
        return content
