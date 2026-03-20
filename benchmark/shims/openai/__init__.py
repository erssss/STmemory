import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: List[_Choice]


class OpenAI:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.chat = _Chat(self)


class _Chat:
    def __init__(self, client: OpenAI):
        self.client = client
        self.completions = _Completions(client)


class _Completions:
    def __init__(self, client: OpenAI):
        self.client = client

    def create(self, model: str, messages: List[Dict[str, Any]], temperature: float = 0.0, response_format: Any = None) -> _Response:
        payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if response_format is not None:
            payload["response_format"] = response_format
        url = f"{self.client.base_url}/chat/completions"
        req = urllib.request.Request(
            url=url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.client.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenAI shim got no choices: {data}")
        content = ((choices[0] or {}).get("message") or {}).get("content")
        if content is None:
            raise RuntimeError(f"OpenAI shim got empty content: {data}")
        return _Response(choices=[_Choice(message=_Message(content=str(content)))])

