import json
import os
import urllib.error
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
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
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
        headers = {"Content-Type": "application/json"}
        if str(self.client.api_key or "").strip():
            headers["Authorization"] = f"Bearer {self.client.api_key}"
        req = urllib.request.Request(
            url=url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if int(getattr(e, "code", 0) or 0) != 404:
                raise
            base = str(self.client.base_url or "").rstrip("/")
            root = base[:-3] if base.endswith("/v1") else base
            ollama_url = f"{root}/api/chat"
            ollama_payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
            temp = float(temperature or 0.0)
            if temp != 0.0:
                ollama_payload["options"] = {"temperature": temp}
            if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                ollama_payload["format"] = "json"
            ollama_req = urllib.request.Request(
                url=ollama_url,
                method="POST",
                data=json.dumps(ollama_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(ollama_req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            ollama_data = json.loads(raw) if raw.strip() else {}
            content = (((ollama_data.get("message") or {}) or {}).get("content") or "")
            return _Response(choices=[_Choice(message=_Message(content=str(content)))])
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenAI shim got no choices: {data}")
        content = ((choices[0] or {}).get("message") or {}).get("content")
        if content is None:
            raise RuntimeError(f"OpenAI shim got empty content: {data}")
        return _Response(choices=[_Choice(message=_Message(content=str(content)))])
