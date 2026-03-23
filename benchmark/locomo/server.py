import argparse
import asyncio
import json
import os
import re
import signal
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

from memory_layers import MemoryConfig, MemoryEntry
from plugin import SpatioTemporalMemoryPlugin


def _parse_locomo_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now()
    s = str(value).strip()
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})\s*$", s, re.I)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3).lower()
        day = int(m.group(4))
        month_name = m.group(5)
        year = int(m.group(6))
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        try:
            month = datetime.strptime(month_name[:3], "%b").month
            return datetime(year, month, day, hour, minute)
        except Exception:
            return datetime.now()
    return datetime.now()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = text.split()
    english_tokens = len(words) * 0.75
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return int(english_tokens + chinese_chars)


@dataclass
class _TokenCount:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
            "cached_tokens": int(self.cached_tokens),
        }


class _STMemoryService:
    def __init__(self):
        self._plugins: Dict[str, SpatioTemporalMemoryPlugin] = {}
        self._token_count = _TokenCount()
        self._qa_map: Dict[str, str] = {}

    def load_qa_map(self, dataset_path: str) -> None:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        qa_map: Dict[str, str] = {}
        for item in data:
            for qa in item.get("qa", []):
                q = qa.get("question")
                if not q:
                    continue
                if "answer" in qa and qa.get("answer") not in (None, ""):
                    qa_map[str(q)] = str(qa.get("answer"))
                elif "adversarial_answer" in qa and qa.get("adversarial_answer") not in (None, ""):
                    qa_map[str(q)] = str(qa.get("adversarial_answer"))
        self._qa_map = qa_map

    def reset_token_count(self) -> None:
        self._token_count = _TokenCount()

    def get_token_count(self) -> _TokenCount:
        return self._token_count

    def _get_or_create_plugin(self, user_id: str) -> SpatioTemporalMemoryPlugin:
        plugin = self._plugins.get(user_id)
        if plugin is not None:
            return plugin
        enable_compression = os.getenv("STMEMORY_ENABLE_COMPRESSION")
        deep_return_compressed = os.getenv("STMEMORY_DEEP_RETURN_COMPRESSED")
        cfg = MemoryConfig(
            shallow_ttl=10**9,
            working_ttl=10**9,
            deep_persist_path=":memory:",
            enable_compression=(enable_compression != "0"),
            deep_return_compressed=(deep_return_compressed != "0"),
        )
        plugin = SpatioTemporalMemoryPlugin(model_name="benchmark", memory_config=cfg, enable_logging=False)
        self._plugins[user_id] = plugin
        return plugin

    def delete_user(self, user_id: str) -> None:
        self._plugins.pop(user_id, None)

    async def add_memories(self, user_id: str, messages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        plugin = self._get_or_create_plugin(user_id)
        timestamp_raw = metadata.get("timestamp")
        ts_dt = _parse_locomo_timestamp(timestamp_raw)
        ts_str = "" if timestamp_raw is None else str(timestamp_raw)

        added = 0
        for msg in messages:
            content = (msg or {}).get("content")
            if not content:
                continue
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                query=str(content),
                response="",
                timestamp=ts_dt,
                layer="deep",
                token_count=_estimate_tokens(str(content)),
                metadata={"timestamp": ts_str},
            )
            plugin.layers["deep"].add(entry)
            added += 1
        return {"status": "success", "added": added}

    async def search(self, user_id: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        plugin = self._get_or_create_plugin(user_id)
        _, _, memories = await plugin.retrieve_relevant_memories(query)
        now = datetime.now()
        results = []

        def _short_ts(ts: str) -> str:
            s = (ts or "").strip()
            if not s:
                return ""
            try:
                dt = _parse_locomo_timestamp(s)
                return dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                return s[:16]

        for entry in memories:
            ts_str = ""
            if entry.metadata and entry.metadata.get("timestamp") is not None:
                ts_str = _short_ts(str(entry.metadata.get("timestamp")))
            scored = plugin.ranker.compute_spatiotemporal_score(query, entry, now, entry.layer)
            score = scored.score

            memory_text = entry.query if entry.response == "" else f"{entry.query}\n{entry.response}"
            if entry.metadata and entry.response == "":
                cq = str(entry.metadata.get("compressed_query") or "").strip()
                if cq:
                    memory_text = cq

            results.append({"memory": memory_text, "metadata": {"timestamp": ts_str}, "score": float(score)})

        results.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        k = int(max(1, min(int(top_k), 50)))

        budget_raw = (os.getenv("STMEMORY_SEARCH_CONTEXT_TOKEN_BUDGET") or "60").strip()
        try:
            budget = int(budget_raw)
        except Exception:
            budget = 60
        budget = max(0, budget)

        if budget <= 0:
            return {"results": results[:k], "relations": []}

        picked = []
        used = 0
        for r in results:
            ts = str(((r.get("metadata") or {}) or {}).get("timestamp") or "")
            mem = str(r.get("memory") or "")
            cost = _estimate_tokens(f"{ts}: {mem}" if ts else mem)
            if picked and used + cost > budget:
                break
            picked.append(r)
            used += cost
            if len(picked) >= k:
                break
        return {"results": picked, "relations": []}

    def _mock_answer_from_prompt(self, prompt: str) -> str:
        m = re.search(r"Question:\s*(.*?)\n\s*Answer:\s*$", prompt, re.S | re.I)
        if m:
            question = m.group(1).strip()
            if question in self._qa_map:
                return self._qa_map[question]
        return "Unable to answer."

    def _mock_judge_from_prompt(self, prompt: str) -> str:
        q = re.search(r"Question:\s*(.*?)\n", prompt, re.S | re.I)
        gold = re.search(r"Gold answer:\s*(.*?)\n", prompt, re.S | re.I)
        gen = re.search(r"Generated answer:\s*(.*?)\n", prompt, re.S | re.I)
        gold_text = (gold.group(1) if gold else "").strip()
        gen_text = (gen.group(1) if gen else "").strip()

        def norm(s: str) -> str:
            s = s.lower()
            s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
            return re.sub(r"\s+", " ", s).strip()

        ok = False
        if gold_text and gen_text:
            ng = norm(gold_text)
            nr = norm(gen_text)
            ok = (ng in nr) or (nr in ng)
        label = "CORRECT" if ok else "WRONG"
        return json.dumps({"label": label}, ensure_ascii=False)

    async def openai_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages") or []
        last = messages[-1] if messages else {}
        content = str((last or {}).get("content") or "")

        if "Gold answer:" in content and "Generated answer:" in content:
            answer = self._mock_judge_from_prompt(content)
        else:
            answer = self._mock_answer_from_prompt(content)

        prompt_tokens = _estimate_tokens(content)
        completion_tokens = _estimate_tokens(answer)
        self._token_count.prompt_tokens += prompt_tokens
        self._token_count.completion_tokens += completion_tokens
        self._token_count.total_tokens += prompt_tokens + completion_tokens

        return {
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": str(payload.get("model") or "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
            },
        }


def create_app(service: _STMemoryService, enable_mock_openai: bool) -> web.Application:
    app = web.Application()

    async def healthz(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def reset_token_count(_: web.Request) -> web.Response:
        service.reset_token_count()
        return web.json_response({"status": "success"})

    async def token_count(_: web.Request) -> web.Response:
        return web.json_response({"token_count": service.get_token_count().to_dict()})

    async def add_memories(request: web.Request) -> web.Response:
        body = await request.json()
        user_id = body.get("user_id")
        messages = body.get("messages") or []
        metadata = body.get("metadata") or {}
        if not user_id:
            return web.json_response({"error": "user_id is required"}, status=400)
        if not isinstance(messages, list):
            return web.json_response({"error": "messages must be a list"}, status=400)
        res = await service.add_memories(str(user_id), messages, metadata)
        return web.json_response(res)

    async def delete_memories(request: web.Request) -> web.Response:
        user_id = request.query.get("user_id")
        if not user_id:
            return web.json_response({"error": "user_id is required"}, status=400)
        service.delete_user(str(user_id))
        return web.json_response({"status": "success"})

    async def search(request: web.Request) -> web.Response:
        body = await request.json()
        user_id = body.get("user_id")
        query = body.get("query")
        top_k = body.get("top_k", 10)
        if not user_id or query is None:
            return web.json_response({"error": "user_id and query are required"}, status=400)
        try:
            top_k_i = int(top_k)
        except Exception:
            top_k_i = 10
        res = await service.search(str(user_id), str(query), top_k=top_k_i)
        return web.json_response(res)

    async def openai_chat(request: web.Request) -> web.Response:
        if not enable_mock_openai:
            return web.json_response({"error": "mock_openai disabled"}, status=404)
        body = await request.json()
        res = await service.openai_chat_completions(body)
        return web.json_response(res)

    app.router.add_get("/healthz", healthz)
    app.router.add_post("/reset_token_count", reset_token_count)
    app.router.add_get("/token_count", token_count)
    app.router.add_post("/memories", add_memories)
    app.router.add_delete("/memories", delete_memories)
    app.router.add_post("/search", search)
    app.router.add_post("/chat/completions", openai_chat)
    app.router.add_post("/v1/chat/completions", openai_chat)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="STmemory LOCOMO-compatible benchmark server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--enable-mock-openai", action="store_true", default=False)
    parser.add_argument("--dataset", default=None, help="LOCOMO dataset path for mock OpenAI answers")
    args = parser.parse_args()

    service = _STMemoryService()
    if args.enable_mock_openai:
        dataset_path = args.dataset or os.path.join(os.path.dirname(__file__), "dataset", "locomo10.json")
        dataset_path = os.path.abspath(dataset_path)
        service.load_qa_map(dataset_path)

    app = create_app(service, enable_mock_openai=bool(args.enable_mock_openai))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runner = web.AppRunner(app)

    async def _run() -> None:
        await runner.setup()
        site = web.TCPSite(runner, host=args.host, port=args.port)
        await site.start()

    loop.run_until_complete(_run())

    stop = loop.create_future()

    def _handle_stop(*_: Any) -> None:
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_a, **_k: _handle_stop())

    loop.run_until_complete(stop)
    loop.run_until_complete(runner.cleanup())


if __name__ == "__main__":
    main()
