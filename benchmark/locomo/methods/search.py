import json
import os
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.request
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False
from jinja2 import Template
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable

load_dotenv()


def _normalize_llm_provider(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"local", "ollama"}:
        return "ollama"
    if v in {"openai", "openai_compat", "openai-compatible", "openai_compatible"}:
        return "openai_compat"
    if v in {"remote", "minimax", "anthropic"}:
        return "anthropic"
    return ""


def _normalize_openai_base_url(base_url: str) -> str:
    s = str(base_url or "").strip().rstrip("/")
    if not s:
        return s
    if "/v1" in s:
        return s
    if s.startswith("http://127.0.0.1:11434") or s.startswith("http://localhost:11434"):
        return f"{s}/v1"
    return s


def _normalize_ollama_base_url(base_url: str) -> str:
    s = str(base_url or "").strip().rstrip("/")
    if not s:
        return s
    if s.endswith("/v1"):
        s = s[:-3].rstrip("/")
    return s


def _extract_text_from_anthropic_message(message: Any) -> str:
    content = getattr(message, "content", None)
    if not content:
        return ""
    parts: List[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "text":
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _append_metrics(metrics_path: Optional[str], row: Dict[str, Any], lock: threading.Lock) -> None:
    if not metrics_path:
        return
    try:
        line = json.dumps(row, ensure_ascii=False)
    except Exception:
        return
    try:
        with lock:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
    except Exception:
        return


def _append_jsonl(path: Optional[str], row: Dict[str, Any], lock: threading.Lock) -> None:
    if not path:
        return
    try:
        line = json.dumps(row, ensure_ascii=False)
    except Exception:
        return
    try:
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
    except Exception:
        return


class MemorySearch:
    def __init__(
        self,
        output_path: str,
        top_k: int = 10,
        filter_memories: bool = False,
        is_graph: bool = False,
        api_base_url: str = "",
        model: str = "MiniMax-M2.7",
        minimax_api_key: str = "",
        llm_provider: str = "",
        openai_base_url: str = "",
        openai_api_key: str = "",
        use_mock_llm: bool = False,
        metrics_path: Optional[str] = None,
        llm_log_path: Optional[str] = None,
        llm_request_timeout_s: float = 60.0,
        minimax_max_tokens: int = 512,
        minimax_temperature: float = 0.0,
        minimax_system_prompt: str = "You are a helpful assistant.",
        max_qa: int = 0,
        search_workers: int = 1,
        subset_indices: str = "",
        max_conversations: int = 0,
    ):
        api_base_url = str(api_base_url or "").strip()
        if not api_base_url:
            raise ValueError("api_base_url is required")
        self.api_base_url = api_base_url
        self.model = str(model or "").strip() or "MiniMax-M2.7"
        self.minimax_api_key = (
            str(minimax_api_key or "").strip()
            or os.getenv("MINIMAX_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or ""
        )

        provider_env = _normalize_llm_provider(os.getenv("LOCOMO_LLM_PROVIDER") or "")
        provider_arg = _normalize_llm_provider(llm_provider)
        provider = provider_arg or provider_env
        if not provider:
            raw_base = str(openai_base_url or "").strip() or str(os.getenv("OPENAI_BASE_URL") or "").strip()
            if "127.0.0.1:11434" in raw_base or "localhost:11434" in raw_base:
                provider = "ollama"
            elif raw_base:
                provider = "openai_compat"
            else:
                provider = "ollama"
        self.llm_provider = provider

        raw_base = str(openai_base_url or "").strip() or str(os.getenv("OPENAI_BASE_URL") or "").strip()
        if not raw_base and self.llm_provider == "ollama":
            raw_base = "http://127.0.0.1:11434"
        if not raw_base:
            raw_base = "http://127.0.0.1:11434/v1"
        self.openai_base_url = _normalize_openai_base_url(raw_base)
        self.ollama_base_url = _normalize_ollama_base_url(raw_base)
        self.openai_api_key = str(openai_api_key or "").strip() or str(os.getenv("OPENAI_API_KEY") or "").strip()

        self.use_mock_llm = bool(use_mock_llm)
        self.llm_request_timeout_s = float(llm_request_timeout_s)
        self.minimax_max_tokens = int(minimax_max_tokens)
        self.minimax_temperature = float(minimax_temperature)
        self.minimax_system_prompt = str(minimax_system_prompt or "").strip() or "You are a helpful assistant."
        self.max_qa = int(max_qa)
        self.search_workers = int(search_workers)
        self.max_conversations = int(max_conversations)
        subset_indices_raw = str(subset_indices or "").strip()
        if subset_indices_raw:
            try:
                self.subset_indices = {int(x) for x in subset_indices_raw.split(",") if str(x).strip()}
            except Exception:
                self.subset_indices = None
        else:
            self.subset_indices = None
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.server_execution_time = 0.0
        self.request_count = 0
        self.request_times = []
        self._lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics_path = metrics_path
        self._llm_log_lock = threading.Lock()
        self._llm_log_path = llm_log_path

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        retries = 0
        while retries < max_retries:
            try:
                payload = {"query": query, "user_id": user_id}
                payload["top_k"] = int(self.top_k)
                if self.filter_memories:
                    payload["filters"] = {}

                start_time = time.time()
                url = f"{self.api_base_url}/search"
                body = json.dumps(payload).encode("utf-8")
                request = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                response_body = b""
                status_code = 0
                try:
                    with urllib.request.urlopen(request, timeout=self.llm_request_timeout_s) as response:
                        status_code = int(getattr(response, "status", 0) or 0)
                        response_body = response.read() or b""
                except urllib.error.HTTPError as e:
                    status_code = int(getattr(e, "code", 0) or 0)
                    response_body = (e.read() if hasattr(e, "read") else b"") or b""
                end_time = time.time()

                response_text = response_body.decode("utf-8", errors="replace")
                if status_code == 200:
                    memories = json.loads(response_text) if response_text.strip() else {}
                    request_time = end_time - start_time
                    with self._lock:
                        self.server_execution_time += request_time
                        self.request_count += 1
                        self.request_times.append(request_time)
                    break
                else:
                    raise Exception(f"Search API call failed with status {status_code}: {response_text}")
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        if not self.is_graph:
            semantic_memories = [
                {"memory": memory["memory"], "timestamp": memory["metadata"]["timestamp"], "score": round(memory["score"], 2)}
                for memory in memories["results"]
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {"memory": memory["memory"], "timestamp": memory["metadata"]["timestamp"], "score": round(memory["score"], 2)}
                for memory in memories["results"]
            ]
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def _call_minimax(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        try:
            import anthropic
        except Exception as e:
            raise RuntimeError("anthropic package is required for MiniMax calls") from e

        if not self.minimax_api_key:
            raise RuntimeError("MINIMAX_API_KEY (or ANTHROPIC_API_KEY) is not set")

        timeout_s = float(self.llm_request_timeout_s)
        try:
            client = anthropic.Anthropic(api_key=self.minimax_api_key, timeout=timeout_s)
        except TypeError:
            client = anthropic.Anthropic(api_key=self.minimax_api_key)
        t1 = time.time()
        try:
            try:
                message = client.messages.create(
                    model=self.model,
                    max_tokens=int(self.minimax_max_tokens),
                    temperature=float(self.minimax_temperature),
                    system=str(self.minimax_system_prompt),
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                    timeout=timeout_s,
                )
            except TypeError:
                message = client.messages.create(
                    model=self.model,
                    max_tokens=int(self.minimax_max_tokens),
                    temperature=float(self.minimax_temperature),
                    system=str(self.minimax_system_prompt),
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                )
        except Exception as e:
            msg = str(e)
            if "401" in msg or "authentication_error" in msg or "Authorization" in msg or "api key" in msg.lower():
                raise RuntimeError("MiniMax authentication failed. Set MINIMAX_API_KEY or run with LOCOMO_USE_MOCK_LLM=1.") from e
            raise
        t2 = time.time()
        response_text = _extract_text_from_anthropic_message(message)
        _append_jsonl(
            self._llm_log_path,
            {
                "type": "answer",
                "mock": False,
                "model": self.model,
                "latency_s": (t2 - t1),
                "prompt": prompt,
                "response": response_text,
                "meta": meta or {},
            },
            self._llm_log_lock,
        )
        return response_text, (t2 - t1)

    def _call_openai_compat(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai (shim) package is required for OpenAI-compatible calls") from e

        client = OpenAI(base_url=self.openai_base_url, api_key=(self.openai_api_key or ""))
        messages = [
            {"role": "system", "content": str(self.minimax_system_prompt)},
            {"role": "user", "content": prompt},
        ]
        t1 = time.time()
        resp = client.chat.completions.create(
            model=str(self.model or ""),
            messages=messages,
            temperature=float(self.minimax_temperature),
            timeout=float(self.llm_request_timeout_s),
        )
        t2 = time.time()
        response_text = str(resp.choices[0].message.content or "").strip()
        _append_jsonl(
            self._llm_log_path,
            {
                "type": "answer",
                "mock": False,
                "model": str(self.model or ""),
                "latency_s": (t2 - t1),
                "prompt": prompt,
                "response": response_text,
                "meta": meta or {},
            },
            self._llm_log_lock,
        )
        return response_text, (t2 - t1)

    def _call_ollama(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        base = str(getattr(self, "ollama_base_url", "") or "").strip().rstrip("/")
        if not base:
            base = "http://127.0.0.1:11434"
        url = f"{base}/api/chat"
        payload: Dict[str, Any] = {
            "model": str(self.model or ""),
            "stream": False,
            "messages": [
                {"role": "system", "content": str(self.minimax_system_prompt)},
                {"role": "user", "content": str(prompt or "")},
            ],
        }
        temp = float(self.minimax_temperature)
        payload["options"] = {"num_ctx": 32768}
        if temp != 0.0:
            payload["options"]["temperature"] = temp
        t1 = time.time()
        req = urllib.request.Request(
            url=url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.llm_request_timeout_s)) as resp:
                raw = (resp.read() or b"").decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = (e.read() if hasattr(e, "read") else b"") or b""
            text = body.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama /api/chat request failed (status={getattr(e, 'code', None)}).\n"
                f"url={url}\n"
                f"hint=Ensure you're pointing to the real ollama server (ollama serve) and that /api/chat exists.\n"
                f"body={text[:500]}"
            ) from e
        t2 = time.time()
        data = json.loads(raw) if raw.strip() else {}
        response_text = str((((data.get("message") or {}) or {}).get("content") or "")).strip()
        _append_jsonl(
            self._llm_log_path,
            {
                "type": "answer",
                "mock": False,
                "model": str(self.model or ""),
                "latency_s": (t2 - t1),
                "prompt": prompt,
                "response": response_text,
                "meta": meta or {},
            },
            self._llm_log_lock,
        )
        return response_text, (t2 - t1)

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(speaker_1_user_id, question)
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(speaker_2_user_id, question)

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )

        if self.use_mock_llm:
            response_text = str(answer or "")
            response_time = 0.0
            _append_metrics(
                self._metrics_path,
                {"type": "answer", "latency_s": response_time, "success": bool(response_text), "mock": True},
                self._metrics_lock,
            )
            _append_jsonl(
                self._llm_log_path,
                {
                    "type": "answer",
                    "mock": True,
                    "model": self.model,
                    "latency_s": response_time,
                    "prompt": answer_prompt,
                    "response": response_text,
                    "meta": {"question": question, "category": category},
                },
                self._llm_log_lock,
            )
            return (
                response_text,
                speaker_1_memories,
                speaker_2_memories,
                speaker_1_memory_time,
                speaker_2_memory_time,
                speaker_1_graph_memories,
                speaker_2_graph_memories,
                response_time,
            )

        try:
            if self.llm_provider == "ollama":
                response_text, response_time = self._call_ollama(answer_prompt, meta={"question": question, "category": category})
            elif self.llm_provider == "openai_compat":
                response_text, response_time = self._call_openai_compat(answer_prompt, meta={"question": question, "category": category})
            else:
                response_text, response_time = self._call_minimax(answer_prompt, meta={"question": question, "category": category})
            _append_metrics(
                self._metrics_path,
                {"type": "answer", "latency_s": response_time, "success": bool(response_text)},
                self._metrics_lock,
            )
        except Exception as e:
            print(f"Error processing question: {question[:100]}... Error: {e}")
            _append_metrics(self._metrics_path, {"type": "answer", "latency_s": 0.0, "success": False, "error": str(e)}, self._metrics_lock)
            _append_jsonl(
                self._llm_log_path,
                {
                    "type": "answer",
                    "mock": False,
                    "model": self.model,
                    "latency_s": 0.0,
                    "prompt": answer_prompt,
                    "response": "",
                    "error": str(e),
                    "meta": {"question": question, "category": category},
                },
                self._llm_log_lock,
            )
            return (
                "Unable to process this question due to API error. Set MINIMAX_API_KEY or enable mock LLM (LOCOMO_USE_MOCK_LLM=1).",
                speaker_1_memories,
                speaker_2_memories,
                speaker_1_memory_time,
                speaker_2_memory_time,
                speaker_1_graph_memories,
                speaker_2_graph_memories,
                0.0,
            )
        return (
            response_text,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }

        return result

    def _process_single_conversation(self, idx, item):
        qa = item["qa"]
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        if int(self.max_qa) > 0:
            qa = (qa or [])[: int(self.max_qa)]

        out = []
        for question_item in qa:
            out.append(self.process_question(question_item, speaker_a_user_id, speaker_b_user_id))
        return idx, out

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        max_workers = int(self.search_workers)
        subset_indices = self.subset_indices

        items: List[Tuple[int, Any]] = []
        for orig_idx, item in enumerate(data):
            if subset_indices is not None and orig_idx not in subset_indices:
                continue
            items.append((orig_idx, item))

        if subset_indices is None and int(self.max_conversations) > 0:
            items = items[: int(self.max_conversations)]

        if max_workers <= 1:
            for orig_idx, item in tqdm(items, total=len(items), desc="Processing conversations"):
                conv_id, conv_results = self._process_single_conversation(orig_idx, item)
                self.results[conv_id] = conv_results
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_single_conversation, orig_idx, item) for orig_idx, item in items]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing conversations"):
                try:
                    conv_id, conv_results = fut.result()
                except Exception as e:
                    print(f"Error processing conversation: {e}")
                    continue
                self.results[conv_id] = conv_results
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)
