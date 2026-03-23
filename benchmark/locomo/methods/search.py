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
    def __init__(self, output_path, top_k=10, filter_memories=False, is_graph=False):
        api_base_url = os.getenv("API_BASE_URL")
        if api_base_url:
            self.api_base_url = api_base_url
        else:
            raise ValueError("api_base_url is not set")
        model = os.getenv("MODEL")
        if model:
            self.model = model
        else:
            raise ValueError("model is not set")
        minimax_api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if minimax_api_key:
            self.minimax_api_key = minimax_api_key
        else:
            self.minimax_api_key = ""
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
        self._metrics_path = os.getenv("MINIMAX_METRICS_PATH")
        self._llm_log_lock = threading.Lock()
        self._llm_log_path = (os.getenv("LOCOMO_LLM_LOG_PATH") or "").strip() or None

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
                    with urllib.request.urlopen(request, timeout=60) as response:
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

        timeout_s = float(os.getenv("LOCOMO_LLM_REQUEST_TIMEOUT_S") or os.getenv("MINIMAX_REQUEST_TIMEOUT_S") or "60")
        try:
            client = anthropic.Anthropic(api_key=self.minimax_api_key, timeout=timeout_s)
        except TypeError:
            client = anthropic.Anthropic(api_key=self.minimax_api_key)
        t1 = time.time()
        try:
            try:
                message = client.messages.create(
                    model=self.model,
                    max_tokens=int(os.getenv("MINIMAX_MAX_TOKENS") or "512"),
                    temperature=float(os.getenv("MINIMAX_TEMPERATURE") or "0.0"),
                    system=os.getenv("MINIMAX_SYSTEM_PROMPT") or "You are a helpful assistant.",
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
                    max_tokens=int(os.getenv("MINIMAX_MAX_TOKENS") or "512"),
                    temperature=float(os.getenv("MINIMAX_TEMPERATURE") or "0.0"),
                    system=os.getenv("MINIMAX_SYSTEM_PROMPT") or "You are a helpful assistant.",
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

        if os.getenv("LOCOMO_USE_MOCK_LLM") == "1":
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

        max_qa = int(os.getenv("LOCOMO_MAX_QA") or "0")
        if max_qa > 0:
            qa = (qa or [])[:max_qa]

        out = []
        for question_item in qa:
            out.append(self.process_question(question_item, speaker_a_user_id, speaker_b_user_id))
        return idx, out

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        max_workers = int(os.getenv("LOCOMO_SEARCH_WORKERS") or "1")
        subset_indices_raw = (os.getenv("LOCOMO_SUBSET_INDICES") or "").strip()
        subset_indices: Optional[set] = None
        if subset_indices_raw:
            try:
                subset_indices = {int(x) for x in subset_indices_raw.split(",") if str(x).strip()}
            except Exception:
                subset_indices = None

        items: List[Tuple[int, Any]] = []
        for orig_idx, item in enumerate(data):
            if subset_indices is not None and orig_idx not in subset_indices:
                continue
            items.append((orig_idx, item))

        max_conversations = int(os.getenv("LOCOMO_MAX_CONVERSATIONS") or "0")
        if subset_indices is None and max_conversations > 0:
            items = items[:max_conversations]

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
