import json
import os
import re
import time
import sys
from collections import defaultdict

import requests
from jinja2 import Template
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(
        iterable,
        total=None,
        desc=None,
        leave=True,
        file=None,
        mininterval=0.2,
        **kwargs,
    ):
        if file is None:
            file = sys.stdout
        if total is None:
            try:
                total = len(iterable)
            except Exception:
                total = None
        prefix = (str(desc).strip() + ": ") if desc else ""
        last_render = 0.0
        count = 0
        last_line = ""
        for item in iterable:
            yield item
            count += 1
            now = time.time()
            if (now - last_render) < float(mininterval) and (not total or count < total):
                continue
            if total:
                last_line = f"{prefix}{count}/{total}"
            else:
                last_line = f"{prefix}{count}"
            try:
                file.write("\r" + last_line)
                file.flush()
            except Exception:
                pass
            last_render = now
        if last_line:
            try:
                if leave:
                    file.write("\n")
                else:
                    file.write("\r" + (" " * len(last_line)) + "\r")
                file.flush()
            except Exception:
                pass

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()


class MemorySearch:
    def __init__(
        self,
        output_path,
        top_k=10,
        filter_memories=False,
        is_graph=False,
        model=None,
        api_base_url=None,
        openai_base_url=None,
        base_url=None,
        local_db_path=None,
        ollama_base_url=None,
        ollama_embedding_model=None,
        ollama_chat_timeout=600,
        ollama_num_predict=64,
        ollama_keep_alive="-1",
        context_k=10,
        max_conversations=0,
        max_questions_per_conversation=0,
        use_full_context=False,
        anti_pollution=False,
        isolation_mode=None,
    ):
        resolved_api_base_url = api_base_url or os.getenv("API_BASE_URL") or (
            "http://localhost:8000"
        )
        resolved_openai_base_url = (
            openai_base_url
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:11434"
        )

        if base_url:
            base_url_str = str(base_url)
            if "/v1" in base_url_str or "11434" in base_url_str:
                resolved_openai_base_url = base_url_str
            else:
                resolved_api_base_url = base_url_str

        normalized_api_base_url = str(resolved_api_base_url).strip().rstrip("/")
        self._local_mode = normalized_api_base_url.lower() in {"local", "sdk", "inproc", "in-process"}
        self.api_base_url = normalized_api_base_url

        self.model = model or os.getenv("MODEL") or "llama3"
        self.use_full_context = bool(
            use_full_context
            or str(os.getenv("FULL_CONTEXT", "")).strip().lower() in {"1", "true", "yes", "y"}
        )

        base_str = str(resolved_openai_base_url).rstrip("/")
        if base_str.endswith("/v1"):
            base_str = base_str[:-3].rstrip("/")
        self.ollama_base_url = base_str
        self.local_db_path = (
            local_db_path
            or os.getenv("LOCAL_DB_PATH")
            or os.path.join("results", "stmem_locomo.db")
        )
        self.ollama_embedding_model = (
            ollama_embedding_model
            or os.getenv("OLLAMA_EMBEDDING_MODEL")
            or "nomic-embed-text"
        )
        self.ollama_chat_timeout = int(ollama_chat_timeout) if ollama_chat_timeout else 600
        self.ollama_num_predict = int(ollama_num_predict) if ollama_num_predict else 64
        self.ollama_keep_alive = (
            ollama_keep_alive
            if ollama_keep_alive is not None and str(ollama_keep_alive).strip() != ""
            else "-1"
        )
        keep_alive_str = str(self.ollama_keep_alive).strip()
        if keep_alive_str == "-1":
            self._ollama_keep_alive_payload = -1
        else:
            self._ollama_keep_alive_payload = self.ollama_keep_alive
        self.context_k = int(context_k) if context_k else 10
        self.max_conversations = int(max_conversations) if max_conversations else 0
        self.max_questions_per_conversation = (
            int(max_questions_per_conversation) if max_questions_per_conversation else 0
        )
        self.anti_pollution = bool(anti_pollution)
        self.isolation_mode = isolation_mode
        self._memory = None
        if self._local_mode and not self.use_full_context:
            self.api_base_url = "local"
            self._memory = self._create_local_memory(ollama_base_url=ollama_base_url)

        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.server_execution_time = 0.0
        self.request_count = 0
        self.request_times = []
        self.ollama_request_times = []
        self.ollama_prompt_eval_counts = []
        self.ollama_eval_counts = []
        self.ollama_total_durations = []

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

        self._pin_ollama_model_if_needed()

    def _call_ollama_chat(self, prompt, num_predict):
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_predict": int(num_predict) if num_predict else self.ollama_num_predict},
            "keep_alive": self._ollama_keep_alive_payload,
            "think": False,
        }
        r = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.ollama_chat_timeout,
        )
        r.raise_for_status()
        data = r.json() if r.content else {}
        response_content = (data.get("message") or {}).get("content", "")
        meta = {
            "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
            "eval_count": int(data.get("eval_count") or 0),
            "total_duration": int(data.get("total_duration") or 0),
        }
        return response_content, meta

    def _pin_ollama_model_if_needed(self):
        keep_alive_str = str(self.ollama_keep_alive).strip()
        if not keep_alive_str or keep_alive_str == "0":
            return

        try:
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [{"role": "user", "content": "ping"}],
                "options": {"num_predict": 1},
                "keep_alive": self._ollama_keep_alive_payload,
            }
            requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=min(30, int(self.ollama_chat_timeout) if self.ollama_chat_timeout else 30),
            ).raise_for_status()
        except Exception:
            return

    def _create_local_memory(self, ollama_base_url=None):
        import sys

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        src_path = os.path.join(repo_root, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from stmem import Memory

        db_dir = os.path.dirname(os.path.abspath(self.local_db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        resolved_ollama_base_url = (
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )

        config = {
            "vector_store": {
                "provider": "sqlite",
                "config": {
                    "database_path": self.local_db_path,
                    "collection_name": "memories_locomo",
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "ollama_base_url": resolved_ollama_base_url,
                    "model": self.model or "qwen3.5:9b",
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "OLLAMA_EMBEDDING_BASE_URL": resolved_ollama_base_url,
                    "model": self.ollama_embedding_model,
                },
            },
            "intelligent_memory": {
                "enabled": True,
                "fallback_to_simple_add": True,
            },
        }

        if self.anti_pollution:
            config["pollution_detection"] = {
                "enabled": True,
                "run_on_add": True,
                "run_on_intelligent_add": True,
                "prompt_injection_scan": True,
                "mark_only": True,
            }
            config["isolation"] = {
                "mode": str(self.isolation_mode or "strict"),
                "default_user_id": "user",
                "map_session_to_run": True,
                "enforce_on_search": True,
                "enforce_on_intelligent_add": True,
                "allow_cross_session_search": True,
            }

        return Memory(config=config)

    def _iter_locomo_messages(self, conversation):
        if not isinstance(conversation, dict):
            return

        session_items = []
        for k, v in conversation.items():
            m = re.fullmatch(r"session_(\d+)", str(k))
            if not m:
                continue
            session_no = int(m.group(1))
            session_items.append((session_no, v))

        for session_no, session in sorted(session_items, key=lambda x: x[0]):
            date_time = conversation.get(f"session_{session_no}_date_time")
            if not isinstance(session, list):
                continue
            for idx, msg in enumerate(session):
                if not isinstance(msg, dict):
                    continue
                yield session_no, idx, date_time, msg

    def _build_full_context_memories(self, conversation, speaker_name):
        memories = []
        for session_no, idx, date_time, msg in self._iter_locomo_messages(conversation):
            speaker = msg.get("speaker")
            text = msg.get("text")
            if speaker != speaker_name or not isinstance(text, str) or not text.strip():
                continue
            dia_id = msg.get("dia_id")
            if isinstance(dia_id, str) and dia_id.strip():
                stamp = dia_id.strip()
            else:
                stamp = f"S{session_no}:{idx + 1}"
            if isinstance(date_time, str) and date_time.strip():
                stamp = f"{date_time.strip()} | {stamp}"
            memories.append(
                {
                    "memory": f"{speaker_name}: {text.strip()}",
                    "timestamp": stamp,
                    "score": 1.0,
                }
            )
        return memories

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        if self._local_mode:
            start_time = time.time()
            memories = self._memory.search(
                query=query,
                user_id=user_id,
                limit=self.top_k,
            )
            end_time = time.time()
            request_time = end_time - start_time
            self.server_execution_time += request_time
            self.request_count += 1
            self.request_times.append(request_time)
        else:
            is_v1_api = self.api_base_url.endswith("/api/v1")
            if is_v1_api:
                search_url = f"{self.api_base_url}/memories/search"
            else:
                search_url = f"{self.api_base_url}/search"

            retries = 0
            while retries < max_retries:
                try:
                    payload = {
                        "query": query,
                        "user_id": user_id,
                    }
                    if self.filter_memories:
                        payload["filters"] = {}

                    start_time = time.time()
                    response = requests.post(
                        search_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=600,
                    )
                    end_time = time.time()

                    if response.status_code == 200:
                        memories = response.json()
                        request_time = end_time - start_time
                        self.server_execution_time += request_time
                        self.request_count += 1
                        self.request_times.append(request_time)
                        break
                    else:
                        raise Exception(
                            "Search API call failed with status "
                            f"{response.status_code}: {response.text}"
                        )
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    time.sleep(retry_delay)

        if not self.is_graph:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = [
                {
                    "source": relation["source"],
                    "relationship": relation["relationship"],
                    "target": relation["target"],
                }
                for relation in memories["relations"]
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(
        self,
        speaker_1_user_id,
        speaker_2_user_id,
        question,
        answer,
        category,
        conversation=None,
    ):
        if self.use_full_context:
            speaker_a = None
            speaker_b = None
            if isinstance(conversation, dict):
                speaker_a = conversation.get("speaker_a")
                speaker_b = conversation.get("speaker_b")

            speaker_1_name = (
                speaker_1_user_id.split("_")[0] if isinstance(speaker_1_user_id, str) else ""
            )
            speaker_2_name = (
                speaker_2_user_id.split("_")[0] if isinstance(speaker_2_user_id, str) else ""
            )
            if isinstance(speaker_a, str) and speaker_a.strip():
                speaker_1_name = speaker_a.strip() if speaker_1_name == speaker_a.strip() else speaker_1_name
            if isinstance(speaker_b, str) and speaker_b.strip():
                speaker_2_name = speaker_b.strip() if speaker_2_name == speaker_b.strip() else speaker_2_name

            speaker_1_memories = self._build_full_context_memories(conversation, speaker_1_name)
            speaker_2_memories = self._build_full_context_memories(conversation, speaker_2_name)
            speaker_1_graph_memories = None
            speaker_2_graph_memories = None
            speaker_1_memory_time = 0.0
            speaker_2_memory_time = 0.0
            effective_k = len(speaker_1_memories)
        else:
            (
                speaker_1_memories,
                speaker_1_graph_memories,
                speaker_1_memory_time,
            ) = self.search_memory(speaker_1_user_id, question)
            (
                speaker_2_memories,
                speaker_2_graph_memories,
                speaker_2_memory_time,
            ) = self.search_memory(speaker_2_user_id, question)
            effective_k = self.context_k if self.context_k > 0 else len(speaker_1_memories)
        search_1_memory = [
            f"{item['timestamp']}: {item['memory']}"
            for item in speaker_1_memories[:effective_k]
        ]
        search_2_memory = [
            f"{item['timestamp']}: {item['memory']}"
            for item in speaker_2_memories[:effective_k]
        ]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(
                speaker_1_graph_memories,
                indent=4,
            ),
            speaker_2_graph_memories=json.dumps(
                speaker_2_graph_memories,
                indent=4,
            ),
            question=question,
        )
        answer_prompt = (
            answer_prompt
            + "\n\nReturn only the final answer (<= 6 words). No explanation."
        )
        prompt_chars = len(answer_prompt)

        t1 = time.time()
        meta = {"prompt_eval_count": 0, "eval_count": 0, "total_duration": 0}
        try:
            response_content, meta = self._call_ollama_chat(answer_prompt, self.ollama_num_predict)
        except requests.exceptions.Timeout as e:
            reduced_k = max(3, effective_k // 2) if effective_k else 3
            reduced_num_predict = min(self.ollama_num_predict, 64)
            print(
                "Ollama /api/chat timeout. "
                f"prompt_chars={prompt_chars}, context_k={effective_k}, "
                f"retry_context_k={reduced_k}, retry_num_predict={reduced_num_predict}"
            )
            try:
                reduced_search_1 = [
                    f"{item['timestamp']}: {item['memory']}"
                    for item in speaker_1_memories[:reduced_k]
                ]
                reduced_search_2 = [
                    f"{item['timestamp']}: {item['memory']}"
                    for item in speaker_2_memories[:reduced_k]
                ]
                reduced_prompt = template.render(
                    speaker_1_user_id=speaker_1_user_id.split("_")[0],
                    speaker_2_user_id=speaker_2_user_id.split("_")[0],
                    speaker_1_memories=json.dumps(reduced_search_1, indent=4),
                    speaker_2_memories=json.dumps(reduced_search_2, indent=4),
                    speaker_1_graph_memories=json.dumps(
                        speaker_1_graph_memories,
                        indent=4,
                    ),
                    speaker_2_graph_memories=json.dumps(
                        speaker_2_graph_memories,
                        indent=4,
                    ),
                    question=question,
                )
                reduced_prompt = (
                    reduced_prompt
                    + "\n\nReturn only the final answer (<= 6 words). No explanation."
                )
                response_content, meta = self._call_ollama_chat(reduced_prompt, reduced_num_predict)
            except Exception as e2:
                print(
                    f"Error processing question after retry: {question[:100]}... Error: {e2}"
                )
                return (
                    "Unable to process this question due to API error.",
                    speaker_1_memories,
                    speaker_2_memories,
                    speaker_1_memory_time,
                    speaker_2_memory_time,
                    speaker_1_graph_memories,
                    speaker_2_graph_memories,
                    0.0,
                    0,
                    0,
                    0,
                )
        except Exception as e:
            print(f"Error processing question: {question[:100]}... Error: {e}")
            # Return a default response for failed questions
            return (
                "Unable to process this question due to API error.",
                speaker_1_memories,
                speaker_2_memories,
                speaker_1_memory_time,
                speaker_2_memory_time,
                speaker_1_graph_memories,
                speaker_2_graph_memories,
                0.0,
                0,
                0,
                0,
            )
        t2 = time.time()
        response_time = t2 - t1
        self.ollama_request_times.append(response_time)
        self.ollama_prompt_eval_counts.append(int(meta.get("prompt_eval_count") or 0))
        self.ollama_eval_counts.append(int(meta.get("eval_count") or 0))
        self.ollama_total_durations.append(int(meta.get("total_duration") or 0))
        return (
            response_content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            int(meta.get("prompt_eval_count") or 0),
            int(meta.get("eval_count") or 0),
            int(meta.get("total_duration") or 0),
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
            ollama_prompt_eval_count,
            ollama_eval_count,
            ollama_total_duration,
        ) = self.answer_question(
            speaker_a_user_id,
            speaker_b_user_id,
            question,
            answer,
            category,
            conversation=val.get("_conversation"),
        )

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
            "ollama_prompt_eval_count": ollama_prompt_eval_count,
            "ollama_eval_count": ollama_eval_count,
            "ollama_total_duration": ollama_total_duration,
        }

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        total_conversations = (
            min(len(data), self.max_conversations)
            if self.max_conversations and self.max_conversations > 0
            else len(data)
        )
        for idx, item in tqdm(
            enumerate(data[:total_conversations]),
            total=total_conversations,
            desc="Processing conversations",
            file=sys.stdout,
        ):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            effective_qa = (
                qa[: self.max_questions_per_conversation]
                if self.max_questions_per_conversation
                and self.max_questions_per_conversation > 0
                else qa
            )
            for question_item in tqdm(
                effective_qa,
                total=len(effective_qa),
                desc=f"Processing questions for conversation {idx}",
                leave=False,
                file=sys.stdout,
            ):  
                if isinstance(question_item, dict):
                    question_item["_conversation"] = conversation
                result = self.process_question(
                    question_item,
                    speaker_a_user_id,
                    speaker_b_user_id,
                )
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
