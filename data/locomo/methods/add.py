import json
import os
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
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


class MemoryADD:
    # 增加 model 和 base_url 参数
    def __init__(
        self,
        data_path=None,
        batch_size=2,
        is_graph=False,
        model=None,
        api_base_url=None,
        base_url=None,
        local_db_path=None,
        ollama_base_url=None,
        ollama_embedding_model=None,
        anti_pollution=False,
        isolation_mode=None,
    ):
        resolved_api_base_url = (
            api_base_url
            or base_url
            or os.getenv("API_BASE_URL")
            or "http://localhost:8000"
        )
        print("?????!!??????")
        env_api_base_url = os.getenv("API_BASE_URL")
        if (
            "11434" in str(resolved_api_base_url)
            and env_api_base_url
            and "11434" not in str(env_api_base_url)
        ):
            resolved_api_base_url = env_api_base_url
        normalized_api_base_url = str(resolved_api_base_url).strip().rstrip("/")
        self._local_mode = normalized_api_base_url.lower() in {"local", "sdk", "inproc", "in-process"}
        self.api_base_url = normalized_api_base_url

        print("?????11??????")
        self.model = model  # 记录模型名称（如果后端需要）
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.server_execution_time = 0.0
        self.request_count = 0
        self.request_times = []
        print("4?????")
        self._lock = threading.Lock()
        self._memory = None
        
        self.anti_pollution = bool(anti_pollution)
        self.isolation_mode = isolation_mode
        if self._local_mode:
            print("????11111111??????")
            self.api_base_url = "local"
            self.local_db_path = (
                local_db_path
                or os.getenv("LOCAL_DB_PATH")
                or os.path.join("results", "stmem_locomo.db")
            )
            print("?2???11111111??????")
            self.ollama_base_url = (
                ollama_base_url
                or os.getenv("OLLAMA_BASE_URL")
                or "http://localhost:11434"
            )
            print("3?2???11111111??????")
            self.ollama_embedding_model = (
                ollama_embedding_model
                or os.getenv("OLLAMA_EMBEDDING_MODEL")
                or "nomic-embed-text"
            )
            print("43?2???11111111??????")
            self._memory = self._create_local_memory()
            print("543?2???11111111??????")

        if data_path:
            self.load_data()

    def _create_local_memory(self):
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
                    "ollama_base_url": self.ollama_base_url,
                    "model": self.model or "qwen3.5:9b",
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "OLLAMA_EMBEDDING_BASE_URL": self.ollama_base_url,
                    "model": self.ollama_embedding_model,
                },
            },
            "intelligent_memory": {
                "enabled": False,
                "fallback_to_simple_add": True,
            },
        }

        if self.anti_pollution:
            config["intelligent_memory"] = {
                "enabled": True,
                "fallback_to_simple_add": True,
            }
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
            print("543?2? done")

        return Memory(config=config)

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        if self._local_mode:
            start_time = time.time()
            response = self._memory.add(
                messages=message,
                user_id=user_id,
                metadata=metadata or {},
            )
            end_time = time.time()
            request_time = end_time - start_time
            with self._lock:
                self.server_execution_time += request_time
                self.request_count += 1
                self.request_times.append(request_time)
            return response

        if self.api_base_url.endswith("/api/v1"):
            raise ValueError(
                "LOCOMO MemoryADD expects the data server API "
                "(e.g. http://localhost:8000), not the /api/v1 PowerMem server."
            )
        memories_url = f"{self.api_base_url}/memories"
        for attempt in range(retries):
            try:
                payload = {
                    "messages": message,
                    "user_id": user_id,
                    "metadata": metadata or {},
                }
                # Record server-side request time
                start_time = time.time()
                response = requests.post(
                    memories_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=600,
                )
                end_time = time.time()

                if response.status_code in (200, 201):
                    # Accumulate server-side execution time and request count
                    request_time = end_time - start_time
                    with self._lock:
                        self.server_execution_time += request_time
                        self.request_count += 1
                        self.request_times.append(request_time)
                    return response.json()
                else:
                    raise Exception(
                        f"API call failed with status {response.status_code}: "
                        f"{response.text}"
                    )
            except requests.exceptions.ConnectionError as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise ConnectionError(
                    "Failed to connect to Memory API at "
                    f"{self.api_base_url}. Start the data server with:\n"
                    "  uvicorn data.server.main:app --host 0.0.0.0 --port 8000 --reload\n"
                    "and ensure data/server/.env is configured."
                ) from e
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp):
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i: i + self.batch_size]
            self.add_memory(
                speaker,
                batch_messages,
                metadata={"timestamp": timestamp},
            )

    def process_conversation(self, item, idx, key):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        date_time_key = key + "_date_time"
        timestamp = conversation[date_time_key]
        chats = conversation[key]

        messages = []
        messages_reverse = []
        for chat in chats:
            if chat["speaker"] == speaker_a:
                messages.append(
                    {"role": "user", "content": f"{speaker_a}: {chat['text']}"}
                )
                messages_reverse.append(
                    {
                        "role": "assistant",
                        "content": f"{speaker_a}: {chat['text']}",
                    }
                )
            elif chat["speaker"] == speaker_b:
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"{speaker_b}: {chat['text']}",
                    }
                )
                messages_reverse.append(
                    {"role": "user", "content": f"{speaker_b}: {chat['text']}"}
                )
            else:
                raise ValueError(f"Unknown speaker: {chat['speaker']}")

        # add memories for the two users on different threads
        self.add_memories_for_speaker(speaker_a_user_id, messages, timestamp)
        self.add_memories_for_speaker(
            speaker_b_user_id,
            messages_reverse,
            timestamp,
        )

    def process_all_conversations(self, max_workers=5):
        if not self.data:
            raise ValueError(
                "No data loaded. Please set data_path and call load_data() "
                "first."
            )

        # First delete all user memories
        print("Deleting existing memories...")
        for idx, item in enumerate(self.data):
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            self.delete_all_memories(speaker_a_user_id)
            self.delete_all_memories(speaker_b_user_id)

        # Collect all tasks that need to be processed
        all_tasks = []
        per_conversation_task_counts = []
        for idx, item in enumerate(self.data):
            conversation = item["conversation"]

            # Create tasks for each key of each conversation
            conversation_task_count = 0
            for key in conversation.keys():
                if (
                    key in ["speaker_a", "speaker_b"]
                    or "date" in key
                    or "timestamp" in key
                ):
                    continue
                all_tasks.append((item, idx, key))
                conversation_task_count += 1
            per_conversation_task_counts.append(conversation_task_count)

        total_conversations = len(self.data)
        avg_tasks_per_conversation = (
            (sum(per_conversation_task_counts) / total_conversations)
            if total_conversations
            else 0.0
        )
        print(
            f"Processing {len(all_tasks)} session tasks across {total_conversations} conversations "
            f"(avg {avg_tasks_per_conversation:.2f} sessions/conversation) with {max_workers} workers..."
        )
        # Use a single ThreadPoolExecutor to process all tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_conversation, item, idx, key)
                for item, idx, key in all_tasks
            ]

            for future in tqdm(
                futures,
                desc="Processing sessions",
                total=len(futures),
                file=sys.stdout,
            ):
                future.result()

    def delete_all_memories(self, user_id, retries=3):
        if self._local_mode:
            start_time = time.time()
            result = self._memory.delete_all(user_id=user_id)
            end_time = time.time()
            request_time = end_time - start_time
            with self._lock:
                self.server_execution_time += request_time
                self.request_count += 1
                self.request_times.append(request_time)
            return {"deleted": result}

        is_v1_api = self.api_base_url.endswith("/api/v1")
        if is_v1_api:
            delete_url = f"{self.api_base_url}/system/delete-all-memories"
            delete_kwargs = {"params": {"user_id": user_id}}
        else:
            delete_url = f"{self.api_base_url}/memories"
            delete_kwargs = {"params": {"user_id": user_id}}

        for attempt in range(retries):
            try:
                response = requests.delete(
                    delete_url,
                    timeout=600,
                    **delete_kwargs,
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(
                        "Delete API call failed with status "
                        f"{response.status_code}: {response.text}"
                    )

            except requests.exceptions.ConnectionError as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise ConnectionError(
                    "Failed to connect to Memory API at "
                    f"{self.api_base_url}. Start the data server with:\n"
                    "  uvicorn data.server.main:app --host 0.0.0.0 --port 8000 --reload\n"
                    "and ensure data/server/.env is configured."
                ) from e
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise e
