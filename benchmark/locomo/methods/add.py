import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib.error
import urllib.parse
import urllib.request
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable

load_dotenv()


class MemoryADD:
    def __init__(
        self,
        api_base_url: str,
        data_path: str = None,
        batch_size: int = 2,
        is_graph: bool = False,
        add_workers: int = 10,
        subset_indices: str = "",
        max_conversations: int = 0,
        add_future_timeout_s: float = 300.0,
    ):
        api_base_url = str(api_base_url or "").strip()
        if not api_base_url:
            raise ValueError("api_base_url is required")
        self.api_base_url = api_base_url
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.server_execution_time = 0.0
        self.request_count = 0
        self.request_times = []
        self._lock = threading.Lock()
        self.add_workers = int(add_workers)
        self.max_conversations = int(max_conversations)
        self.add_future_timeout_s = float(add_future_timeout_s)
        subset_indices_raw = str(subset_indices or "").strip()
        if subset_indices_raw:
            try:
                self.subset_indices = {int(x) for x in subset_indices_raw.split(",") if str(x).strip()}
            except Exception:
                self.subset_indices = None
        else:
            self.subset_indices = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                payload = {"messages": message, "user_id": user_id, "metadata": metadata or {}}
                start_time = time.time()
                url = f"{self.api_base_url}/memories"
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
                    with urllib.request.urlopen(request, timeout=120) as response:
                        status_code = int(getattr(response, "status", 0) or 0)
                        response_body = response.read() or b""
                except urllib.error.HTTPError as e:
                    status_code = int(getattr(e, "code", 0) or 0)
                    response_body = (e.read() if hasattr(e, "read") else b"") or b""
                end_time = time.time()

                response_text = response_body.decode("utf-8", errors="replace")
                if status_code == 200:
                    request_time = end_time - start_time
                    with self._lock:
                        self.server_execution_time += request_time
                        self.request_count += 1
                        self.request_times.append(request_time)
                    if not response_text.strip():
                        return {}
                    return json.loads(response_text)
                else:
                    raise Exception(f"API call failed with status {status_code}: {response_text}")
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp):
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

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
                messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
            elif chat["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
            else:
                raise ValueError(f"Unknown speaker: {chat['speaker']}")

        self.add_memories_for_speaker(speaker_a_user_id, messages, timestamp)
        self.add_memories_for_speaker(speaker_b_user_id, messages_reverse, timestamp)

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        max_workers = int(self.add_workers if self.add_workers is not None else max_workers)
        subset_indices = self.subset_indices

        items = []
        for orig_idx, item in enumerate(self.data):
            if subset_indices is not None and orig_idx not in subset_indices:
                continue
            items.append((orig_idx, item))

        if subset_indices is None and int(self.max_conversations) > 0:
            items = items[: int(self.max_conversations)]

        print("Deleting existing memories...")
        for idx, item in items:
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            self.delete_all_memories(speaker_a_user_id)
            self.delete_all_memories(speaker_b_user_id)

        all_tasks = []
        for idx, item in items:
            conversation = item["conversation"]

            for key in conversation.keys():
                if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                    continue
                all_tasks.append((item, idx, key))

        print(f"Processing {len(all_tasks)} tasks with {max_workers} workers...")
        future_timeout_s = float(self.add_future_timeout_s)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx, key) for item, idx, key in all_tasks]

            for future in tqdm(as_completed(futures), desc="Processing conversations", total=len(futures)):
                try:
                    future.result(timeout=future_timeout_s)
                except Exception as e:
                    print(f"Error processing conversation: {e}")

    def delete_all_memories(self, user_id, retries=3):
        for attempt in range(retries):
            try:
                params = urllib.parse.urlencode({"user_id": user_id})
                url = f"{self.api_base_url}/memories?{params}"
                request = urllib.request.Request(url, method="DELETE")
                response_body = b""
                status_code = 0
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        status_code = int(getattr(response, "status", 0) or 0)
                        response_body = response.read() or b""
                except urllib.error.HTTPError as e:
                    status_code = int(getattr(e, "code", 0) or 0)
                    response_body = (e.read() if hasattr(e, "read") else b"") or b""

                response_text = response_body.decode("utf-8", errors="replace")
                if status_code == 200:
                    if not response_text.strip():
                        return {}
                    return json.loads(response_text)
                else:
                    raise Exception(f"Delete API call failed with status {status_code}: {response_text}")

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise e
