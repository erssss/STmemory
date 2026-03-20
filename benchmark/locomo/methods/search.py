import json
import os
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib.error
import urllib.request
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable

load_dotenv()


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
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_api_key = openai_api_key
        else:
            raise ValueError("openai_api_key is not set")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if openai_base_url:
            self.openai_base_url = openai_base_url
        else:
            print("openai_base_url is not set, using default base url: https://api.openai.com/v1")
            self.openai_base_url = "https://api.openai.com/v1"
        self.openai_client = OpenAI(base_url=self.openai_base_url, api_key=self.openai_api_key)
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.server_execution_time = 0.0
        self.request_count = 0
        self.request_times = []
        self._lock = threading.Lock()

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        retries = 0
        while retries < max_retries:
            try:
                payload = {"query": query, "user_id": user_id}
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

        t1 = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model, messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
            )
        except Exception as e:
            print(f"Error processing question: {question[:100]}... Error: {e}")
            return (
                "Unable to process this question due to API error.",
                speaker_1_memories,
                speaker_2_memories,
                speaker_1_memory_time,
                speaker_2_memory_time,
                speaker_1_graph_memories,
                speaker_2_graph_memories,
                0.0,
            )
        t2 = time.time()
        response_time = t2 - t1
        return (
            response.choices[0].message.content,
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

        out = []
        for question_item in qa:
            out.append(self.process_question(question_item, speaker_a_user_id, speaker_b_user_id))
        return idx, out

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        max_workers = int(os.getenv("LOCOMO_SEARCH_WORKERS") or "1")
        max_conversations = int(os.getenv("LOCOMO_MAX_CONVERSATIONS") or "0")
        if max_conversations > 0:
            data = data[:max_conversations]

        if max_workers <= 1:
            for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
                conv_id, conv_results = self._process_single_conversation(idx, item)
                self.results[conv_id] = conv_results
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_single_conversation, idx, item) for idx, item in enumerate(data)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing conversations"):
                conv_id, conv_results = fut.result()
                self.results[conv_id] = conv_results
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)
