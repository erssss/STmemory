import json
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_METRICS_LOCK = threading.Lock()

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def extract_json(text):
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text
    return json_str


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


def _append_metrics(metrics_path: Optional[str], row: Dict[str, Any]) -> None:
    if not metrics_path:
        return
    try:
        line = json.dumps(row, ensure_ascii=False)
    except Exception:
        return
    try:
        with _METRICS_LOCK:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
    except Exception:
        return


def _append_jsonl(path: Optional[str], row: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        line = json.dumps(row, ensure_ascii=False)
    except Exception:
        return
    try:
        with _METRICS_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
    except Exception:
        return


def _mock_judge(gold_answer: str, generated_answer: str) -> int:
    gold = str(gold_answer or "").strip().lower()
    gen = str(generated_answer or "").strip().lower()
    if not gold or not gen:
        return 0
    if gold in gen:
        return 1
    gold_tokens = {t for t in re.split(r"\W+", gold) if t}
    gen_tokens = {t for t in re.split(r"\W+", gen) if t}
    if not gold_tokens or not gen_tokens:
        return 0
    overlap = len(gold_tokens & gen_tokens) / max(1, len(gold_tokens))
    return 1 if overlap >= 0.5 else 0


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


@dataclass
class LocomoJudgeConfig:
    use_mock_llm: bool = False
    model: str = "MiniMax-M2.7"
    metrics_path: Optional[str] = None
    llm_log_path: Optional[str] = None
    request_timeout_s: float = 60.0
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str = "You are a rigorous evaluator."
    llm_provider: str = ""
    openai_base_url: str = ""
    openai_api_key: str = ""


def evaluate_llm_judge(question, gold_answer, generated_answer, config: Optional[LocomoJudgeConfig] = None):
    cfg = config or LocomoJudgeConfig()
    if bool(cfg.use_mock_llm):
        score = _mock_judge(gold_answer, generated_answer)
        _append_metrics(cfg.metrics_path, {"type": "judge", "latency_s": 0.0, "success": True, "mock": True})
        prompt = ACCURACY_PROMPT.format(question=question, gold_answer=gold_answer, generated_answer=generated_answer)
        _append_jsonl(
            cfg.llm_log_path,
            {
                "type": "judge",
                "mock": True,
                "model": str(cfg.model or "MiniMax-M2.7"),
                "latency_s": 0.0,
                "prompt": prompt,
                "response": json.dumps({"label": "CORRECT" if score == 1 else "WRONG"}),
                "meta": {"question": question},
            },
        )
        return score

    provider = _normalize_llm_provider(cfg.llm_provider) or _normalize_llm_provider(os.getenv("LOCOMO_LLM_PROVIDER") or "")
    if not provider:
        raw_base = str(getattr(cfg, "openai_base_url", "") or "").strip() or str(os.getenv("OPENAI_BASE_URL") or "").strip()
        if "127.0.0.1:11434" in raw_base or "localhost:11434" in raw_base:
            provider = "ollama"
        elif raw_base:
            provider = "openai_compat"
        else:
            provider = "anthropic"

    prompt = ACCURACY_PROMPT.format(question=question, gold_answer=gold_answer, generated_answer=generated_answer)

    if provider == "ollama":
        raw_base = str(getattr(cfg, "openai_base_url", "") or "").strip() or str(os.getenv("OPENAI_BASE_URL") or "").strip() or "http://127.0.0.1:11434"
        base = _normalize_ollama_base_url(raw_base)
        url = f"{base}/api/chat"
        payload: Dict[str, Any] = {
            "model": str(cfg.model or ""),
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": str(cfg.system_prompt)},
                {"role": "user", "content": prompt},
            ],
        }
        temp = float(cfg.temperature)
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
            with urllib.request.urlopen(req, timeout=float(cfg.request_timeout_s)) as resp:
                raw = (resp.read() or b"").decode("utf-8", errors="replace")
        except Exception as e:
            raise RuntimeError(f"Ollama /api/chat judge request failed. url={url}") from e
        t2 = time.time()
        data = json.loads(raw) if raw.strip() else {}
        content = str((((data.get("message") or {}) or {}).get("content") or ""))
        try:
            label = json.loads(extract_json(content)).get("label")
        except Exception:
            label = None
        if label not in ("CORRECT", "WRONG"):
            s = content.upper()
            if "CORRECT" in s and "WRONG" not in s:
                label = "CORRECT"
            elif "WRONG" in s and "CORRECT" not in s:
                label = "WRONG"

        ok = bool(label in ("CORRECT", "WRONG"))
        _append_metrics(cfg.metrics_path, {"type": "judge", "latency_s": (t2 - t1), "success": ok, "provider": "ollama"})
        _append_jsonl(
            cfg.llm_log_path,
            {
                "type": "judge",
                "mock": False,
                "model": str(cfg.model or ""),
                "latency_s": (t2 - t1),
                "prompt": prompt,
                "response": content,
                "meta": {"question": question},
            },
        )
        return 1 if label == "CORRECT" else 0

    if provider == "openai_compat":
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai (shim) package is required for OpenAI-compatible judge") from e

        base_url = _normalize_openai_base_url(
            str(getattr(cfg, "openai_base_url", "") or "").strip()
            or str(os.getenv("OPENAI_BASE_URL") or "").strip()
            or "http://127.0.0.1:11434/v1"
        )
        api_key = str(getattr(cfg, "openai_api_key", "") or "").strip() or str(os.getenv("OPENAI_API_KEY") or "").strip()
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=float(cfg.request_timeout_s))
        t1 = time.time()
        try:
            resp = client.chat.completions.create(
                model=str(cfg.model or ""),
                messages=[
                    {"role": "system", "content": str(cfg.system_prompt)},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=float(cfg.temperature),
            )
        except Exception:
            resp = client.chat.completions.create(
                model=str(cfg.model or ""),
                messages=[
                    {"role": "system", "content": str(cfg.system_prompt)},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(cfg.temperature),
            )
        t2 = time.time()
        content = str(resp.choices[0].message.content or "")
        try:
            label = json.loads(extract_json(content)).get("label")
        except Exception:
            label = None
        if label not in ("CORRECT", "WRONG"):
            s = content.upper()
            if "CORRECT" in s and "WRONG" not in s:
                label = "CORRECT"
            elif "WRONG" in s and "CORRECT" not in s:
                label = "WRONG"

        ok = bool(label in ("CORRECT", "WRONG"))
        _append_metrics(cfg.metrics_path, {"type": "judge", "latency_s": (t2 - t1), "success": ok, "provider": "openai_compat"})
        _append_jsonl(
            cfg.llm_log_path,
            {
                "type": "judge",
                "mock": False,
                "model": str(cfg.model or ""),
                "latency_s": (t2 - t1),
                "prompt": prompt,
                "response": content,
                "meta": {"question": question},
            },
        )
        return 1 if label == "CORRECT" else 0

    try:
        import anthropic
    except Exception as e:
        raise RuntimeError("anthropic package is required for MiniMax calls") from e

    api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY (or ANTHROPIC_API_KEY) is not set")

    timeout_s = float(cfg.request_timeout_s)
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
    except TypeError:
        client = anthropic.Anthropic(api_key=api_key)

    t1 = time.time()
    try:
        try:
            message = client.messages.create(
                model=str(cfg.model or "MiniMax-M2.7"),
                max_tokens=int(cfg.max_tokens),
                temperature=float(cfg.temperature),
                system=str(cfg.system_prompt),
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
                model=str(cfg.model or "MiniMax-M2.7"),
                max_tokens=int(cfg.max_tokens),
                temperature=float(cfg.temperature),
                system=str(cfg.system_prompt),
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

    content = _extract_text_from_anthropic_message(message)
    try:
        label = json.loads(extract_json(content)).get("label")
        ok = bool(label in ("CORRECT", "WRONG"))
    except Exception:
        label = None
        ok = False
    _append_metrics(cfg.metrics_path, {"type": "judge", "latency_s": (t2 - t1), "success": ok})
    _append_jsonl(
        cfg.llm_log_path,
        {
            "type": "judge",
            "mock": False,
            "model": str(cfg.model or "MiniMax-M2.7"),
            "latency_s": (t2 - t1),
            "prompt": prompt,
            "response": content,
            "meta": {"question": question},
        },
    )
    if label == "CORRECT":
        return 1
    if label == "WRONG":
        return 0
    return 0
