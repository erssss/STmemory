import json
import os
import re
import threading
import time
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


def evaluate_llm_judge(question, gold_answer, generated_answer):
    if os.getenv("LOCOMO_USE_MOCK_LLM") == "1":
        score = _mock_judge(gold_answer, generated_answer)
        _append_metrics(os.getenv("MINIMAX_METRICS_PATH"), {"type": "judge", "latency_s": 0.0, "success": True, "mock": True})
        prompt = ACCURACY_PROMPT.format(question=question, gold_answer=gold_answer, generated_answer=generated_answer)
        _append_jsonl(
            (os.getenv("LOCOMO_LLM_LOG_PATH") or "").strip() or None,
            {
                "type": "judge",
                "mock": True,
                "model": os.getenv("MODEL") or "MiniMax-M2.7",
                "latency_s": 0.0,
                "prompt": prompt,
                "response": json.dumps({"label": "CORRECT" if score == 1 else "WRONG"}),
                "meta": {"question": question},
            },
        )
        return score

    try:
        import anthropic
    except Exception as e:
        raise RuntimeError("anthropic package is required for MiniMax calls") from e

    api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY (or ANTHROPIC_API_KEY) is not set")

    timeout_s = float(os.getenv("LOCOMO_LLM_REQUEST_TIMEOUT_S") or os.getenv("MINIMAX_REQUEST_TIMEOUT_S") or "60")
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
    except TypeError:
        client = anthropic.Anthropic(api_key=api_key)

    prompt = ACCURACY_PROMPT.format(question=question, gold_answer=gold_answer, generated_answer=generated_answer)
    t1 = time.time()
    try:
        try:
            message = client.messages.create(
                model=os.getenv("MODEL") or "MiniMax-M2.7",
                max_tokens=int(os.getenv("MINIMAX_JUDGE_MAX_TOKENS") or os.getenv("MINIMAX_MAX_TOKENS") or "256"),
                temperature=float(os.getenv("MINIMAX_TEMPERATURE") or "0.0"),
                system=os.getenv("MINIMAX_SYSTEM_PROMPT") or "You are a rigorous evaluator.",
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
                model=os.getenv("MODEL") or "MiniMax-M2.7",
                max_tokens=int(os.getenv("MINIMAX_JUDGE_MAX_TOKENS") or os.getenv("MINIMAX_MAX_TOKENS") or "256"),
                temperature=float(os.getenv("MINIMAX_TEMPERATURE") or "0.0"),
                system=os.getenv("MINIMAX_SYSTEM_PROMPT") or "You are a rigorous evaluator.",
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
    metrics_path = os.getenv("MINIMAX_METRICS_PATH")
    try:
        label = json.loads(extract_json(content)).get("label")
        ok = bool(label in ("CORRECT", "WRONG"))
    except Exception:
        label = None
        ok = False
    _append_metrics(metrics_path, {"type": "judge", "latency_s": (t2 - t1), "success": ok})
    prompt = ACCURACY_PROMPT.format(question=question, gold_answer=gold_answer, generated_answer=generated_answer)
    _append_jsonl(
        (os.getenv("LOCOMO_LLM_LOG_PATH") or "").strip() or None,
        {
            "type": "judge",
            "mock": False,
            "model": os.getenv("MODEL") or "MiniMax-M2.7",
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
