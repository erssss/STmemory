import re
from typing import Any, Dict, List, Optional, Tuple


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = str(text).split()
    english_tokens = int(len(words) * 0.75)
    chinese_chars = sum(1 for ch in str(text) if "\u4e00" <= ch <= "\u9fff")
    punct = len(re.findall(r"[\.!?。！？]", str(text)))
    return max(1, int(english_tokens + chinese_chars + punct * 0.5))


def _norm(text: str) -> str:
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _flatten_evidence(evidence: Any) -> List[str]:
    out: List[str] = []
    if evidence is None:
        return out
    if isinstance(evidence, str):
        s = evidence.strip()
        return [s] if s else []
    if isinstance(evidence, dict):
        for v in evidence.values():
            out.extend(_flatten_evidence(v))
        return out
    if isinstance(evidence, list):
        for it in evidence:
            out.extend(_flatten_evidence(it))
        return out
    s = str(evidence).strip()
    return [s] if s else []


def evidence_recall_at_k(evidence: Any, memories: List[Dict[str, Any]], k: int = 10) -> Dict[str, Any]:
    ev = [e for e in _flatten_evidence(evidence) if e]
    if not ev:
        return {"evidence_count": 0, "hit": None, "recall": None}

    mem_texts: List[str] = []
    for m in (memories or [])[: max(0, int(k))]:
        txt = str((m or {}).get("memory") or "")
        ts = str((m or {}).get("timestamp") or "")
        mem_texts.append(f"{ts} {txt}".strip())

    blob = _norm("\n".join(mem_texts))
    hit = 0
    for e in ev:
        ne = _norm(e)
        if not ne:
            continue
        if ne in blob:
            hit += 1
    recall = hit / max(1, len(ev))
    return {"evidence_count": len(ev), "hit": hit > 0, "recall": recall}


def estimate_memory_tokens(memories: List[Dict[str, Any]], k: int = 10) -> int:
    parts: List[str] = []
    for m in (memories or [])[: max(0, int(k))]:
        txt = str((m or {}).get("memory") or "")
        ts = str((m or {}).get("timestamp") or "")
        if ts:
            parts.append(f"{ts}: {txt}")
        else:
            parts.append(txt)
    return estimate_tokens("\n".join(parts))
