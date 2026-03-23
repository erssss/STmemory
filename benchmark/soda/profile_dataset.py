import argparse
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _parse_ts(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _extract_entities(text: str, limit: int = 20) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", str(text or ""))
    out: List[str] = []
    seen = set()
    for t in tokens:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _summary_stats(nums: List[float]) -> Dict[str, Any]:
    nums = [float(x) for x in nums if x is not None]
    if not nums:
        return {"n": 0}
    nums_sorted = sorted(nums)
    def pct(p: float) -> float:
        if not nums_sorted:
            return 0.0
        i = int(round((len(nums_sorted) - 1) * p))
        return float(nums_sorted[max(0, min(i, len(nums_sorted) - 1))])
    return {
        "n": len(nums),
        "mean": float(statistics.mean(nums)),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": float(max(nums)),
        "min": float(min(nums)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile LOCOMO dataset patterns for temporal/spatial memory")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f) or []

    qa_counts: List[int] = []
    turn_counts: List[int] = []
    char_counts: List[int] = []
    time_gaps_s: List[float] = []
    entity_counter = Counter()
    q_time_markers = Counter()
    by_cat = defaultdict(lambda: {"n": 0, "turns": [], "chars": [], "q_has_time": 0})

    time_words = ["today", "yesterday", "last", "ago", "tomorrow", "今天", "昨天", "前天", "上周", "上个月", "去年", "最近", "刚刚"]

    for item in data:
        qa = (item or {}).get("qa") or []
        qa_non_adv = 0
        for q in qa:
            try:
                c = int((q or {}).get("category", -1))
            except Exception:
                c = -1
            if c == 5:
                continue
            qa_non_adv += 1
            qtext = str((q or {}).get("question") or "")
            has_time = any(w in qtext.lower() for w in time_words)
            if has_time:
                q_time_markers["has_time_marker"] += 1
            by_cat[str(c)]["n"] += 1
            if has_time:
                by_cat[str(c)]["q_has_time"] += 1

        qa_counts.append(qa_non_adv)

        conv = (item or {}).get("conversation") or {}
        ts_list: List[datetime] = []
        turns = 0
        chars = 0
        for k, v in (conv or {}).items():
            if not isinstance(k, str) or not k.startswith("session_") or k.endswith("_date_time"):
                continue
            if not isinstance(v, list):
                continue
            dt_key = f"{k}_date_time"
            dt_val = conv.get(dt_key)
            dt = _parse_ts(str(dt_val or ""))
            if dt is not None:
                ts_list.append(dt)
            for chat in v:
                if not isinstance(chat, dict):
                    continue
                t = str(chat.get("text") or "")
                turns += 1
                chars += len(t)
                for ent in _extract_entities(t):
                    entity_counter[ent.lower()] += 1

        turn_counts.append(turns)
        char_counts.append(chars)

        ts_list = sorted(ts_list)
        for a, b in zip(ts_list, ts_list[1:]):
            gap = (b - a).total_seconds()
            if gap >= 0:
                time_gaps_s.append(gap)

    profile = {
        "dataset": os.path.basename(dataset_path),
        "conversations": len(data),
        "qa_non_adversarial": _summary_stats([float(x) for x in qa_counts]),
        "turns": _summary_stats([float(x) for x in turn_counts]),
        "chars": _summary_stats([float(x) for x in char_counts]),
        "session_time_gaps_seconds": _summary_stats(time_gaps_s),
        "top_entities": [{"entity": k, "count": int(v)} for k, v in entity_counter.most_common(50)],
        "question_time_markers": dict(q_time_markers),
        "by_category": {
            k: {
                "n": int(v["n"]),
                "q_has_time_ratio": (float(v["q_has_time"]) / float(v["n"])) if v["n"] else 0.0,
            }
            for k, v in by_cat.items()
        },
        "recommended_config": {
            "deep_expand_temporal_neighbors": 1 if _summary_stats(time_gaps_s).get("p50", 0) > 0 else 0,
            "compression_target_ratio": 0.5,
            "deep_vector_candidates": 50,
        },
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

