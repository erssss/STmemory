import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LocomoThresholds:
    max_add_p95_s: Optional[float] = None
    max_search_p95_s: Optional[float] = None


def _flatten_evaluation_metrics(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for conv_id, items in (metrics or {}).items():
        for item in items or []:
            rows.append(
                {
                    "conversation_id": conv_id,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "response": item.get("response", ""),
                    "category": item.get("category", ""),
                    "bleu_score": item.get("bleu_score", ""),
                    "f1_score": item.get("f1_score", ""),
                    "llm_score": item.get("llm_score", ""),
                }
            )
    return rows


def write_evaluation_csv(evaluation_metrics_path: str, csv_path: str) -> int:
    with open(evaluation_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows = _flatten_evaluation_metrics(metrics)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["conversation_id", "category", "question", "answer", "response", "bleu_score", "f1_score", "llm_score"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize_metrics(evaluation_metrics_path: str) -> Dict[str, Any]:
    with open(evaluation_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows = _flatten_evaluation_metrics(metrics)

    def to_float(x: Any) -> Optional[float]:
        try:
            if x is None or x == "":
                return None
            return float(x)
        except Exception:
            return None

    bleu = [to_float(r["bleu_score"]) for r in rows]
    f1 = [to_float(r["f1_score"]) for r in rows]
    llm = [to_float(r["llm_score"]) for r in rows]
    bleu_v = [v for v in bleu if v is not None]
    f1_v = [v for v in f1 if v is not None]
    llm_v = [v for v in llm if v is not None]

    by_cat: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cat = str(r.get("category", ""))
        by_cat.setdefault(cat, {"count": 0, "bleu": [], "f1": [], "llm": []})
        by_cat[cat]["count"] += 1
        bv = to_float(r.get("bleu_score"))
        fv = to_float(r.get("f1_score"))
        lv = to_float(r.get("llm_score"))
        if bv is not None:
            by_cat[cat]["bleu"].append(bv)
        if fv is not None:
            by_cat[cat]["f1"].append(fv)
        if lv is not None:
            by_cat[cat]["llm"].append(lv)

    def mean(vals: List[float]) -> Optional[float]:
        return (sum(vals) / len(vals)) if vals else None

    cat_summary: Dict[str, Any] = {}
    for cat, d in by_cat.items():
        cat_summary[cat] = {
            "count": d["count"],
            "bleu_mean": mean(d["bleu"]),
            "f1_mean": mean(d["f1"]),
            "llm_mean": mean(d["llm"]),
        }

    return {
        "count": len(rows),
        "bleu_mean": mean(bleu_v),
        "f1_mean": mean(f1_v),
        "llm_mean": mean(llm_v),
        "by_category": cat_summary,
    }


def parse_latency_from_evaluation_txt(evaluation_txt_path: str) -> Dict[str, Any]:
    if not os.path.exists(evaluation_txt_path):
        return {}
    with open(evaluation_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    def parse_block(action: str) -> Dict[str, Any]:
        m = re.search(rf"action:\s*{re.escape(action)}\s*\n(.*?)(?:\n\s*\n|$)", text, re.S | re.I)
        if not m:
            return {}
        block = m.group(1)
        p95 = re.search(r"P95 request time:\s*([0-9.]+)\s*seconds", block, re.I)
        avg = re.search(r"Average request time:\s*([0-9.]+)\s*seconds", block, re.I)
        total = re.search(r"Total server execution time:\s*([0-9.]+)\s*seconds", block, re.I)
        reqs = re.search(r"Total requests:\s*([0-9]+)", block, re.I)
        return {
            "p95_s": float(p95.group(1)) if p95 else None,
            "avg_s": float(avg.group(1)) if avg else None,
            "total_s": float(total.group(1)) if total else None,
            "requests": int(reqs.group(1)) if reqs else None,
        }

    return {"add": parse_block("add"), "search": parse_block("search")}


def check_thresholds(latency: Dict[str, Any], thresholds: LocomoThresholds) -> Tuple[bool, List[str]]:
    ok = True
    reasons: List[str] = []

    add_p95 = ((latency.get("add") or {}).get("p95_s"))
    search_p95 = ((latency.get("search") or {}).get("p95_s"))

    if thresholds.max_add_p95_s is not None and add_p95 is not None:
        if float(add_p95) > float(thresholds.max_add_p95_s):
            ok = False
            reasons.append(f"add_p95_s {add_p95:.4f} > {thresholds.max_add_p95_s:.4f}")
    if thresholds.max_search_p95_s is not None and search_p95 is not None:
        if float(search_p95) > float(thresholds.max_search_p95_s):
            ok = False
            reasons.append(f"search_p95_s {search_p95:.4f} > {thresholds.max_search_p95_s:.4f}")

    return ok, reasons

