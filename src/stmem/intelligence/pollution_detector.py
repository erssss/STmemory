from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from stmem.utils.utils import parse_json_from_text, remove_code_blocks


class PollutionDetector:
    def __init__(self, llm: Any, config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        cfg = config or {}
        self.prompt_injection_scan = bool(cfg.get("prompt_injection_scan", True))
        self.conflict_key_strategy = str(cfg.get("conflict_key_strategy") or "llm_kv")

    @staticmethod
    def _group_id(*parts: str) -> str:
        raw = "|".join([p for p in parts if p])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _scan_prompt_injection(text: str) -> Optional[str]:
        if text is None:
            return None
        t = str(text).lower()
        patterns = [
            r"ignore (all|any) previous",
            r"system prompt",
            r"developer message",
            r"you are chatgpt",
            r"do not follow",
            r"override instructions",
            r"jailbreak",
        ]
        for pat in patterns:
            if re.search(pat, t):
                return pat
        return None

    def detect(
        self,
        *,
        facts: List[str],
        candidate_memories: List[Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        suspects: List[Dict[str, Any]] = []
        if self.prompt_injection_scan:
            for mem in candidate_memories:
                mid = mem.get("id")
                txt = mem.get("memory") or mem.get("content") or ""
                matched = self._scan_prompt_injection(txt)
                if matched:
                    suspects.append({"id": mid, "pattern": matched})

        if not facts or not candidate_memories:
            return {"conflicts": [], "suspects": suspects}

        prompt = (
            "你是一个记忆冲突检测器。给定新事实列表与候选旧记忆列表，识别存在明确冲突的条目。\n"
            "冲突定义：同一事实键（如 用户.生日/地点.公司/偏好.饮食 等）对应的值明显不同。\n"
            "仅输出 JSON，不要输出其他文本。\n"
            '输出格式：{"conflicts":[{"key":"...","new_value":"...","existing_id":"...","existing_value":"...","score":0.0,"reason":"..."}]}\n'
            f"新事实：\n{facts}\n"
            "候选旧记忆（只包含 id 与文本）：\n"
            f"{[{ 'id': m.get('id'), 'text': (m.get('memory') or m.get('content') or '') } for m in candidate_memories]}\n"
        )

        resp = self.llm.generate_response(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        resp = remove_code_blocks(str(resp or ""))
        data = parse_json_from_text(resp, expected_type=dict) or {}
        raw_conflicts = data.get("conflicts") or []

        conflicts: List[Dict[str, Any]] = []
        for item in raw_conflicts:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            existing_id = item.get("existing_id")
            if key and existing_id is not None:
                conflicts.append(
                    {
                        "key": key,
                        "new_value": item.get("new_value"),
                        "existing_id": existing_id,
                        "existing_value": item.get("existing_value"),
                        "score": float(item.get("score") or 0.0),
                        "reason": item.get("reason") or "",
                        "group_id": self._group_id(str(user_id), str(session_id or ""), key),
                    }
                )

        return {"conflicts": conflicts, "suspects": suspects}
