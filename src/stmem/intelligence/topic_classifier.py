from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

from stmem.utils.utils import parse_json_from_text, remove_code_blocks


class TopicClassifier:
    def __init__(self, llm: Any, config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        cfg = config or {}
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.7))
        self.mode = str(cfg.get("mode") or "soft")
        self.store_key_topic_id = str(cfg.get("store_key_topic_id") or "_topic_id")
        self.store_key_topic_path = str(cfg.get("store_key_topic_path") or "_topic_path")
        self.store_key_topic_confidence = str(
            cfg.get("store_key_topic_confidence") or "_topic_confidence"
        )

    @staticmethod
    def _normalize_topic_path(topic_path: str) -> str:
        return "/".join([seg.strip() for seg in str(topic_path).split("/") if seg.strip()])

    @staticmethod
    def _topic_id(topic_path: str) -> str:
        normalized = TopicClassifier._normalize_topic_path(topic_path).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def classify(self, text: str) -> Optional[Dict[str, Any]]:
        if text is None or str(text).strip() == "":
            return None

        prompt = (
            "请从输入文本中抽取一个主题路径（最多 3 层），并给出置信度。\n"
            "只输出 JSON，不要输出其他文本。\n"
            '输出格式：{"topic_path":"一级/二级/三级","confidence":0.0}\n'
            "输入：\n"
            f"{text}"
        )
        resp = self.llm.generate_response(messages=[{"role": "user", "content": prompt}])
        resp = remove_code_blocks(str(resp or ""))
        data = parse_json_from_text(resp, expected_type=dict)
        if not data:
            return None

        topic_path = self._normalize_topic_path(data.get("topic_path") or "")
        if not topic_path:
            return None

        confidence = data.get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        return {
            "topic_path": topic_path,
            "topic_id": self._topic_id(topic_path),
            "confidence": max(0.0, min(1.0, confidence)),
        }

    def apply_to_metadata(self, metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        md = dict(metadata or {})
        classified = self.classify(text)
        if not classified:
            return md

        if self.mode == "soft" and classified["confidence"] < self.confidence_threshold:
            return md

        md.setdefault(self.store_key_topic_path, classified["topic_path"])
        md.setdefault(self.store_key_topic_id, classified["topic_id"])
        md.setdefault(self.store_key_topic_confidence, classified["confidence"])
        return md

