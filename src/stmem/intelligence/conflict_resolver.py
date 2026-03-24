from __future__ import annotations

from typing import Any, Dict, List, Optional


class ConflictResolver:
    def __init__(self, storage: Any, llm: Any):
        self.storage = storage
        self.llm = llm

    def list_group(self, *, user_id: str, group_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        return self.storage.get_all_memories(user_id=user_id, limit=limit, filters={"_conflict_group_id": group_id})

    def suggest(self, *, user_id: str, group_id: str) -> Dict[str, Any]:
        memories = self.list_group(user_id=user_id, group_id=group_id)
        texts = [m.get("memory") or m.get("content") or "" for m in memories]
        prompt = (
            "你是记忆冲突修正助手。给定一组互相冲突的记忆文本，请给出：\n"
            "1) 冲突点摘要\n2) 建议保留的信息\n3) 建议丢弃或归档的信息\n"
            "只输出 JSON，不要输出其他文本。\n"
            '输出格式：{"summary":"...","keep":["..."],"drop":["..."]}\n'
            f"输入：\n{texts}"
        )
        resp = self.llm.generate_response(messages=[{"role": "user", "content": prompt}])
        return {"group_id": group_id, "suggestion": resp, "count": len(memories)}

    def merge(
        self,
        *,
        user_id: str,
        group_id: str,
        strategy: str = "summarize",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        memories = self.list_group(user_id=user_id, group_id=group_id)
        texts = [m.get("memory") or m.get("content") or "" for m in memories]
        prompt = (
            "你是记忆合并器。请把输入的一组冲突记忆合并成一条更一致的记忆，尽量保留可验证事实，避免臆测。\n"
            "只输出合并后的纯文本，不要输出 JSON。\n"
            f"输入：\n{texts}"
        )
        merged_text = self.llm.generate_response(messages=[{"role": "user", "content": prompt}])
        merged_text = str(merged_text or "").strip()
        if not merged_text:
            return {"group_id": group_id, "merged": False, "reason": "empty_merge"}

        new_meta = dict(metadata or {})
        new_meta.update(
            {
                "_conflict_group_id": group_id,
                "_conflict_state": "merged",
                "type": "conflict_merged",
                "source_count": len(memories),
            }
        )
        new_id = self.storage.add_memory({"content": merged_text, "user_id": user_id, "metadata": new_meta})

        for m in memories:
            mid = m.get("id")
            if mid is None:
                continue
            self.storage.update_memory(
                mid,
                {
                    "metadata": {
                        "_conflict_state": "resolved",
                        "archived": True,
                        "_merged_into": new_id,
                    }
                },
                user_id,
            )

        return {"group_id": group_id, "merged": True, "new_memory_id": new_id, "source_count": len(memories)}

