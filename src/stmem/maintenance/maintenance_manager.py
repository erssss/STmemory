from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


class MaintenanceManager:
    def __init__(self, storage: Any, config: Optional[Dict[str, Any]] = None):
        self.storage = storage
        self.config = config or {}
        self.batch_size = int(self.config.get("batch_size", 200))
        self.archive_enabled = bool(self.config.get("archive_enabled", True))
        self.dry_run_default = bool(self.config.get("dry_run_default", True))
        self.ttl_days = self.config.get("ttl_days") or {}

    @staticmethod
    def _now_ts() -> int:
        return int(datetime.now(timezone.utc).timestamp())

    def _ttl_for(self, memory_type: Optional[str]) -> Optional[int]:
        if not memory_type:
            return None
        v = self.ttl_days.get(str(memory_type))
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    def run(
        self,
        *,
        user_id: str,
        run_id: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        is_dry_run = self.dry_run_default if dry_run is None else bool(dry_run)

        stats = {
            "user_id": user_id,
            "run_id": run_id,
            "dry_run": is_dry_run,
            "scanned": 0,
            "deleted": 0,
            "archived": 0,
            "skipped": 0,
            "errors": 0,
        }

        offset = 0
        now_ts = self._now_ts()
        while True:
            batch = self.storage.get_all_memories(
                user_id=user_id,
                run_id=run_id,
                limit=self.batch_size,
                offset=offset,
            )
            if not batch:
                break

            for mem in batch:
                stats["scanned"] += 1
                mid = mem.get("id")
                md = mem.get("metadata") or {}
                try:
                    if md.get("archived") is True:
                        stats["skipped"] += 1
                        continue

                    if mem.get("should_forget") is True:
                        if not is_dry_run:
                            self.storage.delete_memory(mid, user_id=user_id)
                        stats["deleted"] += 1
                        continue

                    created_ts = mem.get("created_at_ts")
                    if created_ts is None:
                        stats["skipped"] += 1
                        continue

                    ttl_days = self._ttl_for(mem.get("memory_type"))
                    if ttl_days is None:
                        stats["skipped"] += 1
                        continue

                    if now_ts - int(created_ts) >= ttl_days * 86400:
                        if self.archive_enabled:
                            if not is_dry_run:
                                self.storage.update_memory(
                                    mid,
                                    {
                                        "metadata": {
                                            "archived": True,
                                            "_archived_at": datetime.now(timezone.utc).isoformat(),
                                        }
                                    },
                                    user_id=user_id,
                                )
                            stats["archived"] += 1
                        else:
                            if not is_dry_run:
                                self.storage.delete_memory(mid, user_id=user_id)
                            stats["deleted"] += 1
                        continue

                    stats["skipped"] += 1
                except Exception:
                    stats["errors"] += 1

            offset += len(batch)

        return stats

