from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


def build_time_window_filters(
    config: Optional[Dict[str, Any]], *, now: Optional[datetime] = None
) -> Dict[str, Any]:
    cfg = config or {}
    if not cfg.get("enabled"):
        return {}

    days = int(cfg.get("default_days", 30))
    store_key = str(cfg.get("store_key_created_ts") or "created_at_ts")
    now_dt = now or datetime.now(timezone.utc)
    start = now_dt - timedelta(days=days)
    start_ts = int(start.timestamp())
    return {f"{store_key}_gte": start_ts}

