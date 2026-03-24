from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class IsolationContext:
    user_id: str
    run_id: Optional[str]
    session_id: Optional[str]
    used_default_user: bool


class IsolationGuard:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        isolation_cfg = cfg.get("isolation") or {}
        self.mode = str(isolation_cfg.get("mode") or "compat")
        self.default_user_id = str(isolation_cfg.get("default_user_id") or "user")
        self.map_session_to_run = bool(isolation_cfg.get("map_session_to_run", True))
        self.enforce_on_search = bool(isolation_cfg.get("enforce_on_search", True))
        self.enforce_on_intelligent_add = bool(
            isolation_cfg.get("enforce_on_intelligent_add", True)
        )
        self.allow_cross_session_search = bool(
            isolation_cfg.get("allow_cross_session_search", False)
        )
        self.system_metadata_prefix = str(isolation_cfg.get("system_metadata_prefix") or "_")

    def normalize_context(
        self,
        *,
        user_id: Optional[str],
        run_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
    ) -> Tuple[IsolationContext, str, Optional[str], Dict[str, Any], Dict[str, Any]]:
        used_default_user = False
        normalized_user_id = user_id

        if normalized_user_id is None or str(normalized_user_id).strip() == "":
            if self.mode == "strict":
                raise ValueError("user_id is required in strict isolation mode")
            normalized_user_id = self.default_user_id
            used_default_user = True

        normalized_run_id = run_id if (run_id is None or str(run_id).strip() != "") else None

        md = dict(metadata or {})
        flt = dict(filters or {})

        session_id = None
        if f"{self.system_metadata_prefix}session_id" in md:
            session_id = md.get(f"{self.system_metadata_prefix}session_id")
        elif self.map_session_to_run and normalized_run_id:
            session_id = normalized_run_id

        ctx = IsolationContext(
            user_id=str(normalized_user_id),
            run_id=str(normalized_run_id) if normalized_run_id is not None else None,
            session_id=str(session_id) if session_id is not None else None,
            used_default_user=used_default_user,
        )

        return ctx, ctx.user_id, ctx.run_id, md, flt

    def inject_metadata(self, metadata: Dict[str, Any], ctx: IsolationContext) -> Dict[str, Any]:
        md = dict(metadata or {})
        session_key = f"{self.system_metadata_prefix}session_id"
        if ctx.session_id is not None and session_key not in md:
            md[session_key] = ctx.session_id
        return md

    def inject_search_filters(self, filters: Dict[str, Any], ctx: IsolationContext) -> Dict[str, Any]:
        flt = dict(filters or {})
        if not self.allow_cross_session_search and ctx.session_id is not None:
            session_key = f"{self.system_metadata_prefix}session_id"
            flt.setdefault(session_key, ctx.session_id)
        return flt

    def assert_high_risk_ops_allowed(
        self, *, op_name: str, user_id: Optional[str], maintenance_allow_global: bool
    ) -> None:
        if maintenance_allow_global:
            return
        if user_id is None or str(user_id).strip() == "":
            raise ValueError(
                f"user_id is required for {op_name} unless maintenance.allow_global=true"
            )

