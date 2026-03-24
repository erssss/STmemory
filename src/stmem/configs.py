from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except Exception:
    from stmem.utils.pydantic_compat import BaseModel, Field


class ComponentConfig(BaseModel):
    provider: str
    config: Dict[str, Any] = Field(default_factory=dict)


class RerankerConfig(BaseModel):
    enabled: bool = False
    provider: str = "qwen"
    config: Dict[str, Any] = Field(default_factory=dict)


class IntelligentMemoryConfig(BaseModel):
    enabled: bool = True
    initial_retention: float = 1.0
    decay_rate: float = 0.1
    reinforcement_factor: float = 0.3
    working_threshold: float = 0.3
    short_term_threshold: float = 0.6
    long_term_threshold: float = 0.8
    fallback_to_simple_add: bool = False
    review_intervals: Optional[List[int]] = None


class MemoryDecayConfig(BaseModel):
    enabled: bool = True
    algorithm: str = "ebbinghaus"
    base_retention: float = 1.0
    forgetting_rate: float = 0.1
    reinforcement_factor: float = 0.3


class MemoryIsolationConfig(BaseModel):
    mode: str = "compat"
    default_user_id: str = "user"
    map_session_to_run: bool = True
    enforce_on_search: bool = True
    enforce_on_intelligent_add: bool = True
    allow_cross_session_search: bool = False
    system_metadata_prefix: str = "_"


class TopicIsolationConfig(BaseModel):
    enabled: bool = False
    provider: str = "llm"
    confidence_threshold: float = 0.7
    mode: str = "soft"
    store_key_topic_id: str = "_topic_id"
    store_key_topic_path: str = "_topic_path"
    store_key_topic_confidence: str = "_topic_confidence"


class TimeWindowConfig(BaseModel):
    enabled: bool = False
    default_days: int = 30
    long_term_days: int = 365
    max_recent: int = 500
    store_key_created_ts: str = "created_at_ts"
    store_key_updated_ts: str = "updated_at_ts"


class ValidityWeightsConfig(BaseModel):
    recency: float = 0.35
    importance: float = 0.25
    access: float = 0.2
    conflict_penalty: float = 0.2


class ValidityConfig(BaseModel):
    enabled: bool = False
    weights: ValidityWeightsConfig = Field(default_factory=ValidityWeightsConfig)
    conflict_demotion: float = 0.2


class PollutionDetectionConfig(BaseModel):
    enabled: bool = False
    run_on_add: bool = True
    run_on_intelligent_add: bool = True
    prompt_injection_scan: bool = True
    conflict_key_strategy: str = "llm_kv"
    mark_only: bool = True


class MaintenanceConfig(BaseModel):
    enabled: bool = False
    allow_global: bool = False
    dry_run_default: bool = True
    archive_enabled: bool = True
    archive_store: str = "sub_store"
    ttl_days: Dict[str, int] = Field(default_factory=lambda: {
        "working": 30,
        "short_term": 180,
        "long_term": 3650,
    })


class MemoryConfig(BaseModel):
    llm: Optional[ComponentConfig] = None
    embedder: Optional[ComponentConfig] = None
    vector_store: Optional[ComponentConfig] = None
    graph_store: Optional[ComponentConfig] = None

    reranker: RerankerConfig = Field(default_factory=RerankerConfig)

    intelligent_memory: Optional[IntelligentMemoryConfig] = None
    memory_decay: Optional[MemoryDecayConfig] = None

    timezone: Optional[Any] = None

    custom_fact_extraction_prompt: Optional[str] = None
    custom_update_memory_prompt: Optional[str] = None
    custom_importance_evaluation_prompt: Optional[str] = None

    telemetry: Optional[Dict[str, Any]] = None
    audit: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    query_rewrite: Optional[Dict[str, Any]] = None
    isolation: Optional[MemoryIsolationConfig] = None
    topic_isolation: Optional[TopicIsolationConfig] = None
    time_window: Optional[TimeWindowConfig] = None
    validity: Optional[ValidityConfig] = None
    pollution_detection: Optional[PollutionDetectionConfig] = None
    maintenance: Optional[MaintenanceConfig] = None

    sub_stores: Optional[List[Dict[str, Any]]] = None
