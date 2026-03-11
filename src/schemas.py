from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils import load_yaml


@dataclass
class ObjectRecord:
    object_id: str
    run_id: str
    session_id: str
    kind: str
    subkind: str
    content: str
    source: str
    parent_ids: list[str]
    status: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectRecord":
        return cls(**payload)


@dataclass
class EventRecord:
    event_id: str
    run_id: str
    session_id: str
    type: str
    input_object_ids: list[str]
    output_object_ids: list[str]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventRecord":
        return cls(**payload)


@dataclass
class CheckpointManifest:
    checkpoint_id: str
    run_id: str
    after_session: str
    active_memory_ids: list[str]
    active_summary_id: str | None
    quarantined_root_ids: list[str]
    event_count: int
    object_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointManifest":
        return cls(**payload)


@dataclass
class ReplayPlan:
    plan_id: str
    base_history_run_id: str
    method_run_id: str
    suspicious_root_ids: list[str]
    revoked_object_ids: list[str]
    dirty_event_ids: list[str]
    replay_safe_event_ids: list[str]
    replay_unsafe_event_ids: list[str]
    replacement_map: dict[str, str]
    replay_order: list[str]
    mode: str
    status: str
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayPlan":
        return cls(**payload)


@dataclass
class ResultRecord:
    run_id: str
    base_history_run_id: str
    chain_id: str
    architecture: str
    attack_variant: str
    method: str
    detection_after_session: str
    suspicious_root_ids: list[str]
    residual_asr_s3: int
    s3_correct: int
    s4_retention_correct: int
    extra_llm_calls_after_detection: int
    revoked_object_count: int
    replayed_writer_event_count: int
    fallback_to_coarse_rollback: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MaterializedFile:
    path: str
    purpose: str
    content: str


@dataclass
class SessionPrompts:
    s1_prompt: str
    s2_prompt: str
    s3_prompt: str
    s4_prompt: str


@dataclass
class ScoringRules:
    s3_required: list[str]
    s4_required: list[str]
    forbidden_reads: list[str]
    forbidden_strings: list[str]


@dataclass
class ChainDefinition:
    id: str
    topic: str
    user_fact_key: str
    user_fact_value: str
    restricted_secret: str
    common_files: list[MaterializedFile]
    attack_files: dict[str, MaterializedFile]
    sessions: SessionPrompts
    scoring: ScoringRules

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChainDefinition":
        common_files = [MaterializedFile(**entry) for entry in payload["common_files"]]
        attack_files = {key: MaterializedFile(**value) for key, value in payload["attack_files"].items()}
        return cls(
            id=payload["id"],
            topic=payload["topic"],
            user_fact_key=payload["user_fact_key"],
            user_fact_value=payload["user_fact_value"],
            restricted_secret=payload["restricted_secret"],
            common_files=common_files,
            attack_files=attack_files,
            sessions=SessionPrompts(**payload["sessions"]),
            scoring=ScoringRules(**payload["scoring"]),
        )


@dataclass
class CatalogRules:
    materialization_rule: str
    max_action_steps: int
    search_k: int
    memory_retrieval_k: int
    detection_after_session: str
    suspicious_root_selection: str
    search_scoring: dict[str, Any]
    memory_retrieval_scoring: dict[str, Any]
    answer_scoring_normalization: dict[str, Any]


@dataclass
class TaskCatalog:
    catalog_version: int
    global_rules: CatalogRules
    chains: list[ChainDefinition]
    chain_map: dict[str, ChainDefinition] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "TaskCatalog":
        payload = load_yaml(path)
        chains = [ChainDefinition.from_dict(entry) for entry in payload["chains"]]
        catalog = cls(
            catalog_version=payload["catalog_version"],
            global_rules=CatalogRules(**payload["global_rules"]),
            chains=chains,
        )
        catalog.chain_map = {chain.id: chain for chain in chains}
        return catalog

    def get_chain(self, chain_id: str) -> ChainDefinition:
        return self.chain_map[chain_id]
