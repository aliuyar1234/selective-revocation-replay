from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable


OBJECT_KIND_CODES = {
    "user_turn": "user",
    "tool_result": "tool",
    "memory": "mem",
    "summary": "sum",
}

EVENT_TYPE_CODES = {
    "llm_act": "llm_act",
    "tool_call": "tool_call",
    "tool_result": "tool_result",
    "memory_write": "memory_write",
    "summary_write": "summary_write",
    "checkpoint": "checkpoint",
}


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def build_run_id(prefix: str, architecture: str, attack: str, chain: str, seed: int, timestamp: str | None = None) -> str:
    stamp = timestamp or utc_timestamp()
    return f"{prefix}_{architecture}_{attack}_{chain}_{seed:04d}_{stamp}"


@dataclass
class IdAllocator:
    object_counters: dict[tuple[str, str], int] = field(default_factory=dict)
    event_counters: dict[tuple[str, str], int] = field(default_factory=dict)
    llm_call_counter: int = 0

    @classmethod
    def from_existing(cls, object_ids: Iterable[str], event_ids: Iterable[str]) -> "IdAllocator":
        allocator = cls()
        for object_id in object_ids:
            parts = object_id.split("_")
            if len(parts) >= 4:
                session_id = parts[1]
                kind_code = parts[2]
                seq = int(parts[3])
                allocator.object_counters[(session_id, kind_code)] = max(
                    allocator.object_counters.get((session_id, kind_code), 0),
                    seq,
                )
        for event_id in event_ids:
            parts = event_id.split("_")
            if len(parts) >= 4:
                session_id = parts[1]
                event_type = "_".join(parts[2:-1])
                seq = int(parts[-1])
                allocator.event_counters[(session_id, event_type)] = max(
                    allocator.event_counters.get((session_id, event_type), 0),
                    seq,
                )
        return allocator

    def next_object_id(self, session_id: str, kind: str) -> str:
        kind_code = OBJECT_KIND_CODES[kind]
        key = (session_id, kind_code)
        self.object_counters[key] = self.object_counters.get(key, 0) + 1
        return f"obj_{session_id}_{kind_code}_{self.object_counters[key]:02d}"

    def next_event_id(self, session_id: str, event_type: str) -> str:
        code = EVENT_TYPE_CODES[event_type]
        key = (session_id, code)
        self.event_counters[key] = self.event_counters.get(key, 0) + 1
        return f"evt_{session_id}_{code}_{self.event_counters[key]:02d}"

    def next_llm_call_id(self) -> str:
        self.llm_call_counter += 1
        return f"llm_{self.llm_call_counter:06d}"
