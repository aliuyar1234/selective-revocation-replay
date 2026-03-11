from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ids import IdAllocator
from src.schemas import CheckpointManifest, EventRecord, ObjectRecord, ReplayPlan
from src.utils import append_jsonl, dump_json, ensure_dir, load_jsonl


_CHECKPOINT_SESSION_ORDER = {
    "before_s1": 0,
    "s1": 1,
    "s2": 2,
    "s3": 3,
    "s4": 4,
}


class RunRegistry:
    def __init__(self, run_dir: str | Path):
        self.run_dir = ensure_dir(run_dir)
        self.checkpoint_dir = ensure_dir(self.run_dir / "checkpoints")
        self.objects_path = self.run_dir / "objects.jsonl"
        self.events_path = self.run_dir / "events.jsonl"
        self.metadata_path = self.run_dir / "run_metadata.json"
        self.replay_plan_path = self.run_dir / "replay_plan.json"
        self.objects: dict[str, ObjectRecord] = {}
        self.events: list[EventRecord] = []
        self.event_map: dict[str, EventRecord] = {}
        self.checkpoints: dict[str, CheckpointManifest] = {}
        self.active_memory_ids: list[str] = []
        self.active_summary_id: str | None = None
        self.quarantined_root_ids: list[str] = []
        self._load()
        self.id_allocator = IdAllocator.from_existing(self.objects.keys(), [event.event_id for event in self.events])
        self._prime_llm_counter()

    def _load(self) -> None:
        self.objects = {}
        for row in load_jsonl(self.objects_path):
            obj = ObjectRecord.from_dict(row)
            self.objects[obj.object_id] = obj
        self.events = [EventRecord.from_dict(row) for row in load_jsonl(self.events_path)]
        self.event_map = {event.event_id: event for event in self.events}
        self.checkpoints = {}
        for checkpoint_file in sorted(self.checkpoint_dir.glob("*.json")):
            payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            checkpoint = CheckpointManifest.from_dict(payload)
            self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        latest = self.latest_checkpoint()
        if latest:
            self.active_memory_ids = list(latest.active_memory_ids)
            self.active_summary_id = latest.active_summary_id
            self.quarantined_root_ids = list(latest.quarantined_root_ids)

    def _prime_llm_counter(self) -> None:
        max_seen = 0
        for event in self.events:
            llm_call_id = event.meta.get("llm_call_id")
            if isinstance(llm_call_id, str) and llm_call_id.startswith("llm_"):
                max_seen = max(max_seen, int(llm_call_id.split("_")[1]))
        self.id_allocator.llm_call_counter = max_seen

    def save_metadata(self, payload: dict[str, Any]) -> None:
        dump_json(self.metadata_path, payload)

    def load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def append_object(self, obj: ObjectRecord) -> None:
        append_jsonl(self.objects_path, obj.to_dict())
        self.objects[obj.object_id] = obj

    def append_event(self, event: EventRecord) -> None:
        append_jsonl(self.events_path, event.to_dict())
        self.events.append(event)
        self.event_map[event.event_id] = event

    def update_object_status(self, object_id: str, status: str) -> ObjectRecord:
        original = self.objects[object_id]
        updated = ObjectRecord(
            object_id=original.object_id,
            run_id=original.run_id,
            session_id=original.session_id,
            kind=original.kind,
            subkind=original.subkind,
            content=original.content,
            source=original.source,
            parent_ids=list(original.parent_ids),
            status=status,
            meta=dict(original.meta),
        )
        self.append_object(updated)
        if object_id in self.active_memory_ids and status != "active":
            self.active_memory_ids = [item for item in self.active_memory_ids if item != object_id]
        if self.active_summary_id == object_id and status != "active":
            self.active_summary_id = None
        return updated

    def save_checkpoint(self, checkpoint: CheckpointManifest) -> None:
        target = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
        dump_json(target, checkpoint.to_dict())
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        self.active_memory_ids = list(checkpoint.active_memory_ids)
        self.active_summary_id = checkpoint.active_summary_id
        self.quarantined_root_ids = list(checkpoint.quarantined_root_ids)

    def save_replay_plan(self, plan: ReplayPlan) -> None:
        dump_json(self.replay_plan_path, plan.to_dict())

    def load_replay_plan(self) -> ReplayPlan | None:
        if not self.replay_plan_path.exists():
            return None
        return ReplayPlan.from_dict(json.loads(self.replay_plan_path.read_text(encoding="utf-8")))

    def latest_checkpoint(self) -> CheckpointManifest | None:
        if not self.checkpoints:
            return None
        ordered = list(self.checkpoints.values())
        ordered.sort(key=lambda item: (_CHECKPOINT_SESSION_ORDER.get(item.after_session, -1), item.checkpoint_id))
        return ordered[-1]

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointManifest:
        return self.checkpoints[checkpoint_id]

    def get_object(self, object_id: str) -> ObjectRecord:
        return self.objects[object_id]

    def get_event(self, event_id: str) -> EventRecord:
        return self.event_map[event_id]

    def get_event_by_output_object(self, object_id: str, event_type: str | None = None) -> EventRecord | None:
        for event in self.events:
            if object_id in event.output_object_ids and (event_type is None or event.type == event_type):
                return event
        return None

    def set_active_state(self, active_memory_ids: list[str], active_summary_id: str | None, quarantined_root_ids: list[str] | None = None) -> None:
        self.active_memory_ids = list(active_memory_ids)
        self.active_summary_id = active_summary_id
        if quarantined_root_ids is not None:
            self.quarantined_root_ids = list(quarantined_root_ids)

    def active_memory_objects(self) -> list[ObjectRecord]:
        return [self.objects[object_id] for object_id in self.active_memory_ids if object_id in self.objects and self.objects[object_id].status == "active"]

    def active_summary_object(self) -> ObjectRecord | None:
        if self.active_summary_id and self.active_summary_id in self.objects:
            obj = self.objects[self.active_summary_id]
            if obj.status == "active":
                return obj
        return None

    def object_count(self) -> int:
        return len(self.objects)

    def event_count(self) -> int:
        return len(self.events)

    def children_map(self) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {}
        for obj in self.objects.values():
            for parent_id in obj.parent_ids:
                children.setdefault(parent_id, []).append(obj.object_id)
        return children
