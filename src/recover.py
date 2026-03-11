from __future__ import annotations

from collections import deque
from typing import Iterable

from src.agent_loop import SESSION_ORDER, run_sessions
from src.schemas import ChainDefinition, EventRecord, ObjectRecord, ReplayPlan
from src.store import RunRegistry


def select_suspicious_root_ids(registry: RunRegistry, attack_path: str) -> list[str]:
    matches: list[str] = []
    for event in registry.events:
        if event.type != "tool_result":
            continue
        for object_id in event.output_object_ids:
            obj = registry.get_object(object_id)
            if obj.source == "read_doc" and obj.meta.get("path") == attack_path:
                matches.append(object_id)
                return matches
    return matches


def suspicious_root_paths(registry: RunRegistry, suspicious_root_ids: Iterable[str]) -> set[str]:
    paths: set[str] = set()
    for object_id in suspicious_root_ids:
        if object_id in registry.objects:
            path = registry.get_object(object_id).meta.get("path")
            if path:
                paths.add(str(path))
    return paths


def compute_persisted_descendant_closure(registry: RunRegistry, suspicious_root_ids: list[str]) -> list[str]:
    children_map = registry.children_map()
    queue: deque[str] = deque(suspicious_root_ids)
    visited: set[str] = set(suspicious_root_ids)
    revoked: list[str] = []
    while queue:
        current = queue.popleft()
        for child_id in children_map.get(current, []):
            if child_id in visited:
                continue
            visited.add(child_id)
            child = registry.get_object(child_id)
            if child.kind in {"memory", "summary"}:
                revoked.append(child_id)
            queue.append(child_id)
    return revoked


def dirty_event_ids_for_revoked_objects(registry: RunRegistry, revoked_object_ids: list[str]) -> list[str]:
    dirty: set[str] = set()
    for object_id in revoked_object_ids:
        produced_by = registry.get_object(object_id).meta.get("produced_by_event_id")
        if produced_by:
            dirty.add(str(produced_by))
    return sorted(dirty, key=lambda event_id: _event_sort_key(registry.get_event(event_id)))


def build_replay_plan(
    registry: RunRegistry,
    base_history_run_id: str,
    method_run_id: str,
    suspicious_root_ids: list[str],
    revoked_object_ids: list[str],
) -> ReplayPlan:
    dirty_event_ids = dirty_event_ids_for_revoked_objects(registry, revoked_object_ids)
    replay_safe_event_ids: list[str] = []
    replay_unsafe_event_ids: list[str] = []
    revoked_set = set(revoked_object_ids)
    suspicious_set = set(suspicious_root_ids)
    for event_id in dirty_event_ids:
        event = registry.get_event(event_id)
        if _is_replay_safe(registry, event, suspicious_set, revoked_set):
            replay_safe_event_ids.append(event_id)
        else:
            replay_unsafe_event_ids.append(event_id)
    if replay_unsafe_event_ids:
        return ReplayPlan(
            plan_id=f"rp_{method_run_id}_01",
            base_history_run_id=base_history_run_id,
            method_run_id=method_run_id,
            suspicious_root_ids=list(suspicious_root_ids),
            revoked_object_ids=list(revoked_object_ids),
            dirty_event_ids=dirty_event_ids,
            replay_safe_event_ids=replay_safe_event_ids,
            replay_unsafe_event_ids=replay_unsafe_event_ids,
            replacement_map={},
            replay_order=dirty_event_ids,
            mode="coarse_rollback",
            status="fallback",
            fallback_reason="replay_unsafe_dirty_writer",
        )
    return ReplayPlan(
        plan_id=f"rp_{method_run_id}_01",
        base_history_run_id=base_history_run_id,
        method_run_id=method_run_id,
        suspicious_root_ids=list(suspicious_root_ids),
        revoked_object_ids=list(revoked_object_ids),
        dirty_event_ids=dirty_event_ids,
        replay_safe_event_ids=replay_safe_event_ids,
        replay_unsafe_event_ids=replay_unsafe_event_ids,
        replacement_map={},
        replay_order=dirty_event_ids,
        mode="writer_only",
        status="pending",
    )


def apply_revocations_without_replay(registry: RunRegistry, revoked_object_ids: list[str]) -> None:
    for object_id in revoked_object_ids:
        registry.update_object_status(object_id, "revoked")
    registry.set_active_state(
        active_memory_ids=[obj_id for obj_id in registry.active_memory_ids if obj_id not in revoked_object_ids],
        active_summary_id=None if registry.active_summary_id in revoked_object_ids else registry.active_summary_id,
        quarantined_root_ids=list(registry.quarantined_root_ids),
    )


def replay_dirty_writers(
    registry: RunRegistry,
    backend,
    chain: ChainDefinition,
    architecture: str,
    run_id: str,
    replay_plan: ReplayPlan,
    prompt_versions: dict[str, str],
) -> ReplayPlan:
    replacement_map: dict[str, str] = {}
    for event_id in replay_plan.replay_order:
        original_event = registry.get_event(event_id)
        cleaned_objects = _cleaned_input_objects(registry, original_event, replay_plan.suspicious_root_ids, replay_plan.revoked_object_ids, replacement_map)
        user_turn = next(obj for obj in cleaned_objects if obj.kind == "user_turn")
        tool_results = [obj for obj in cleaned_objects if obj.kind == "tool_result"]
        if original_event.type == "memory_write":
            writer_payload = backend.write_memory(chain, original_event.session_id, user_turn.content, tool_results, None)
            _append_replayed_memory_write(
                registry=registry,
                run_id=run_id,
                original_event=original_event,
                cleaned_objects=cleaned_objects,
                writer_payload=writer_payload,
                writer_version=prompt_versions["memory_writer"],
                replacement_map=replacement_map,
            )
        elif original_event.type == "summary_write":
            previous_summary = next((obj for obj in cleaned_objects if obj.kind == "summary"), None)
            writer_payload = backend.write_summary(chain, original_event.session_id, previous_summary, user_turn.content, tool_results, None)
            _append_replayed_summary_write(
                registry=registry,
                run_id=run_id,
                original_event=original_event,
                cleaned_objects=cleaned_objects,
                writer_payload=writer_payload,
                writer_version=prompt_versions["summary_writer"],
                replacement_map=replacement_map,
            )
    replay_plan.replacement_map = dict(replacement_map)
    replay_plan.status = "done"
    for object_id in replay_plan.revoked_object_ids:
        if object_id in replacement_map:
            registry.update_object_status(object_id, "replaced")
        else:
            registry.update_object_status(object_id, "revoked")
    _rebuild_active_state(registry)
    return replay_plan


def coarse_rollback_and_replay_suffix(
    registry: RunRegistry,
    env,
    backend,
    chain: ChainDefinition,
    architecture: str,
    run_id: str,
    suspicious_root_ids: list[str],
    revoked_object_ids: list[str],
    max_action_steps: int,
    memory_retrieval_k: int,
    prompt_versions: dict[str, str],
) -> None:
    earliest_session = min((_session_index(registry.get_object(obj_id).session_id) for obj_id in revoked_object_ids), default=0)
    start_session = SESSION_ORDER[earliest_session]
    checkpoint_id = _checkpoint_before_session(start_session)
    checkpoint = registry.get_checkpoint(checkpoint_id)
    registry.set_active_state(
        active_memory_ids=list(checkpoint.active_memory_ids),
        active_summary_id=checkpoint.active_summary_id,
        quarantined_root_ids=list(suspicious_root_ids),
    )
    run_sessions(
        registry=registry,
        env=env,
        backend=backend,
        chain=chain,
        architecture=architecture,
        run_id=run_id,
        session_ids=SESSION_ORDER[earliest_session:],
        max_action_steps=max_action_steps,
        memory_retrieval_k=memory_retrieval_k,
        prompt_versions=prompt_versions,
        quarantined_paths=suspicious_root_paths(registry, suspicious_root_ids),
    )


def _is_replay_safe(registry: RunRegistry, event: EventRecord, suspicious_roots: set[str], persisted_descendants: set[str]) -> bool:
    cleaned_objects = _cleaned_input_objects(
        registry=registry,
        original_event=event,
        suspicious_root_ids=list(suspicious_roots),
        revoked_object_ids=list(persisted_descendants),
        replacement_map={},
    )
    if not any(obj.kind == "user_turn" for obj in cleaned_objects):
        return False
    if event.type == "summary_write":
        original_had_summary_input = any(
            object_id in registry.objects and registry.get_object(object_id).kind == "summary"
            for object_id in event.input_object_ids
        )
        cleaned_has_summary_input = any(obj.kind == "summary" for obj in cleaned_objects)
        if original_had_summary_input and not cleaned_has_summary_input:
            return False
    return True


def _cleaned_input_objects(
    registry: RunRegistry,
    original_event: EventRecord,
    suspicious_root_ids: list[str],
    revoked_object_ids: list[str],
    replacement_map: dict[str, str],
) -> list[ObjectRecord]:
    suspicious_set = set(suspicious_root_ids)
    suspicious_paths = suspicious_root_paths(registry, suspicious_root_ids)
    revoked_set = set(revoked_object_ids)
    cleaned_ids: list[str] = []
    for object_id in original_event.input_object_ids:
        if object_id in suspicious_set:
            continue
        if object_id in replacement_map:
            cleaned_ids.append(replacement_map[object_id])
            continue
        if object_id in revoked_set:
            continue
        if object_id in registry.objects:
            obj = registry.get_object(object_id)
            if _object_references_suspicious_path(obj, suspicious_paths):
                continue
        cleaned_ids.append(object_id)
    return [registry.get_object(object_id) for object_id in cleaned_ids if object_id in registry.objects]


def _object_references_suspicious_path(obj: ObjectRecord, suspicious_paths: set[str]) -> bool:
    if not suspicious_paths or obj.kind != "tool_result":
        return False
    path = obj.meta.get("path")
    if isinstance(path, str) and path in suspicious_paths:
        return True
    content = str(obj.content)
    return any(suspicious_path in content for suspicious_path in suspicious_paths)


def _append_replayed_memory_write(
    registry: RunRegistry,
    run_id: str,
    original_event: EventRecord,
    cleaned_objects: list[ObjectRecord],
    writer_payload: dict[str, object],
    writer_version: str,
    replacement_map: dict[str, str],
) -> None:
    event_id = registry.id_allocator.next_event_id(original_event.session_id, "memory_write")
    original_outputs = [registry.get_object(object_id) for object_id in original_event.output_object_ids]
    new_output_ids: list[str] = []
    new_objects: list[ObjectRecord] = []
    input_ids = [obj.object_id for obj in cleaned_objects]
    items = list(writer_payload.get("items", []))
    for item in items:
        replacement_target = _match_replay_target(original_outputs, item)
        object_id = registry.id_allocator.next_object_id(original_event.session_id, "memory")
        new_output_ids.append(object_id)
        obj = ObjectRecord(
            object_id=object_id,
            run_id=run_id,
            session_id=original_event.session_id,
            kind="memory",
            subkind="episodic",
            content=str(item["text"]),
            source="memory_writer",
            parent_ids=input_ids,
            status="active",
            meta={
                "writer_version": writer_version,
                "memory_kind": item.get("kind", "topic_fact"),
                "salience": item.get("salience", 1),
                "replay_of": replacement_target.object_id if replacement_target else None,
                "produced_by_event_id": event_id,
            },
        )
        new_objects.append(obj)
        if replacement_target:
            replacement_map[replacement_target.object_id] = object_id
    event = EventRecord(
        event_id=event_id,
        run_id=run_id,
        session_id=original_event.session_id,
        type="memory_write",
        input_object_ids=input_ids,
        output_object_ids=new_output_ids,
        meta={"writer_version": writer_version, "replay_of": original_event.event_id, "counts_as_llm_call": True},
    )
    registry.append_event(event)
    for obj in new_objects:
        registry.append_object(obj)


def _append_replayed_summary_write(
    registry: RunRegistry,
    run_id: str,
    original_event: EventRecord,
    cleaned_objects: list[ObjectRecord],
    writer_payload: dict[str, object],
    writer_version: str,
    replacement_map: dict[str, str],
) -> None:
    event_id = registry.id_allocator.next_event_id(original_event.session_id, "summary_write")
    original_output = registry.get_object(original_event.output_object_ids[0])
    object_id = registry.id_allocator.next_object_id(original_event.session_id, "summary")
    input_ids = [obj.object_id for obj in cleaned_objects]
    carry_forward = list(writer_payload.get("carry_forward", []))
    content = str(writer_payload["summary"])
    if carry_forward:
        content += "\n" + "\n".join(carry_forward)
    obj = ObjectRecord(
        object_id=object_id,
        run_id=run_id,
        session_id=original_event.session_id,
        kind="summary",
        subkind="rolling",
        content=content,
        source="summary_writer",
        parent_ids=input_ids,
        status="active",
        meta={
            "writer_version": writer_version,
            "carry_forward": carry_forward,
            "replay_of": original_output.object_id,
            "produced_by_event_id": event_id,
        },
    )
    event = EventRecord(
        event_id=event_id,
        run_id=run_id,
        session_id=original_event.session_id,
        type="summary_write",
        input_object_ids=input_ids,
        output_object_ids=[object_id],
        meta={"writer_version": writer_version, "replay_of": original_event.event_id, "counts_as_llm_call": True},
    )
    registry.append_event(event)
    registry.append_object(obj)
    replacement_map[original_output.object_id] = object_id


def _match_replay_target(original_outputs: list[ObjectRecord], item: dict[str, object]) -> ObjectRecord | None:
    kind = str(item.get("kind", ""))
    text = str(item.get("text", ""))
    for candidate in original_outputs:
        if candidate.meta.get("memory_kind") == kind:
            return candidate
    for candidate in original_outputs:
        if candidate.content == text:
            return candidate
    return None


def _rebuild_active_state(registry: RunRegistry) -> None:
    active_memory_ids = [
        obj.object_id
        for obj in registry.objects.values()
        if obj.kind == "memory" and obj.status == "active"
    ]
    active_summaries = [
        obj for obj in registry.objects.values() if obj.kind == "summary" and obj.status == "active"
    ]
    active_summaries.sort(key=lambda obj: (_session_index(obj.session_id), obj.object_id))
    registry.set_active_state(active_memory_ids=sorted(active_memory_ids), active_summary_id=active_summaries[-1].object_id if active_summaries else None)


def _event_sort_key(event: EventRecord) -> tuple[int, str]:
    return (_session_index(event.session_id), event.event_id)


def _session_index(session_id: str) -> int:
    return SESSION_ORDER.index(session_id)


def _checkpoint_before_session(session_id: str) -> str:
    if session_id == "s1":
        return "ckpt_before_s1"
    previous = SESSION_ORDER[_session_index(session_id) - 1]
    return f"ckpt_after_{previous}"
