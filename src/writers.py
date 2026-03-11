from __future__ import annotations

from src.schemas import EventRecord, ObjectRecord
from src.store import RunRegistry


def write_memory_objects(
    registry: RunRegistry,
    run_id: str,
    session_id: str,
    user_turn_obj: ObjectRecord,
    tool_result_objs: list[ObjectRecord],
    writer_payload: dict[str, object],
    writer_version: str = "memory_writer_v1",
) -> list[ObjectRecord]:
    items = list(writer_payload.get("items", []))
    input_object_ids = [user_turn_obj.object_id] + [obj.object_id for obj in tool_result_objs]
    event_id = registry.id_allocator.next_event_id(session_id, "memory_write")
    output_ids: list[str] = []
    objects: list[ObjectRecord] = []
    for item in items:
        object_id = registry.id_allocator.next_object_id(session_id, "memory")
        output_ids.append(object_id)
        obj = ObjectRecord(
            object_id=object_id,
            run_id=run_id,
            session_id=session_id,
            kind="memory",
            subkind="episodic",
            content=str(item["text"]),
            source="memory_writer",
            parent_ids=list(input_object_ids),
            status="active",
            meta={
                "writer_version": writer_version,
                "memory_kind": item.get("kind", "topic_fact"),
                "salience": item.get("salience", 1),
                "replay_of": None,
                "produced_by_event_id": event_id,
            },
        )
        objects.append(obj)
    event = EventRecord(
        event_id=event_id,
        run_id=run_id,
        session_id=session_id,
        type="memory_write",
        input_object_ids=input_object_ids,
        output_object_ids=output_ids,
        meta={"writer_version": writer_version, "replay_of": None, "counts_as_llm_call": True},
    )
    registry.append_event(event)
    for obj in objects:
        registry.append_object(obj)
        registry.active_memory_ids.append(obj.object_id)
    return objects


def write_summary_object(
    registry: RunRegistry,
    run_id: str,
    session_id: str,
    previous_summary: ObjectRecord | None,
    user_turn_obj: ObjectRecord,
    tool_result_objs: list[ObjectRecord],
    writer_payload: dict[str, object],
    writer_version: str = "summary_writer_v1",
) -> ObjectRecord:
    input_object_ids: list[str] = []
    if previous_summary is not None:
        input_object_ids.append(previous_summary.object_id)
    input_object_ids.append(user_turn_obj.object_id)
    input_object_ids.extend(obj.object_id for obj in tool_result_objs)
    event_id = registry.id_allocator.next_event_id(session_id, "summary_write")
    object_id = registry.id_allocator.next_object_id(session_id, "summary")
    carry_forward = list(writer_payload.get("carry_forward", []))
    content = str(writer_payload["summary"])
    if carry_forward:
        content += "\n" + "\n".join(carry_forward)
    obj = ObjectRecord(
        object_id=object_id,
        run_id=run_id,
        session_id=session_id,
        kind="summary",
        subkind="rolling",
        content=content,
        source="summary_writer",
        parent_ids=input_object_ids,
        status="active",
        meta={
            "writer_version": writer_version,
            "carry_forward": carry_forward,
            "replay_of": None,
            "produced_by_event_id": event_id,
        },
    )
    event = EventRecord(
        event_id=event_id,
        run_id=run_id,
        session_id=session_id,
        type="summary_write",
        input_object_ids=input_object_ids,
        output_object_ids=[object_id],
        meta={"writer_version": writer_version, "replay_of": None, "counts_as_llm_call": True},
    )
    registry.append_event(event)
    registry.append_object(obj)
    registry.active_summary_id = obj.object_id
    return obj
