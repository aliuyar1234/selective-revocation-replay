from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.checkpoint import write_checkpoint
from src.env_workspace import WorkspaceEnv
from src.llm_backend import ActionContext, LLMBackend
from src.retrieval import retrieve_active_memory
from src.schemas import ChainDefinition, EventRecord, ObjectRecord
from src.store import RunRegistry
from src.writers import write_memory_objects, write_summary_object


SESSION_ORDER = ["s1", "s2", "s3", "s4"]


@dataclass
class SessionResult:
    session_id: str
    answer_text: str
    user_turn_object_id: str
    tool_result_object_ids: list[str]
    written_object_ids: list[str]


def prompt_for_session(chain: ChainDefinition, session_id: str) -> str:
    return getattr(chain.sessions, f"{session_id}_prompt")


def run_session(
    registry: RunRegistry,
    env: WorkspaceEnv,
    backend: LLMBackend,
    chain: ChainDefinition,
    architecture: str,
    run_id: str,
    session_id: str,
    user_turn: str,
    max_action_steps: int,
    memory_retrieval_k: int,
    prompt_versions: dict[str, str],
    quarantined_paths: set[str] | None = None,
) -> SessionResult:
    quarantined_paths = quarantined_paths or set()
    user_turn_obj = _append_user_turn_object(registry, run_id, session_id, user_turn)
    tool_results: list[ObjectRecord] = []
    searchable_paths_this_session: set[str] = set()
    previous_summary = registry.active_summary_object()
    answer_text = ""

    for step in range(1, max_action_steps + 1):
        visible_objects = _visible_objects(
            registry=registry,
            architecture=architecture,
            user_turn=user_turn,
            user_turn_obj=user_turn_obj,
            current_session_tool_results=tool_results,
            memory_retrieval_k=memory_retrieval_k,
            token_regex=env.token_regex,
        )
        action_payload = backend.choose_action(
            ActionContext(
                chain=chain,
                session_id=session_id,
                user_turn=user_turn,
                visible_objects=visible_objects,
                current_session_tool_results=tool_results,
                searchable_paths_this_session=searchable_paths_this_session,
                max_action_steps=max_action_steps,
                current_step=step,
            )
        )
        llm_event = EventRecord(
            event_id=registry.id_allocator.next_event_id(session_id, "llm_act"),
            run_id=run_id,
            session_id=session_id,
            type="llm_act",
            input_object_ids=[obj.object_id for obj in visible_objects],
            output_object_ids=[],
            meta={
                "prompt_version": prompt_versions["act"],
                "action": action_payload["action"],
                "action_args": action_payload["args"],
                "llm_call_id": registry.id_allocator.next_llm_call_id(),
                "counts_as_llm_call": True,
            },
        )
        registry.append_event(llm_event)

        if action_payload["action"] == "search_docs":
            query = str(action_payload["args"]["query"])
            entries = env.search_docs(query, k=int(action_payload["args"]["k"]), quarantined_paths=quarantined_paths)
            searchable_paths_this_session.update(entry.path for entry in entries)
            tool_result_obj = _append_tool_trace(
                registry=registry,
                run_id=run_id,
                session_id=session_id,
                source="search_docs",
                subkind="search_result_list",
                content=env.serialize_search_results(entries),
                visible_objects=visible_objects,
                llm_event=llm_event,
                tool_name="search_docs",
                tool_args={"query": query, "k": int(action_payload["args"]["k"])},
                meta_extra={},
            )
            tool_results.append(tool_result_obj)
            continue

        if action_payload["action"] == "read_doc":
            path = str(action_payload["args"]["path"])
            path, content = env.read_doc(
                path=path,
                searchable_paths_this_session=searchable_paths_this_session,
                prompt_visible_texts=[obj.content for obj in visible_objects],
                quarantined_paths=quarantined_paths,
            )
            tool_result_obj = _append_tool_trace(
                registry=registry,
                run_id=run_id,
                session_id=session_id,
                source="read_doc",
                subkind="file_content",
                content=content,
                visible_objects=visible_objects,
                llm_event=llm_event,
                tool_name="read_doc",
                tool_args={"path": path},
                meta_extra={"path": path, "restricted": path in chain.scoring.forbidden_reads},
            )
            tool_results.append(tool_result_obj)
            continue

        if action_payload["action"] == "answer":
            answer_text = str(action_payload["args"]["text"])
            break

    written_object_ids: list[str] = []
    if architecture == "retrieval":
        writer_payload = backend.write_memory(chain, session_id, user_turn, tool_results, answer_text)
        objects = write_memory_objects(
            registry=registry,
            run_id=run_id,
            session_id=session_id,
            user_turn_obj=user_turn_obj,
            tool_result_objs=tool_results,
            writer_payload=writer_payload,
            writer_version=prompt_versions["memory_writer"],
        )
        written_object_ids.extend(obj.object_id for obj in objects)
    elif architecture == "summary":
        writer_payload = backend.write_summary(chain, session_id, previous_summary, user_turn, tool_results, answer_text)
        summary_obj = write_summary_object(
            registry=registry,
            run_id=run_id,
            session_id=session_id,
            previous_summary=previous_summary,
            user_turn_obj=user_turn_obj,
            tool_result_objs=tool_results,
            writer_payload=writer_payload,
            writer_version=prompt_versions["summary_writer"],
        )
        written_object_ids.append(summary_obj.object_id)
    else:
        raise ValueError(f"Unsupported architecture {architecture}")

    write_checkpoint(registry, run_id=run_id, after_session=session_id)
    return SessionResult(
        session_id=session_id,
        answer_text=answer_text,
        user_turn_object_id=user_turn_obj.object_id,
        tool_result_object_ids=[obj.object_id for obj in tool_results],
        written_object_ids=written_object_ids,
    )


def run_sessions(
    registry: RunRegistry,
    env: WorkspaceEnv,
    backend: LLMBackend,
    chain: ChainDefinition,
    architecture: str,
    run_id: str,
    session_ids: list[str],
    max_action_steps: int,
    memory_retrieval_k: int,
    prompt_versions: dict[str, str],
    quarantined_paths: set[str] | None = None,
) -> list[SessionResult]:
    results: list[SessionResult] = []
    for session_id in session_ids:
        results.append(
            run_session(
                registry=registry,
                env=env,
                backend=backend,
                chain=chain,
                architecture=architecture,
                run_id=run_id,
                session_id=session_id,
                user_turn=prompt_for_session(chain, session_id),
                max_action_steps=max_action_steps,
                memory_retrieval_k=memory_retrieval_k,
                prompt_versions=prompt_versions,
                quarantined_paths=quarantined_paths,
            )
        )
    return results


def _append_user_turn_object(registry: RunRegistry, run_id: str, session_id: str, user_turn: str) -> ObjectRecord:
    object_id = registry.id_allocator.next_object_id(session_id, "user_turn")
    obj = ObjectRecord(
        object_id=object_id,
        run_id=run_id,
        session_id=session_id,
        kind="user_turn",
        subkind="user_turn",
        content=user_turn,
        source="user",
        parent_ids=[],
        status="active",
        meta={"produced_by_event_id": None, "replay_of": None},
    )
    registry.append_object(obj)
    return obj


def _visible_objects(
    registry: RunRegistry,
    architecture: str,
    user_turn: str,
    user_turn_obj: ObjectRecord,
    current_session_tool_results: list[ObjectRecord],
    memory_retrieval_k: int,
    token_regex: str,
) -> list[ObjectRecord]:
    visible: list[ObjectRecord] = [user_turn_obj]
    if architecture == "retrieval":
        visible.extend(retrieve_active_memory(user_turn, registry.active_memory_objects(), memory_retrieval_k, token_regex))
    elif architecture == "summary":
        summary_obj = registry.active_summary_object()
        if summary_obj is not None:
            visible.append(summary_obj)
    visible.extend(current_session_tool_results)
    return visible


def _append_tool_trace(
    registry: RunRegistry,
    run_id: str,
    session_id: str,
    source: str,
    subkind: str,
    content: str,
    visible_objects: list[ObjectRecord],
    llm_event: EventRecord,
    tool_name: str,
    tool_args: dict[str, Any],
    meta_extra: dict[str, Any],
) -> ObjectRecord:
    tool_call_event = EventRecord(
        event_id=registry.id_allocator.next_event_id(session_id, "tool_call"),
        run_id=run_id,
        session_id=session_id,
        type="tool_call",
        input_object_ids=[obj.object_id for obj in visible_objects],
        output_object_ids=[],
        meta={"tool_name": tool_name, "tool_args": tool_args, "caused_by_event_id": llm_event.event_id, "counts_as_llm_call": False},
    )
    registry.append_event(tool_call_event)

    object_id = registry.id_allocator.next_object_id(session_id, "tool_result")
    tool_result_event = EventRecord(
        event_id=registry.id_allocator.next_event_id(session_id, "tool_result"),
        run_id=run_id,
        session_id=session_id,
        type="tool_result",
        input_object_ids=[obj.object_id for obj in visible_objects],
        output_object_ids=[object_id],
        meta={
            "tool_name": tool_name,
            "tool_args": tool_args,
            "caused_by_event_id": tool_call_event.event_id,
            "decision_event_id": llm_event.event_id,
            "counts_as_llm_call": False,
        },
    )
    registry.append_event(tool_result_event)

    obj = ObjectRecord(
        object_id=object_id,
        run_id=run_id,
        session_id=session_id,
        kind="tool_result",
        subkind=subkind,
        content=content,
        source=source,
        parent_ids=[obj.object_id for obj in visible_objects],
        status="active",
        meta={
            "produced_by_event_id": tool_result_event.event_id,
            "decision_event_id": llm_event.event_id,
            "replay_of": None,
            **meta_extra,
        },
    )
    registry.append_object(obj)
    return obj
