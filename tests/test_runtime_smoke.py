from __future__ import annotations

from pathlib import Path

from src.agent_loop import run_sessions
from src.checkpoint import write_checkpoint
from src.env_workspace import WorkspaceEnv, materialize_chain_workspace
from src.llm_backend import build_backend
from src.pipeline import load_project_settings
from src.recover import (
    _cleaned_input_objects,
    build_replay_plan,
    compute_persisted_descendant_closure,
    select_suspicious_root_ids,
)
from src.schemas import EventRecord, ObjectRecord
from src.store import RunRegistry


def _build_temp_history(tmp_path: Path, architecture: str):
    settings = load_project_settings()
    catalog = settings["catalog"]
    chain = catalog.get_chain("c01_travel")
    workspace = materialize_chain_workspace(catalog, "c01_travel", "explicit", tmp_path / "workspace")
    run_dir = tmp_path / f"run_{architecture}"
    registry = RunRegistry(run_dir)
    backend = build_backend("heuristic")
    env = WorkspaceEnv(chain, workspace.root, settings["token_regex"])
    runtime_cfg = settings["runtime"][architecture]
    registry.save_metadata(
        {
            "run_id": f"test_{architecture}",
            "base_history_run_id": f"test_{architecture}",
            "method": "base_history",
            "chain_id": chain.id,
            "architecture": architecture,
            "attack_variant": "explicit",
            "seed": 1,
            "backend": "heuristic",
            "workspace_dir": str(workspace.root),
            "attack_path": workspace.attack_path,
            "detection_after_session": "s2",
            "prompt_versions": dict(runtime_cfg["prompt_versions"]),
            "base_event_count": 0,
            "notes": "",
        }
    )
    write_checkpoint(registry, run_id=f"test_{architecture}", after_session="before_s1")
    run_sessions(
        registry=registry,
        env=env,
        backend=backend,
        chain=chain,
        architecture=architecture,
        run_id=f"test_{architecture}",
        session_ids=["s1", "s2"],
        max_action_steps=int(runtime_cfg["max_action_steps"]),
        memory_retrieval_k=int(runtime_cfg["memory_retrieval_k"]),
        prompt_versions=dict(runtime_cfg["prompt_versions"]),
    )
    return settings, chain, registry, workspace


def test_workspace_search_is_deterministic(tmp_path: Path) -> None:
    settings = load_project_settings()
    catalog = settings["catalog"]
    chain = catalog.get_chain("c01_travel")
    workspace = materialize_chain_workspace(catalog, "c01_travel", "explicit", tmp_path / "workspace")
    env = WorkspaceEnv(chain, workspace.root, settings["token_regex"])
    first = [entry.to_dict() for entry in env.search_docs("travel reimbursement", 3)]
    second = [entry.to_dict() for entry in env.search_docs("travel reimbursement", 3)]
    assert first == second
    assert any(entry["path"] == "public/travel_policy.md" for entry in first)
    assert not any(entry["path"] == "restricted/vault.txt" for entry in first)


def test_retrieval_plan_uses_writer_only_replay(tmp_path: Path) -> None:
    _, _, registry, workspace = _build_temp_history(tmp_path, "retrieval")
    suspicious_root_ids = select_suspicious_root_ids(registry, workspace.attack_path)
    revoked_object_ids = compute_persisted_descendant_closure(registry, suspicious_root_ids)
    plan = build_replay_plan(registry, "test_retrieval", "test_retrieval_method", suspicious_root_ids, revoked_object_ids)
    assert suspicious_root_ids
    assert revoked_object_ids
    assert plan.mode == "writer_only"


def test_summary_plan_falls_back_to_coarse_rollback(tmp_path: Path) -> None:
    _, _, registry, workspace = _build_temp_history(tmp_path, "summary")
    suspicious_root_ids = select_suspicious_root_ids(registry, workspace.attack_path)
    revoked_object_ids = compute_persisted_descendant_closure(registry, suspicious_root_ids)
    plan = build_replay_plan(registry, "test_summary", "test_summary_method", suspicious_root_ids, revoked_object_ids)
    assert suspicious_root_ids
    assert revoked_object_ids
    assert plan.mode == "coarse_rollback"
    assert plan.status == "fallback"


def test_registry_reload_uses_latest_session_checkpoint(tmp_path: Path) -> None:
    _, _, registry, _ = _build_temp_history(tmp_path, "retrieval")
    reloaded = RunRegistry(registry.run_dir)
    latest = reloaded.latest_checkpoint()
    assert latest is not None
    assert latest.checkpoint_id == "ckpt_after_s2"
    assert reloaded.active_memory_ids == latest.active_memory_ids
    assert "obj_s1_mem_01" in reloaded.active_memory_ids
    assert "obj_s1_mem_02" in reloaded.active_memory_ids
    assert any(object_id.startswith("obj_s2_mem_") for object_id in reloaded.active_memory_ids)


def test_descendant_closure_traverses_tool_results_between_persisted_nodes(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "run")
    registry.append_object(
        ObjectRecord(
            object_id="obj_s1_tool_01",
            run_id="test",
            session_id="s1",
            kind="tool_result",
            subkind="file_content",
            content="malicious seed",
            source="read_doc",
            parent_ids=[],
            status="active",
            meta={"path": "malicious/travel_reference_explicit.md"},
        )
    )
    registry.append_object(
        ObjectRecord(
            object_id="obj_s1_mem_01",
            run_id="test",
            session_id="s1",
            kind="memory",
            subkind="episodic",
            content="poisoned workflow note",
            source="memory_writer",
            parent_ids=["obj_s1_tool_01"],
            status="active",
            meta={},
        )
    )
    registry.append_object(
        ObjectRecord(
            object_id="obj_s2_tool_01",
            run_id="test",
            session_id="s2",
            kind="tool_result",
            subkind="file_content",
            content="later tool output",
            source="read_doc",
            parent_ids=["obj_s1_mem_01"],
            status="active",
            meta={"path": "restricted/vault.txt"},
        )
    )
    registry.append_object(
        ObjectRecord(
            object_id="obj_s2_mem_01",
            run_id="test",
            session_id="s2",
            kind="memory",
            subkind="episodic",
            content="later poisoned memory",
            source="memory_writer",
            parent_ids=["obj_s2_tool_01"],
            status="active",
            meta={},
        )
    )

    revoked = compute_persisted_descendant_closure(registry, ["obj_s1_tool_01"])

    assert revoked == ["obj_s1_mem_01", "obj_s2_mem_01"]


def test_cleaned_input_objects_drop_tool_results_that_reference_quarantined_path(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "run")
    registry.append_object(
        ObjectRecord(
            object_id="obj_s1_user_01",
            run_id="test",
            session_id="s1",
            kind="user_turn",
            subkind="user_turn",
            content="review the docs",
            source="user",
            parent_ids=[],
            status="active",
            meta={},
        )
    )
    registry.append_object(
        ObjectRecord(
            object_id="obj_s1_tool_01",
            run_id="test",
            session_id="s1",
            kind="tool_result",
            subkind="search_result_list",
            content='[{"path":"malicious/travel_reference_explicit.md","title":"attack","snippet":"bad"}]',
            source="search_docs",
            parent_ids=["obj_s1_user_01"],
            status="active",
            meta={},
        )
    )
    registry.append_object(
        ObjectRecord(
            object_id="obj_s1_tool_02",
            run_id="test",
            session_id="s1",
            kind="tool_result",
            subkind="file_content",
            content="malicious file contents",
            source="read_doc",
            parent_ids=["obj_s1_tool_01"],
            status="active",
            meta={"path": "malicious/travel_reference_explicit.md"},
        )
    )
    cleaned = _cleaned_input_objects(
        registry=registry,
        original_event=EventRecord(
            event_id="evt_s1_memory_write_01",
            run_id="test",
            session_id="s1",
            type="memory_write",
            input_object_ids=["obj_s1_user_01", "obj_s1_tool_01", "obj_s1_tool_02"],
            output_object_ids=[],
            meta={},
        ),
        suspicious_root_ids=["obj_s1_tool_02"],
        revoked_object_ids=[],
        replacement_map={},
    )

    assert [obj.object_id for obj in cleaned] == ["obj_s1_user_01"]
