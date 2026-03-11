from __future__ import annotations

import sys

import pytest

from src.llm_backend import (
    ActionContext,
    LocalQwenBackend,
    _extract_first_json_object,
    _normalize_action_payload,
    _normalize_memory_payload,
    _normalize_summary_payload,
)
from src.pipeline import load_project_settings
from src.qwen_runtime import _bundle_cache_key, bootstrap_local_qwen_stack, repo_root
from src.schemas import ObjectRecord


def test_extract_first_json_object_ignores_prefix_and_suffix() -> None:
    payload = _extract_first_json_object('Thought: ignored\n{"action":"answer","args":{"text":"hi"}}\nDone')
    assert payload["action"] == "answer"
    assert payload["args"]["text"] == "hi"


def test_normalize_action_payload_fixes_k_and_null_fields() -> None:
    payload = _normalize_action_payload({"action": "search_docs", "args": {"query": "travel rules", "k": 999}}, default_query="fallback")
    assert payload == {"action": "search_docs", "args": {"query": "travel rules", "k": 3, "path": None, "text": None}}


def test_normalize_action_payload_rejects_unreadable_read_path() -> None:
    try:
        _normalize_action_payload(
            {"action": "read_doc", "args": {"path": "/workspace/travel_reimbursement_docs"}},
            default_query="fallback",
            allowed_read_paths={"public/travel_policy.md"},
        )
    except RuntimeError as exc:
        assert "unreadable path" in str(exc)
    else:
        raise AssertionError("Expected invalid read_doc path to raise RuntimeError.")


def test_normalize_memory_payload_limits_items_and_sanitizes_kind() -> None:
    payload = _normalize_memory_payload(
        {
            "items": [
                {"text": "Keep this", "kind": "workflow_note", "salience": 9},
                {"text": "Also keep this", "kind": "unknown_kind"},
                {"text": "Drop this third item", "kind": "user_fact"},
            ]
        }
    )
    assert payload == {
        "items": [
            {"text": "Keep this", "kind": "workflow_note", "salience": 1},
            {"text": "Also keep this", "kind": "topic_fact", "salience": 1},
        ]
    }


def test_normalize_summary_payload_builds_fallback_summary() -> None:
    payload = _normalize_summary_payload({"carry_forward": ["alpha", "beta", "", "gamma", "delta"]})
    assert payload == {
        "summary": "Carry forward: alpha | beta | gamma | delta",
        "carry_forward": ["alpha", "beta", "gamma", "delta"],
    }


def test_local_qwen_choose_action_raises_on_invalid_semantics_after_repair(monkeypatch) -> None:
    settings = load_project_settings()
    chain = settings["catalog"].get_chain("c01_travel")
    backend = LocalQwenBackend({"local_model_path": "/models/Qwen2.5-7B-Instruct", "max_tokens": 32})
    calls = iter(
        [
            {"action": "read_doc", "args": {"path": None}},
            {"action": "read_doc", "args": {"path": None}},
        ]
    )

    monkeypatch.setattr(backend, "_generate_json", lambda *args, **kwargs: next(calls))

    with pytest.raises(RuntimeError, match="semantically invalid action JSON after one repair attempt"):
        backend.choose_action(
            ActionContext(
                chain=chain,
                session_id="s1",
                user_turn=chain.sessions.s1_prompt,
                visible_objects=[
                    ObjectRecord(
                        object_id="obj_s1_user_01",
                        run_id="test",
                        session_id="s1",
                        kind="user_turn",
                        subkind="user_turn",
                        content=chain.sessions.s1_prompt,
                        source="user",
                        parent_ids=[],
                        status="active",
                        meta={},
                    )
                ],
                current_session_tool_results=[],
                searchable_paths_this_session=set(),
                max_action_steps=4,
                current_step=1,
            )
        )


def test_local_qwen_generate_json_raises_on_repeated_malformed_outputs(monkeypatch) -> None:
    backend = LocalQwenBackend({"local_model_path": "/models/Qwen2.5-7B-Instruct", "max_tokens": 32})
    outputs = iter(["not json at all", "still not json"])
    monkeypatch.setattr(backend, "_generate_text", lambda *args, **kwargs: next(outputs))

    with pytest.raises(RuntimeError, match="did not return valid JSON after one repair attempt"):
        backend._generate_json(
            system_instruction="Return JSON only.",
            user_prompt="Emit one JSON object.",
            repair_instruction="Repair the output to valid JSON.",
            max_new_tokens=32,
        )


def test_bundle_cache_key_includes_chat_template_kwargs() -> None:
    first = _bundle_cache_key("/models/Qwen3.5-27B", "qwen3_5_text", {"enable_thinking": False})
    second = _bundle_cache_key("/models/Qwen3.5-27B", "qwen3_5_text", {"enable_thinking": True})
    assert first != second


def test_bootstrap_local_qwen_stack_prefers_installed_packages(monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", ["site-packages"])
    monkeypatch.setattr("src.qwen_runtime.find_spec", lambda name: object())

    bootstrap_local_qwen_stack()

    assert sys.path == ["site-packages"]


def test_bootstrap_local_qwen_stack_falls_back_to_vendored_paths(monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", ["site-packages"])
    monkeypatch.setattr("src.qwen_runtime.find_spec", lambda name: None)

    bootstrap_local_qwen_stack()

    vendor_root = repo_root() / "vendor"
    assert sys.path[:2] == [
        str(vendor_root / "hfdeps"),
        str(vendor_root / "transformers-main" / "src"),
    ]
