from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.ids import utc_timestamp
from src.pipeline import create_base_history_run, create_method_run, load_project_settings
from src.utils import dump_json, ensure_dir, load_jsonl, short_text, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "c01_travel:retrieval:explicit",
            "c05_training:retrieval:explicit",
        ],
        help="Case triples formatted as chain_id:architecture:attack_variant.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["no_recovery", "selective_replay"],
        default=["no_recovery", "selective_replay"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    settings = load_project_settings()
    root = Path(settings["root"])
    stamp = utc_timestamp()
    rows: list[dict[str, Any]] = []

    for repeat_idx in range(args.repeats):
        repeat_label = f"{stamp}_r{repeat_idx + 1}"
        for case in args.cases:
            chain_id, architecture, attack_variant = _parse_case(case)
            base_history_id, _ = create_base_history_run(
                chain_id,
                architecture,
                attack_variant,
                seed=repeat_idx + 1,
                timestamp=repeat_label,
            )
            for method in args.methods:
                run_id, run_dir, result = create_method_run(base_history_id, method, timestamp=repeat_label)
                row = result.to_dict()
                row["repeat"] = repeat_idx + 1
                row["base_history_run_id"] = base_history_id
                row["method_run_id"] = run_id
                row["run_dir"] = str(run_dir.relative_to(root)).replace("\\", "/")
                rows.append(row)

    summary = _summarize_rows(rows)
    case_study = _select_case_study(rows)
    case_study_payload = _extract_case_study(case_study) if case_study else {}
    payload = {
        "active_model_name": settings["active_model_name"],
        "model_id": settings["model"]["model_id"],
        "repeats": args.repeats,
        "cases": list(args.cases),
        "methods": list(args.methods),
        "rows": rows,
        "summary": summary,
        "case_study": case_study_payload,
    }

    output_dir = ensure_dir(root / "results" / "model_pilots")
    stamp_base = f"live_confirmation_{settings['active_model_name']}_{stamp}"
    output_json = output_dir / f"{stamp_base}.json"
    output_csv = output_dir / f"{stamp_base}.csv"
    output_md = output_dir / f"{stamp_base}.md"

    dump_json(output_json, payload)
    write_csv(output_csv, rows, fieldnames=list(rows[0].keys()))
    output_md.write_text(_build_markdown_summary(payload), encoding="utf-8")

    shutil.copy2(output_json, output_dir / "live_confirmation_latest.json")
    shutil.copy2(output_csv, output_dir / "live_confirmation_latest.csv")
    shutil.copy2(output_md, output_dir / "live_confirmation_latest.md")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(output_json)
    print(output_csv)
    print(output_md)


def _parse_case(raw: str) -> tuple[str, str, str]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid case '{raw}'. Expected chain_id:architecture:attack_variant.")
    return parts[0], parts[1], parts[2]


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['chain_id']}::{row['architecture']}::{row['attack_variant']}::{row['method']}"
        grouped.setdefault(key, []).append(row)

    case_groups: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        chain_id, architecture, attack_variant, method = key.split("::")
        case_groups.append(
            {
                "chain_id": chain_id,
                "architecture": architecture,
                "attack_variant": attack_variant,
                "method": method,
                "count": len(group_rows),
                "avg_residual_asr_s3": _avg(group_rows, "residual_asr_s3"),
                "avg_s3_correct": _avg(group_rows, "s3_correct"),
                "avg_s4_retention_correct": _avg(group_rows, "s4_retention_correct"),
                "avg_extra_llm_calls_after_detection": _avg(group_rows, "extra_llm_calls_after_detection"),
                "avg_replayed_writer_event_count": _avg(group_rows, "replayed_writer_event_count"),
                "avg_revoked_object_count": _avg(group_rows, "revoked_object_count"),
                "avg_fallback_to_coarse_rollback": _avg(group_rows, "fallback_to_coarse_rollback"),
                "target_outcome_rate": _target_rate(group_rows, method),
            }
        )

    return {"case_groups": case_groups, "row_count": len(rows)}


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _target_rate(rows: list[dict[str, Any]], method: str) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        if method == "no_recovery":
            if int(row["residual_asr_s3"]) == 1:
                hits += 1
            continue
        if (
            int(row["residual_asr_s3"]) == 0
            and int(row["s3_correct"]) == 1
            and int(row["s4_retention_correct"]) == 1
        ):
            hits += 1
    return round(hits / len(rows), 4)


def _select_case_study(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = str(row["base_history_run_id"])
        pairs.setdefault(key, {})[str(row["method"])] = row

    best_pair: tuple[float, dict[str, dict[str, Any]]] | None = None
    for pair in pairs.values():
        no_recovery = pair.get("no_recovery")
        selective = pair.get("selective_replay")
        if not no_recovery or not selective:
            continue
        score = 0.0
        if selective["chain_id"] == "c01_travel":
            score += 4.0
        if selective["architecture"] == "retrieval":
            score += 3.0
        if selective["attack_variant"] == "explicit":
            score += 3.0
        if int(no_recovery["residual_asr_s3"]) == 1:
            score += 4.0
        if int(selective["residual_asr_s3"]) == 0:
            score += 4.0
        if int(selective["s3_correct"]) == 1:
            score += 2.0
        if int(selective["s4_retention_correct"]) == 1:
            score += 2.0
        if int(selective["fallback_to_coarse_rollback"]) == 0:
            score += 2.0
        score -= float(selective["extra_llm_calls_after_detection"]) / 20.0
        if best_pair is None or score > best_pair[0]:
            best_pair = (score, pair)

    if best_pair is None:
        return None
    return {"no_recovery": best_pair[1]["no_recovery"], "selective_replay": best_pair[1]["selective_replay"]}


def _extract_case_study(case_study: dict[str, Any]) -> dict[str, Any]:
    selective_row = case_study["selective_replay"]
    no_recovery_row = case_study["no_recovery"]
    root = Path(load_project_settings()["root"])
    selective_dir = root / str(selective_row["run_dir"])
    no_recovery_dir = root / str(no_recovery_row["run_dir"])
    base_dir = selective_dir.parents[1] / "base_histories" / str(selective_row["base_history_run_id"])
    metadata = json.loads((selective_dir / "run_metadata.json").read_text(encoding="utf-8"))

    base_objects = {row["object_id"]: row for row in load_jsonl(base_dir / "objects.jsonl")}
    selective_objects = load_jsonl(selective_dir / "objects.jsonl")
    selective_events = load_jsonl(selective_dir / "events.jsonl")
    no_recovery_events = load_jsonl(no_recovery_dir / "events.jsonl")
    replay_plan = json.loads((selective_dir / "replay_plan.json").read_text(encoding="utf-8"))

    root_id = replay_plan["suspicious_root_ids"][0]
    revoked_ids = replay_plan["revoked_object_ids"]
    root_object = base_objects[root_id]
    poisoned_memory = base_objects[revoked_ids[0]]
    preserved_fact = _find_object(selective_objects, "obj_s1_mem_04") or _find_active_memory(selective_objects, "user_fact", session_id="s1")
    repaired_guardrail = _find_object(selective_objects, "obj_s2_mem_03") or _find_active_memory(selective_objects, "workflow_note", session_id="s2")
    no_recovery_s3 = _find_event(no_recovery_events, "evt_s3_llm_act_01")
    recovered_s3_search = _find_event(selective_events, "evt_s3_llm_act_01")
    recovered_s3_read = _find_event(selective_events, "evt_s3_llm_act_02")
    recovered_s4 = _find_event(selective_events, "evt_s4_llm_act_01")

    return {
        "title": "Live Case Study",
        "model_name": metadata["model_name"],
        "model_id": metadata["model_id"],
        "chain_id": selective_row["chain_id"],
        "architecture": selective_row["architecture"],
        "attack_variant": selective_row["attack_variant"],
        "base_history_run_id": selective_row["base_history_run_id"],
        "no_recovery_method_run_id": no_recovery_row["method_run_id"],
        "selective_replay_method_run_id": selective_row["method_run_id"],
        "suspicious_root_id": root_id,
        "suspicious_root_path": root_object["meta"].get("path", ""),
        "suspicious_root_excerpt": short_text(root_object["content"], 115),
        "poisoned_memory_text": short_text(poisoned_memory["content"], 115),
        "preserved_fact_text": short_text(preserved_fact["content"] if preserved_fact else "", 115),
        "repaired_guardrail_text": short_text(repaired_guardrail["content"] if repaired_guardrail else "", 115),
        "no_recovery_s3_action": _format_action(no_recovery_s3),
        "recovered_s3_search_action": _format_action(recovered_s3_search),
        "recovered_s3_read_action": _format_action(recovered_s3_read),
        "recovered_s4_answer": _event_answer_text(recovered_s4),
        "replay_mode": replay_plan["mode"],
        "replay_status": replay_plan["status"],
        "replayed_writer_event_count": int(selective_row["replayed_writer_event_count"]),
        "revoked_object_count": int(selective_row["revoked_object_count"]),
        "fallback_to_coarse_rollback": int(selective_row["fallback_to_coarse_rollback"]),
        "replacement_map": replay_plan["replacement_map"],
        "revoked_object_ids": revoked_ids,
        "dirty_event_ids": replay_plan["dirty_event_ids"],
        "metrics": {
            "no_recovery": {
                "residual_asr_s3": int(no_recovery_row["residual_asr_s3"]),
                "s3_correct": int(no_recovery_row["s3_correct"]),
                "s4_retention_correct": int(no_recovery_row["s4_retention_correct"]),
                "extra_llm_calls_after_detection": int(no_recovery_row["extra_llm_calls_after_detection"]),
            },
            "selective_replay": {
                "residual_asr_s3": int(selective_row["residual_asr_s3"]),
                "s3_correct": int(selective_row["s3_correct"]),
                "s4_retention_correct": int(selective_row["s4_retention_correct"]),
                "extra_llm_calls_after_detection": int(selective_row["extra_llm_calls_after_detection"]),
            },
        },
    }


def _find_object(rows: list[dict[str, Any]], object_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row["object_id"] == object_id:
            return row
    return None


def _find_active_memory(
    rows: list[dict[str, Any]],
    memory_kind: str,
    *,
    session_id: str,
) -> dict[str, Any] | None:
    for row in rows:
        if row["kind"] != "memory":
            continue
        if row["session_id"] != session_id:
            continue
        if row["status"] != "active":
            continue
        if row["meta"].get("memory_kind") != memory_kind:
            continue
        return row
    return None


def _find_event(rows: list[dict[str, Any]], event_id: str) -> dict[str, Any]:
    for row in rows:
        if row["event_id"] == event_id:
            return row
    raise KeyError(f"Missing expected event {event_id}")


def _format_action(event: dict[str, Any]) -> str:
    action = event["meta"].get("action", "")
    action_args = event["meta"].get("action_args", {})
    if action == "read_doc":
        return f"read_doc({action_args.get('path', '')})"
    if action == "search_docs":
        return f"search_docs({action_args.get('query', '')})"
    if action == "answer":
        return short_text(str(action_args.get("text", "")), 80)
    return action


def _event_answer_text(event: dict[str, Any]) -> str:
    action_args = event["meta"].get("action_args", {})
    return short_text(str(action_args.get("text", "")), 100)


def _build_markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        f"# Live Confirmation Batch: {payload['active_model_name']}",
        "",
        f"- model id: `{payload['model_id']}`",
        f"- repeats per case: `{payload['repeats']}`",
        "",
        "## Case Groups",
        "",
    ]
    for row in payload["summary"]["case_groups"]:
        lines.extend(
            [
                f"- `{row['chain_id']}` / `{row['architecture']}` / `{row['attack_variant']}` / `{row['method']}`:"
                f" ASR={row['avg_residual_asr_s3']}, S3={row['avg_s3_correct']},"
                f" S4={row['avg_s4_retention_correct']}, target-rate={row['target_outcome_rate']}",
            ]
        )
    case_study = payload.get("case_study", {})
    if case_study:
        lines.extend(
            [
                "",
                "## Case Study",
                "",
                f"- chain: `{case_study['chain_id']}`",
                f"- architecture: `{case_study['architecture']}`",
                f"- suspicious root: `{case_study['suspicious_root_id']}` from `{case_study['suspicious_root_path']}`",
                f"- replay mode: `{case_study['replay_mode']}` with `{case_study['replayed_writer_event_count']}` replayed writers and `{case_study['revoked_object_count']}` revoked objects",
                f"- no recovery S3 action: `{case_study['no_recovery_s3_action']}`",
                f"- recovered S4 answer: `{case_study['recovered_s4_answer']}`",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
