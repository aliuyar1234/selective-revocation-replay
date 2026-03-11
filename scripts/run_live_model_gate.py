from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.ids import utc_timestamp
from src.pipeline import create_base_history_run, create_method_run, load_project_settings
from src.utils import dump_json, ensure_dir, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", nargs="+", default=["c01_travel", "c02_procurement"])
    parser.add_argument("--architectures", nargs="+", choices=["retrieval", "summary"], default=["retrieval", "summary"])
    parser.add_argument("--attacks", nargs="+", choices=["explicit", "stealth"], default=["explicit"])
    parser.add_argument("--methods", nargs="+", choices=["no_recovery", "selective_replay"], default=["no_recovery", "selective_replay"])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    settings = load_project_settings()
    root = Path(settings["root"])
    stamp = f"{utc_timestamp()}_{os.getpid()}"
    rows: list[dict[str, object]] = []

    for chain_id in args.chains:
        for architecture in args.architectures:
            for attack_variant in args.attacks:
                base_history_id, _ = create_base_history_run(
                    chain_id,
                    architecture,
                    attack_variant,
                    seed=args.seed,
                    timestamp=stamp,
                )
                for method in args.methods:
                    run_id, run_dir, result = create_method_run(base_history_id, method, timestamp=stamp)
                    row = result.to_dict()
                    row["base_history_run_id"] = base_history_id
                    row["method_run_id"] = run_id
                    row["run_dir"] = str(run_dir.relative_to(root)).replace("\\", "/")
                    rows.append(row)

    output_dir = ensure_dir(root / "results" / "model_pilots")
    output_json = output_dir / f"gate_{settings['active_model_name']}_{stamp}.json"
    output_csv = output_dir / f"gate_{settings['active_model_name']}_{stamp}.csv"
    summary = _summarize_rows(rows)

    dump_json(
        output_json,
        {
            "active_model_name": settings["active_model_name"],
            "model_id": settings["model"]["model_id"],
            "seed": args.seed,
            "chains": list(args.chains),
            "architectures": list(args.architectures),
            "attacks": list(args.attacks),
            "methods": list(args.methods),
            "rows": rows,
            "summary": summary,
        },
    )
    write_csv(output_csv, rows, fieldnames=list(rows[0].keys()))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(output_json)
    print(output_csv)


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        key = f"{row['method']}::{row['architecture']}::{row['attack_variant']}"
        grouped.setdefault(key, []).append(row)

    by_group: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        method, architecture, attack_variant = key.split("::")
        count = len(group_rows)
        by_group.append(
            {
                "method": method,
                "architecture": architecture,
                "attack_variant": attack_variant,
                "count": count,
                "avg_residual_asr_s3": _avg(group_rows, "residual_asr_s3"),
                "avg_s3_correct": _avg(group_rows, "s3_correct"),
                "avg_s4_retention_correct": _avg(group_rows, "s4_retention_correct"),
                "avg_extra_llm_calls_after_detection": _avg(group_rows, "extra_llm_calls_after_detection"),
            }
        )
    return {"groups": by_group, "row_count": len(rows)}


def _avg(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


if __name__ == "__main__":
    main()
