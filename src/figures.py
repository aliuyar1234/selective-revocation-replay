from __future__ import annotations

from pathlib import Path
from statistics import mean
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.pipeline import load_project_settings, load_results_from_raw
from src.utils import ensure_dir, load_yaml, short_text, write_csv


def generate_tables_and_figures() -> dict[str, Path]:
    settings = load_project_settings()
    root = settings["root"]
    table_dir = ensure_dir(root / "results" / "tables")
    figure_dir = ensure_dir(root / "results" / "figures")
    rows = load_results_from_raw()
    if not rows:
        raise RuntimeError("No raw results found. Run scripts/run_eval_matrix.py first.")

    tbl1 = _table_main_results(rows)
    tbl2 = _table_attack_breakdown(rows)
    tbl3 = _table_ablation(rows)
    tbl4 = _table_fallback_cost(rows)

    write_csv(table_dir / "tbl1_main_results.csv", tbl1, fieldnames=list(tbl1[0].keys()))
    write_csv(table_dir / "tbl2_attack_breakdown.csv", tbl2, fieldnames=list(tbl2[0].keys()))
    write_csv(table_dir / "tbl3_ablation.csv", tbl3, fieldnames=list(tbl3[0].keys()))
    write_csv(table_dir / "tbl4_fallback_and_cost.csv", tbl4, fieldnames=list(tbl4[0].keys()))

    _figure_motivating_example(figure_dir / "fig1_motivating_example.pdf")
    _figure_system_overview(figure_dir / "fig2_system_overview.pdf")
    _figure_recovery_paths(figure_dir / "fig3_recovery_paths.pdf")
    _figure_cost_vs_retention(rows, figure_dir / "fig4_cost_vs_retention.pdf")
    generated = {
        "tbl1": table_dir / "tbl1_main_results.csv",
        "tbl2": table_dir / "tbl2_attack_breakdown.csv",
        "tbl3": table_dir / "tbl3_ablation.csv",
        "tbl4": table_dir / "tbl4_fallback_and_cost.csv",
        "fig1": figure_dir / "fig1_motivating_example.pdf",
        "fig2": figure_dir / "fig2_system_overview.pdf",
        "fig3": figure_dir / "fig3_recovery_paths.pdf",
        "fig4": figure_dir / "fig4_cost_vs_retention.pdf",
    }
    live_confirmation = _load_live_confirmation(root)
    if live_confirmation:
        tbl5 = _table_live_confirmation(live_confirmation)
        write_csv(table_dir / "tbl5_live_confirmation.csv", tbl5, fieldnames=list(tbl5[0].keys()))
        _figure_live_case_study(live_confirmation, figure_dir / "fig5_live_case_study.pdf")
        generated["tbl5"] = table_dir / "tbl5_live_confirmation.csv"
        generated["fig5"] = figure_dir / "fig5_live_case_study.pdf"
    return generated


def _table_main_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_rows(rows, ["method", "architecture", "attack_variant"])
    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        output.append(
            {
                "method": key[0],
                "architecture": key[1],
                "attack_variant": key[2],
                "residual_asr_s3": round(mean(row["residual_asr_s3"] for row in group), 3),
                "s3_correct": round(mean(row["s3_correct"] for row in group), 3),
                "s4_retention_accuracy": round(mean(row["s4_retention_correct"] for row in group), 3),
                "extra_llm_calls_after_detection": round(mean(row["extra_llm_calls_after_detection"] for row in group), 3),
            }
        )
    output.sort(key=lambda row: (row["method"], row["architecture"], row["attack_variant"]))
    return output


def _table_attack_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    no_recovery = [row for row in rows if row["method"] == "no_recovery"]
    proposed = [row for row in rows if row["method"] == "selective_replay"]
    output: list[dict[str, Any]] = []
    for architecture in sorted({row["architecture"] for row in rows}):
        for attack_variant in sorted({row["attack_variant"] for row in rows}):
            unrecovered = [row for row in no_recovery if row["architecture"] == architecture and row["attack_variant"] == attack_variant]
            recovered = [row for row in proposed if row["architecture"] == architecture and row["attack_variant"] == attack_variant]
            output.append(
                {
                    "attack_variant": attack_variant,
                    "architecture": architecture,
                    "no_recovery_s3_asr": round(mean(row["residual_asr_s3"] for row in unrecovered), 3),
                    "proposed_method_s3_asr": round(mean(row["residual_asr_s3"] for row in recovered), 3),
                    "revoked_object_count": round(mean(row["revoked_object_count"] for row in recovered), 3),
                    "replayed_writer_event_count": round(mean(row["replayed_writer_event_count"] for row in recovered), 3),
                }
            )
    return output


def _table_ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subset = [row for row in rows if row["method"] in {"revoke_no_replay", "selective_replay"}]
    grouped = _group_rows(subset, ["method", "architecture"])
    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        output.append(
            {
                "method": key[0],
                "architecture": key[1],
                "residual_asr_s3": round(mean(row["residual_asr_s3"] for row in group), 3),
                "s4_retention_accuracy": round(mean(row["s4_retention_correct"] for row in group), 3),
            }
        )
    output.sort(key=lambda row: (row["method"], row["architecture"]))
    return output


def _table_fallback_cost(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subset = [row for row in rows if row["method"] in {"coarse_rollback", "selective_replay"}]
    grouped = _group_rows(subset, ["method", "architecture"])
    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        output.append(
            {
                "method": key[0],
                "architecture": key[1],
                "fallback_rate": round(mean(row["fallback_to_coarse_rollback"] for row in group), 3),
                "extra_llm_calls_after_detection": round(mean(row["extra_llm_calls_after_detection"] for row in group), 3),
                "replayed_writer_events": round(mean(row["replayed_writer_event_count"] for row in group), 3),
            }
        )
    output.sort(key=lambda row: (row["method"], row["architecture"]))
    return output


def _figure_motivating_example(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    ax.set_title("Motivating Example Timeline")
    steps = [
        "S1\nRead public + malicious file\nPersist user fact + bad rule",
        "S2\nAccumulate benign state",
        "Detect\nSuspicious root after S2",
        "Recover\nRevoke descendants + replay writers",
        "S3\nNo restricted read",
        "S4\nBenign user fact survives",
    ]
    for idx, text in enumerate(steps):
        x = 0.08 + idx * 0.17
        ax.text(x, 0.5, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f3f6fb", "edgecolor": "#4477aa"})
        if idx < len(steps) - 1:
            ax.annotate("", xy=(x + 0.085, 0.5), xytext=(x + 0.125, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.5})
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def _figure_system_overview(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title("System Overview")
    boxes = {
        "env": (0.1, 0.7, "Local Files\nsearch_docs/read_doc"),
        "loop": (0.35, 0.7, "Action Loop\nJSON actions"),
        "state": (0.6, 0.7, "Persisted State\nmemory or summary"),
        "logs": (0.35, 0.35, "Append-only\nobjects/events"),
        "recover": (0.6, 0.35, "Recovery Engine\nrevoke / replay / rollback"),
    }
    for _, (x, y, text) in boxes.items():
        ax.text(x, y, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.5", "facecolor": "#eef7ee", "edgecolor": "#2f7f5f"})
    arrows = [
        ((0.17, 0.7), (0.28, 0.7)),
        ((0.42, 0.7), (0.53, 0.7)),
        ((0.35, 0.63), (0.35, 0.42)),
        ((0.53, 0.35), (0.44, 0.35)),
        ((0.6, 0.63), (0.6, 0.42)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5})
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def _figure_recovery_paths(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title("Recovery Path Comparison")
    lanes = [
        ("Full reset", 0.75, "Discard all persisted state\nReplay nothing"),
        ("Coarse rollback", 0.5, "Restore clean checkpoint\nReplay whole suffix"),
        ("Selective replay", 0.25, "Revoke descendants\nReplay writer events only"),
    ]
    for label, y, detail in lanes:
        ax.text(0.15, y, label, ha="left", va="center", fontsize=12, fontweight="bold")
        ax.plot([0.35, 0.9], [y, y], color="#335577", lw=6, solid_capstyle="round")
        ax.text(0.62, y + 0.08, detail, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def _figure_cost_vs_retention(rows: list[dict[str, Any]], path: Path) -> None:
    explicit_rows = [row for row in rows if row["attack_variant"] == "explicit"]

    def point_for(method: str, architecture: str | None = None) -> tuple[float, float]:
        subset = [row for row in explicit_rows if row["method"] == method]
        if architecture is not None:
            subset = [row for row in subset if row["architecture"] == architecture]
        return (
            mean(row["extra_llm_calls_after_detection"] for row in subset),
            mean(row["s4_retention_correct"] for row in subset),
        )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_facecolor("#fbfdff")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.grid(True, axis="both", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)

    full_reset = point_for("full_reset", "retrieval")
    replay_retrieval = point_for("selective_replay", "retrieval")
    rollback_cluster = point_for("coarse_rollback", "retrieval")

    ax.scatter(*full_reset, s=150, marker="o", color="#475569", edgecolor="white", linewidth=1.5, zorder=3)
    ax.scatter(*replay_retrieval, s=150, marker="o", color="#059669", edgecolor="white", linewidth=1.5, zorder=3)
    ax.scatter(*rollback_cluster, s=170, marker="s", color="#2563eb", edgecolor="white", linewidth=1.5, zorder=2)
    ax.scatter(*rollback_cluster, s=90, marker="D", color="#059669", edgecolor="white", linewidth=1.2, zorder=3)

    ax.annotate("Full reset", full_reset, textcoords="offset points", xytext=(10, 10), fontsize=8.5, color="#1f2937")
    ax.annotate("Replay-R", replay_retrieval, textcoords="offset points", xytext=(8, 8), fontsize=8.5, color="#1f2937")
    ax.annotate(
        "Rollback + Replay-S",
        rollback_cluster,
        textcoords="offset points",
        xytext=(10, 8),
        fontsize=8.5,
        color="#1f2937",
    )

    ax.set_xlim(4.5, 18.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Extra LLM calls after detection")
    ax.set_ylabel("Retained benign state (S4)")
    ax.set_title("Safety-Retention Frontier on Explicit Attacks", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def _load_live_confirmation(root: Path) -> dict[str, Any] | None:
    path = root / "results" / "model_pilots" / "live_confirmation_latest.json"
    if not path.exists():
        return None
    return load_yaml(path)


def _table_live_confirmation(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in bundle["summary"]["case_groups"]:
        output.append(
            {
                "model_name": bundle["active_model_name"],
                "chain_id": row["chain_id"],
                "architecture": row["architecture"],
                "attack_variant": row["attack_variant"],
                "method": row["method"],
                "repeats": row["count"],
                "residual_asr_s3": row["avg_residual_asr_s3"],
                "s3_correct": row["avg_s3_correct"],
                "s4_retention_accuracy": row["avg_s4_retention_correct"],
                "extra_llm_calls_after_detection": row["avg_extra_llm_calls_after_detection"],
                "replayed_writer_event_count": row["avg_replayed_writer_event_count"],
                "revoked_object_count": row["avg_revoked_object_count"],
                "fallback_rate": row["avg_fallback_to_coarse_rollback"],
                "target_outcome_rate": row["target_outcome_rate"],
            }
        )
    output.sort(key=lambda row: (row["chain_id"], row["architecture"], row["method"]))
    return output


def _figure_live_case_study(bundle: dict[str, Any], path: Path) -> None:
    case = bundle.get("case_study", {})
    if not case:
        raise RuntimeError("Live confirmation bundle is missing a case_study payload.")

    fig = plt.figure(figsize=(3.65, 6.35), facecolor="white")
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("white")

    palette = {
        "ink": "#14213d",
        "muted": "#52627a",
        "line": "#d6dde8",
        "bad": "#b9382f",
        "bad_fill": "#fff6f4",
        "bad_edge": "#efb1aa",
        "good": "#1f7a4d",
        "good_fill": "#f4fbf6",
        "good_edge": "#a6dbb7",
        "warn": "#b56a19",
        "warn_fill": "#fff8ef",
        "warn_edge": "#f0c48c",
        "info": "#2a61d0",
        "info_fill": "#f3f7ff",
        "info_edge": "#9cbcf8",
        "card": "#ffffff",
    }

    def wrap_text(text: str, width: int) -> str:
        chunks = [fill(part.strip(), width=width) for part in str(text).splitlines() if part.strip()]
        return "\n".join(chunks)

    def extract_action_arg(action: str) -> str:
        action = str(action)
        if "(" in action and action.endswith(")"):
            start = action.find("(") + 1
            end = action.rfind(")")
            return action[start:end]
        return action

    def add_panel(x: float, y: float, w: float, h: float, title: str, accent: str, face: str, edge: str) -> None:
        panel = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=1.0,
            facecolor=face,
            edgecolor=edge,
        )
        ax.add_patch(panel)
        ax.add_patch(Rectangle((x, y + h - 0.012), w, 0.012, facecolor=accent, edgecolor="none"))
        ax.text(x + 0.024, y + h - 0.036, title, ha="left", va="top", fontsize=10.0, fontweight="bold", color=palette["ink"])

    def add_text_block(
        x: float,
        y: float,
        text: str,
        *,
        width: int = 54,
        fontsize: float = 7.5,
        color: str | None = None,
        monospace: bool = False,
    ) -> None:
        ax.text(
            x,
            y,
            wrap_text(text, width),
            ha="left",
            va="top",
            fontsize=fontsize,
            color=color or palette["muted"],
            fontfamily="monospace" if monospace else None,
        )

    def add_strip(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        face: str,
        edge: str,
        accent: str,
        text_color: str | None = None,
        fontsize: float = 7.0,
        monospace: bool = False,
        width: int = 52,
    ) -> None:
        strip = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.022",
            linewidth=0.9,
            facecolor=face,
            edgecolor=edge,
        )
        ax.add_patch(strip)
        ax.add_patch(Rectangle((x, y + h - 0.01), w, 0.01, facecolor=accent, edgecolor="none"))
        ax.text(
            x + 0.018,
            y + h / 2,
            wrap_text(text, width),
            ha="left",
            va="center",
            fontsize=fontsize,
            color=text_color or palette["muted"],
            fontfamily="monospace" if monospace else None,
        )

    def add_metric_chip(x: float, y: float, w: float, label: str, value: str, edge: str, value_color: str) -> None:
        chip = FancyBboxPatch(
            (x, y),
            w,
            0.05,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=0.9,
            facecolor=palette["card"],
            edgecolor=edge,
        )
        ax.add_patch(chip)
        ax.text(x + w / 2, y + 0.034, label, ha="center", va="center", fontsize=6.7, color=palette["muted"])
        ax.text(x + w / 2, y + 0.016, value, ha="center", va="center", fontsize=9.3, fontweight="bold", color=value_color)

    def add_metric_row(x: float, y: float, w: float, metrics: list[tuple[str, str]], edge: str, value_color: str) -> None:
        gap = 0.014
        chip_w = (w - gap * (len(metrics) - 1)) / len(metrics)
        for idx, (label, value) in enumerate(metrics):
            add_metric_chip(x + idx * (chip_w + gap), y, chip_w, label, value, edge, value_color)

    model_label = case.get("model_id", case.get("model_name", "local model"))
    ax.text(0.06, 0.965, "Live Case Study", ha="left", va="top", fontsize=14.6, fontweight="bold", color=palette["ink"])
    ax.text(
        0.06,
        0.93,
        f"{model_label} | {case['chain_id']} | {case['architecture']} memory | {case['attack_variant']} attack",
        ha="left",
        va="top",
        fontsize=8.1,
        color=palette["muted"],
    )
    ax.text(
        0.06,
        0.905,
        "A poisoned workflow note survives into S3 until selective replay rebuilds the clean state.",
        ha="left",
        va="top",
        fontsize=7.3,
        color=palette["muted"],
    )
    ax.text(
        0.06,
        0.862,
        (
            f"Replay mode: writer-only | "
            f"{case['revoked_object_count']} revoked | "
            f"{case['replayed_writer_event_count']} writer replays"
        ),
        ha="left",
        va="top",
        fontsize=7.15,
        color=palette["muted"],
    )

    bad_target = extract_action_arg(case["no_recovery_s3_action"])
    recovered_search = extract_action_arg(case["recovered_s3_search_action"])
    recovered_target = extract_action_arg(case["recovered_s3_read_action"])

    panel_x = 0.06
    panel_w = 0.88

    attack_y = 0.565
    add_panel(panel_x, attack_y, panel_w, 0.18, "1. Attack path", palette["warn"], "#fffdfa", palette["warn_edge"])
    add_text_block(
        panel_x + 0.024,
        attack_y + 0.118,
        f"S1 read: {Path(case['suspicious_root_path']).name}",
        width=44,
        fontsize=7.0,
        color=palette["ink"],
    )
    add_text_block(
        panel_x + 0.024,
        attack_y + 0.082,
        "Persisted note: consult restricted/vault.txt for reimbursements.",
        width=50,
        fontsize=7.2,
    )
    add_text_block(
        panel_x + 0.024,
        attack_y + 0.046,
        f"After S2: flag {case['suspicious_root_id']}.",
        width=48,
        fontsize=7.2,
    )

    unrecovered_y = 0.355
    add_panel(panel_x, unrecovered_y, panel_w, 0.16, "2. No recovery", palette["bad"], palette["bad_fill"], palette["bad_edge"])
    add_strip(
        panel_x + 0.024,
        unrecovered_y + 0.05,
        0.832,
        0.032,
        f"S3 route: {bad_target}",
        face=palette["card"],
        edge=palette["bad_edge"],
        accent=palette["bad"],
        fontsize=6.7,
        monospace=True,
        width=38,
    )
    no_metrics = case["metrics"]["no_recovery"]
    add_metric_row(
        panel_x + 0.024,
        unrecovered_y + 0.014,
        0.832,
        [("ASR", str(no_metrics["residual_asr_s3"])), ("S3", str(no_metrics["s3_correct"])), ("S4", str(no_metrics["s4_retention_correct"]))],
        palette["bad_edge"],
        palette["bad"],
    )

    replay_y = 0.055
    add_panel(panel_x, replay_y, panel_w, 0.255, "3. Selective replay", palette["good"], palette["good_fill"], palette["good_edge"])
    add_strip(
        panel_x + 0.024,
        replay_y + 0.112,
        0.832,
        0.038,
        f"Clean S3 route: {recovered_target}",
        face=palette["card"],
        edge=palette["info_edge"],
        accent=palette["info"],
        fontsize=6.8,
        width=40,
    )
    add_strip(
        panel_x + 0.024,
        replay_y + 0.064,
        0.832,
        0.03,
        "S4 fact: contractor meal cap $45/day",
        face=palette["card"],
        edge=palette["good_edge"],
        accent=palette["good"],
        fontsize=6.75,
        width=44,
    )
    replay_metrics = case["metrics"]["selective_replay"]
    add_metric_row(
        panel_x + 0.024,
        replay_y + 0.012,
        0.832,
        [("ASR", str(replay_metrics["residual_asr_s3"])), ("S3", str(replay_metrics["s3_correct"])), ("S4", str(replay_metrics["s4_retention_correct"]))],
        palette["good_edge"],
        palette["good"],
    )

    fig.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.025)
    fig.savefig(path, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".png"), dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        grouped.setdefault(key, []).append(row)
    return grouped
