from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.agent_loop import run_sessions
from src.baselines import apply_method
from src.checkpoint import write_checkpoint
from src.env_workspace import WorkspaceEnv, materialize_chain_workspace
from src.ids import build_run_id, utc_timestamp
from src.llm_backend import build_backend
from src.schemas import ResultRecord, TaskCatalog
from src.scoring import result_record_for_run
from src.store import RunRegistry
from src.utils import append_jsonl, dump_json, ensure_dir, load_yaml, write_csv


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_model_config(root: Path, model_index: dict[str, Any], model_name: str, *, env_var_name: str) -> dict[str, Any]:
    available_models = model_index.get("available_models", {})
    if not isinstance(available_models, dict):
        raise ValueError("configs/models.yaml must define available_models as a mapping.")
    model_relpath = available_models.get(model_name)
    if model_relpath is None:
        available_names = ", ".join(sorted(str(name) for name in available_models))
        raise ValueError(f"Unknown model '{model_name}' from {env_var_name}. Available models: {available_names}")
    model_cfg = load_yaml(root / "configs" / str(model_relpath))
    model_cfg["model_name"] = model_name
    model_cfg["config_file"] = str(model_relpath)
    return model_cfg


def load_project_settings() -> dict[str, Any]:
    root = repo_root()
    paths = load_yaml(root / "configs" / "paths.yaml")
    eval_matrix = load_yaml(root / "configs" / "eval_matrix.yaml")
    model_index = load_yaml(root / "configs" / "models.yaml")
    active_model_name = os.environ.get("SRR_ACTIVE_MODEL", model_index["default_model"])
    model_cfg = resolve_model_config(root, model_index, active_model_name, env_var_name="SRR_ACTIVE_MODEL")
    runtime_cfg = {
        "retrieval": load_yaml(root / "configs" / "runtime_retrieval.yaml"),
        "summary": load_yaml(root / "configs" / "runtime_summary.yaml"),
    }
    catalog = TaskCatalog.load(root / paths["catalog_path"])
    return {
        "root": root,
        "paths": paths,
        "eval_matrix": eval_matrix,
        "model_index": model_index,
        "active_model_name": active_model_name,
        "model": model_cfg,
        "runtime": runtime_cfg,
        "catalog": catalog,
        "token_regex": catalog.global_rules.search_scoring["token_regex"],
    }


def create_base_history_run(chain_id: str, architecture: str, attack_variant: str, seed: int = 1, timestamp: str | None = None) -> tuple[str, Path]:
    settings = load_project_settings()
    root = settings["root"]
    catalog = settings["catalog"]
    chain = catalog.get_chain(chain_id)
    runtime_cfg = settings["runtime"][architecture]
    stamp = timestamp or utc_timestamp()
    run_id = build_run_id("hist", architecture, attack_variant, chain_id, seed, stamp)
    workspace = materialize_chain_workspace(catalog, chain_id, attack_variant, root / settings["paths"]["workspace_root"])
    run_dir = ensure_dir(root / settings["paths"]["base_history_root"] / run_id)
    registry = RunRegistry(run_dir)
    backend = build_backend(settings["model"]["backend"], settings["model"])
    metadata = {
        "run_id": run_id,
        "base_history_run_id": run_id,
        "method": "base_history",
        "chain_id": chain_id,
        "architecture": architecture,
        "attack_variant": attack_variant,
        "seed": seed,
        "backend": backend.backend_name,
        "workspace_dir": str(workspace.root),
        "attack_path": workspace.attack_path,
        "detection_after_session": settings["eval_matrix"]["detection_after_session"],
        "prompt_versions": dict(runtime_cfg["prompt_versions"]),
        "model_name": settings["model"]["model_name"],
        "model_id": settings["model"]["model_id"],
        "model_config_file": settings["model"]["config_file"],
        "base_event_count": 0,
        "notes": "",
    }
    registry.save_metadata(metadata)
    env = WorkspaceEnv(chain, workspace.root, settings["token_regex"])
    write_checkpoint(registry, run_id=run_id, after_session="before_s1")
    run_sessions(
        registry=registry,
        env=env,
        backend=backend,
        chain=chain,
        architecture=architecture,
        run_id=run_id,
        session_ids=["s1", "s2"],
        max_action_steps=int(runtime_cfg["max_action_steps"]),
        memory_retrieval_k=int(runtime_cfg["memory_retrieval_k"]),
        prompt_versions=dict(runtime_cfg["prompt_versions"]),
    )
    metadata["base_event_count"] = registry.event_count()
    metadata["base_object_count"] = registry.object_count()
    registry.save_metadata(metadata)
    return run_id, run_dir


def create_method_run(base_history_run_id: str, method: str, timestamp: str | None = None) -> tuple[str, Path, ResultRecord]:
    settings = load_project_settings()
    root = settings["root"]
    base_dir = root / settings["paths"]["base_history_root"] / base_history_run_id
    base_registry = RunRegistry(base_dir)
    base_metadata = base_registry.load_metadata()
    chain_id = str(base_metadata["chain_id"])
    architecture = str(base_metadata["architecture"])
    attack_variant = str(base_metadata["attack_variant"])
    seed = int(base_metadata.get("seed", 1))
    stamp = timestamp or utc_timestamp()
    run_id = build_run_id(f"run_{method}", architecture, attack_variant, chain_id, seed, stamp)
    run_dir = ensure_dir(root / settings["paths"]["method_run_root"] / run_id)
    _fork_run_directory(base_dir=base_dir, method_dir=run_dir, method_run_id=run_id)
    registry = RunRegistry(run_dir)
    chain = settings["catalog"].get_chain(chain_id)
    env = WorkspaceEnv(chain, base_metadata["workspace_dir"], settings["token_regex"])
    backend = build_backend(settings["model"]["backend"], settings["model"])
    runtime_cfg = settings["runtime"][architecture]
    metadata = registry.load_metadata()
    metadata.update(
        {
            "run_id": run_id,
            "base_history_run_id": base_history_run_id,
            "method": method,
            "backend": backend.backend_name,
            "model_name": settings["model"]["model_name"],
            "model_id": settings["model"]["model_id"],
            "model_config_file": settings["model"]["config_file"],
            "base_event_count": registry.event_count(),
            "base_object_count": registry.object_count(),
            "notes": "",
        }
    )
    registry.save_metadata(metadata)
    apply_method(
        registry=registry,
        env=env,
        backend=backend,
        chain=chain,
        architecture=architecture,
        run_id=run_id,
        base_history_run_id=base_history_run_id,
        method=method,
        attack_path=str(base_metadata["attack_path"]),
        max_action_steps=int(runtime_cfg["max_action_steps"]),
        memory_retrieval_k=int(runtime_cfg["memory_retrieval_k"]),
        prompt_versions=dict(runtime_cfg["prompt_versions"]),
    )
    result = result_record_for_run(registry, chain)
    return run_id, run_dir, result


def run_full_matrix(timestamp: str | None = None) -> list[ResultRecord]:
    settings = load_project_settings()
    eval_cfg = settings["eval_matrix"]
    stamp = timestamp or utc_timestamp()
    results: list[ResultRecord] = []
    base_history_ids: list[str] = []
    for chain_id in eval_cfg["full_chain_ids"]:
        for architecture in eval_cfg["architectures"]:
            for attack_variant in eval_cfg["attacks"]:
                base_history_id, _ = create_base_history_run(chain_id, architecture, attack_variant, seed=int(eval_cfg["seed"]), timestamp=stamp)
                base_history_ids.append(base_history_id)
                for method in eval_cfg["methods"]:
                    _, _, result = create_method_run(base_history_id, method, timestamp=stamp)
                    results.append(result)
    raw_jsonl = settings["root"] / settings["paths"]["raw_results_path"]
    raw_csv = settings["root"] / settings["paths"]["raw_results_csv_path"]
    raw_jsonl.unlink(missing_ok=True)
    for result in results:
        append_jsonl(raw_jsonl, result.to_dict())
    write_csv(raw_csv, [result.to_dict() for result in results], fieldnames=list(results[0].to_dict().keys()))
    return results


def load_results_from_raw() -> list[dict[str, Any]]:
    settings = load_project_settings()
    raw_path = settings["root"] / settings["paths"]["raw_results_path"]
    if not raw_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _fork_run_directory(base_dir: Path, method_dir: Path, method_run_id: str) -> None:
    ensure_dir(method_dir)
    base_registry = RunRegistry(base_dir)
    metadata = base_registry.load_metadata()
    metadata["run_id"] = method_run_id
    dump_json(method_dir / "run_metadata.json", metadata)

    with (method_dir / "objects.jsonl").open("w", encoding="utf-8") as object_handle:
        for row in (base_dir / "objects.jsonl").read_text(encoding="utf-8").splitlines():
            if not row.strip():
                continue
            payload = json.loads(row)
            payload["run_id"] = method_run_id
            object_handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")

    with (method_dir / "events.jsonl").open("w", encoding="utf-8") as event_handle:
        for row in (base_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            if not row.strip():
                continue
            payload = json.loads(row)
            payload["run_id"] = method_run_id
            event_handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")

    checkpoint_dir = ensure_dir(method_dir / "checkpoints")
    for checkpoint_file in sorted((base_dir / "checkpoints").glob("*.json")):
        payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        payload["run_id"] = method_run_id
        dump_json(checkpoint_dir / checkpoint_file.name, payload)
