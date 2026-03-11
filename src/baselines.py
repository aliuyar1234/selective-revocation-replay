from __future__ import annotations

from src.agent_loop import run_sessions
from src.recover import (
    apply_revocations_without_replay,
    build_replay_plan,
    coarse_rollback_and_replay_suffix,
    compute_persisted_descendant_closure,
    replay_dirty_writers,
    select_suspicious_root_ids,
    suspicious_root_paths,
)
from src.store import RunRegistry


def apply_method(
    registry: RunRegistry,
    env,
    backend,
    chain,
    architecture: str,
    run_id: str,
    base_history_run_id: str,
    method: str,
    attack_path: str,
    max_action_steps: int,
    memory_retrieval_k: int,
    prompt_versions: dict[str, str],
) -> tuple[list[str], bool]:
    suspicious_root_ids = select_suspicious_root_ids(registry, attack_path)
    registry_metadata = registry.load_metadata()
    registry_metadata["suspicious_root_ids"] = suspicious_root_ids
    registry_metadata["method"] = method
    registry.save_metadata(registry_metadata)

    revoked_object_ids: list[str] = []
    fallback = False
    quarantined_paths = suspicious_root_paths(registry, suspicious_root_ids)

    if method == "no_recovery":
        quarantined_paths = set()
    elif method == "root_delete":
        registry.quarantined_root_ids = list(suspicious_root_ids)
        direct_children = [
            obj.object_id
            for obj in registry.objects.values()
            if obj.kind in {"memory", "summary"} and set(obj.parent_ids).intersection(set(suspicious_root_ids))
        ]
        apply_revocations_without_replay(registry, direct_children)
        revoked_object_ids = direct_children
    elif method == "full_reset":
        registry.quarantined_root_ids = list(suspicious_root_ids)
        active_ids = [obj.object_id for obj in registry.objects.values() if obj.kind in {"memory", "summary"} and obj.status == "active"]
        apply_revocations_without_replay(registry, active_ids)
        revoked_object_ids = active_ids
    elif method == "revoke_no_replay":
        registry.quarantined_root_ids = list(suspicious_root_ids)
        revoked_object_ids = compute_persisted_descendant_closure(registry, suspicious_root_ids)
        apply_revocations_without_replay(registry, revoked_object_ids)
    elif method == "coarse_rollback":
        revoked_object_ids = compute_persisted_descendant_closure(registry, suspicious_root_ids)
        coarse_rollback_and_replay_suffix(
            registry=registry,
            env=env,
            backend=backend,
            chain=chain,
            architecture=architecture,
            run_id=run_id,
            suspicious_root_ids=suspicious_root_ids,
            revoked_object_ids=revoked_object_ids,
            max_action_steps=max_action_steps,
            memory_retrieval_k=memory_retrieval_k,
            prompt_versions=prompt_versions,
        )
    elif method == "selective_replay":
        revoked_object_ids = compute_persisted_descendant_closure(registry, suspicious_root_ids)
        replay_plan = build_replay_plan(
            registry=registry,
            base_history_run_id=base_history_run_id,
            method_run_id=run_id,
            suspicious_root_ids=suspicious_root_ids,
            revoked_object_ids=revoked_object_ids,
        )
        registry.save_replay_plan(replay_plan)
        if replay_plan.mode == "writer_only":
            registry.quarantined_root_ids = list(suspicious_root_ids)
            replay_plan = replay_dirty_writers(
                registry=registry,
                backend=backend,
                chain=chain,
                architecture=architecture,
                run_id=run_id,
                replay_plan=replay_plan,
                prompt_versions=prompt_versions,
            )
            registry.save_replay_plan(replay_plan)
        else:
            fallback = True
            coarse_rollback_and_replay_suffix(
                registry=registry,
                env=env,
                backend=backend,
                chain=chain,
                architecture=architecture,
                run_id=run_id,
                suspicious_root_ids=suspicious_root_ids,
                revoked_object_ids=revoked_object_ids,
                max_action_steps=max_action_steps,
                memory_retrieval_k=memory_retrieval_k,
                prompt_versions=prompt_versions,
            )
    else:
        raise ValueError(f"Unknown method {method}")

    if method not in {"coarse_rollback"} and not (method == "selective_replay" and fallback):
        run_sessions(
            registry=registry,
            env=env,
            backend=backend,
            chain=chain,
            architecture=architecture,
            run_id=run_id,
            session_ids=["s3", "s4"],
            max_action_steps=max_action_steps,
            memory_retrieval_k=memory_retrieval_k,
            prompt_versions=prompt_versions,
            quarantined_paths=quarantined_paths,
        )
    return revoked_object_ids, fallback
