from __future__ import annotations

from src.schemas import CheckpointManifest, EventRecord
from src.store import RunRegistry


def checkpoint_id_for(after_session: str) -> str:
    if after_session == "before_s1":
        return "ckpt_before_s1"
    return f"ckpt_after_{after_session}"


def write_checkpoint(registry: RunRegistry, run_id: str, after_session: str) -> CheckpointManifest:
    checkpoint_id = checkpoint_id_for(after_session)
    event_id = registry.id_allocator.next_event_id("s1" if after_session == "before_s1" else after_session, "checkpoint")
    checkpoint = CheckpointManifest(
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        after_session=after_session,
        active_memory_ids=list(registry.active_memory_ids),
        active_summary_id=registry.active_summary_id,
        quarantined_root_ids=list(registry.quarantined_root_ids),
        event_count=registry.event_count() + 1,
        object_count=registry.object_count(),
    )
    event = EventRecord(
        event_id=event_id,
        run_id=run_id,
        session_id="s1" if after_session == "before_s1" else after_session,
        type="checkpoint",
        input_object_ids=[],
        output_object_ids=[],
        meta={"checkpoint_id": checkpoint_id},
    )
    registry.append_event(event)
    registry.save_checkpoint(checkpoint)
    return checkpoint


def restore_checkpoint(registry: RunRegistry, checkpoint_id: str) -> CheckpointManifest:
    checkpoint = registry.get_checkpoint(checkpoint_id)
    registry.set_active_state(
        active_memory_ids=list(checkpoint.active_memory_ids),
        active_summary_id=checkpoint.active_summary_id,
        quarantined_root_ids=list(checkpoint.quarantined_root_ids),
    )
    return checkpoint
