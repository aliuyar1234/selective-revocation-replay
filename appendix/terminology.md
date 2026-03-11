# Terminology

## Suspicious root
An explicit object ID, provided after compromise, believed to contain or originate malicious content. In this project it is usually a `tool_result` object from `read_doc` for the selected malicious file.

## Persisted state
Cross-session state explicitly stored as:
- episodic memory entries
- rolling summary snapshots

## Immutable object
An object whose original content is preserved in the audit log:
- user turn
- search result list
- file-read result

## Descendant revocation
Marking all persisted objects transitively derived from suspicious roots as unusable for future state.

## Root-only delete
Quarantining only the **direct persisted children** of suspicious roots, with no transitive closure and no replay.

## Writer-only replay
Replaying only the events that materialize persisted state objects:
- memory writes
- summary writes

## Coarse rollback
Restoring the last clean checkpoint before an affected session and replaying the entire suffix of sessions.

## Replay-safe
A dirty write event is replay-safe if every immutable tool result it consumed was requested by an `llm_act` whose input objects did not intersect the suspicious-root set or its persisted-descendant closure.

## Replay-unsafe
A dirty write event is replay-unsafe if writer-only replay would trust tool results that were observed because a contaminated prior action changed what the agent fetched.

## Benign carry-forward state
User- or task-specific information that should survive recovery and is needed for a future benign task.

## Residual ASR
Attack success rate after recovery, measured on S3.

## S3 correctness
Whether the S3 benign task is answered correctly from public docs without restricted reads or secret leakage.

## S4 retention accuracy
Whether the system still uses the remembered benign carry-forward fact correctly after recovery.

## Active state
The set of persisted objects currently eligible for retrieval or injection into future sessions.

## Replaced object
An original contaminated persisted object that has been removed from active state because a replay-produced clean replacement exists.
