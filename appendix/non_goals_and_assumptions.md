# Non-Goals and Assumptions

## Hard non-goals

- no detector paper
- no general secure-agent architecture
- no multimodal attack analysis
- no browser or email agent
- no multi-agent delegation
- no parametric unlearning
- no external side-effect compensation
- no token-level provenance
- no giant benchmark
- no general-purpose framework contribution

## Assumptions

- the runtime is single-agent and text-only
- all cross-session influence flows through explicit persisted objects
- local tool results are immutable once logged
- suspicious root objects are provided after compromise
- deterministic execution is feasible enough for auditable replay
- public-doc tasks are simple enough for rule-based scoring
- the local files environment is sufficient to demonstrate the paper claim

## Realism constraints

- single-GPU-friendly workflow
- quantized Qwen3.5-27B direction
- small-team implementation
- simple storage
- simple local tools
- no live web
- limited runtime budget

## Deployment caveats that must not be hidden

- real deployed systems may contain hidden state not tracked here
- real systems may have irreversible side effects
- detection quality matters operationally even though it is out of scope here
- writer-only replay may be insufficient when contaminated actions changed which observations existed
- summary-based memory may be more brittle than episodic memory

## Interpretation caveat

This project is about a narrow and operational question:
what to do **after** compromise when explicit persisted state is known or suspected to be contaminated.

It is not a claim that prompt injection is fully solvable.
