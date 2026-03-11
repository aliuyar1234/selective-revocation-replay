# Artifact Overview

This artifact packages the public reproducibility surface for the deterministic implementation of selective revocation and replay.

## What is included

- workspace materialization from the frozen chain catalog
- base-history generation for attacked S1-S2 runs
- method-run forking from shared base histories
- recovery and baseline implementations
- frozen aggregate results for the evaluation matrix
- generated tables and figures
- the paper-source package and copied references

## Main output locations

- `results/raw/eval_results.jsonl`: raw result records
- `results/tables/`: CSV tables
- `results/figures/`: PDF and PNG figures
- `results/model_pilots/live_confirmation_latest.*`: focused live-model confirmation summary
- `paper/`: paper draft, references, copied figure/table assets

Per-run histories and method directories are generated locally when the scripts are executed and are intentionally excluded from the public Git surface.

## Important scope note

The released artifact runs end to end with a deterministic heuristic backend for reproducibility. The runtime interface is intentionally narrow so that a local Qwen-style backend can replace the heuristic layer later without changing the recovery or evaluation pipeline.
