# Selective Revocation and Replay

[![Paper PDF](https://img.shields.io/badge/Paper_PDF-Download-0A66C2?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://github.com/aliuyar1234/selective-revocation-replay/raw/main/paper/selective-revocation-and-replay.pdf)
[![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-008080?style=for-the-badge&logo=latex&logoColor=white)](https://github.com/aliuyar1234/selective-revocation-replay/tree/main/paper/usenix_security26)
[![Frozen Results](https://img.shields.io/badge/Results-Frozen_Artifact-2E8B57?style=for-the-badge)](https://github.com/aliuyar1234/selective-revocation-replay/tree/main/results)
[![Artifact Runbook](https://img.shields.io/badge/Artifact-Runbook-5B4B8A?style=for-the-badge)](https://github.com/aliuyar1234/selective-revocation-replay/blob/main/artifact/runbook.md)
[![Pytest](https://img.shields.io/badge/Pytest-25_passed-2ea44f?style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/aliuyar1234/selective-revocation-replay/tree/main/tests)

This repository contains the code, paper source, and frozen evaluation outputs for **Selective Revocation and Replay: Post-Compromise Recovery of Explicit Persisted State in Memory-Augmented LLM Agents**.

> **Abstract.** This project studies post-compromise recovery for explicit persisted state in memory-augmented LLM agents after persistent indirect prompt injection has already been written into durable state. The core mechanism revokes persisted descendants of suspicious roots and replays only dirty state-writing events when replay is sound, with fallback to coarse rollback otherwise. In a frozen deterministic matrix spanning 8 task chains, 2 memory architectures, 2 attack variants, 5 methods, and 1 ablation, selective replay is the only method that simultaneously achieves zero residual attack success and non-zero benign retention on explicit attacks. On retrieval memory, it matches rollback's retained state while reducing post-detection cost from 17 to 9 extra LLM calls. A focused live confirmation on `qwen2.5:14b` shows the same recovery wedge on repeated explicit retrieval-memory runs.

## Research focus

The project studies post-compromise recovery in a narrow but fully reproducible setting:

- one single-agent runtime
- text-only local files workspace
- two tools: `search_docs` and `read_doc`
- two persisted-state architectures: episodic retrieval memory and rolling summaries
- append-only provenance logging
- selective descendant revocation
- writer-only replay with coarse rollback fallback
- a frozen evaluation matrix over 8 chains, 2 architectures, 2 attack variants, 5 methods, and 1 ablation

## Main findings

- In the deterministic artifact, selective replay is the only method that combines zero residual explicit attack success with non-zero retained benign state.
- In retrieval memory, selective replay matches rollback's retained state while reducing post-detection cost from 17 to 9 extra LLM calls.
- In the focused live confirmation, all six unrecovered reruns keep the attack active while all six selective-replay reruns restore clean follow-up behavior and retained remembered facts.
- The strongest end-to-end live case succeeds with only four revocations and two replayed writer events.

## What is included

- implementation code for the runtime, recovery logic, scoring, and plotting
- the submission-style LaTeX paper under `paper/usenix_security26/`
- the public compiled paper PDF at `paper/selective-revocation-and-replay.pdf`
- frozen aggregate results under `results/raw/`, `results/tables/`, and `results/figures/`
- the focused live-confirmation summary under `results/model_pilots/live_confirmation_latest.*`
- an artifact package with a compact runbook under `artifact/`

Internal planning notes, local handoff material, QA screenshots, transient paper-build files, and release bundles are intentionally kept out of the public Git surface. The only large vendored subtree that remains is the optional local-Qwen fallback under `vendor/`.

## Reproducibility profile

The default backend is `heuristic_artifact`, which makes the frozen evaluation deterministic and fully reproducible from the repository.

Optional model-backed configurations remain available through `configs/models.yaml`:

- `qwen2_5_14b_ollama`
- `qwen2_5_32b_ollama`
- `qwen2_5_7b_instruct`
- `qwen3_5_27b_instruct`

For local Qwen checkpoints, set a local model path in the corresponding config or provide `LOCAL_QWEN_MODEL_PATH` in your environment before running the backend. If `transformers` and its runtime dependencies are already installed in your environment, the local-Qwen path uses them directly. The vendored `vendor/` tree is a fallback for that optional path, not a requirement for the deterministic artifact.

## Main entry points

- `python scripts/build_workspace.py --chain c01_travel --attack explicit`
- `python scripts/run_base_history.py --chain c01_travel --architecture retrieval --attack explicit`
- `python scripts/apply_method.py --base-history <run_id> --method selective_replay`
- `python scripts/run_eval_matrix.py`
- `python scripts/run_live_confirmation_batch.py`
- `python scripts/make_tables.py`
- `python scripts/make_figures.py`
- `python scripts/build_paper_artifacts.py`

## Repository layout

- `configs/`: runtime, evaluation, and backend settings
- `prompts/`: prompt templates
- `src/`: runtime, recovery, scoring, and plotting modules
- `scripts/`: command-line entry points
- `data/workspace/`: chain workspaces and source files
- `results/`: frozen aggregate outputs and paper-ready artifacts
- `paper/`: public compiled PDF and standalone LaTeX source package
- `artifact/`: compact reproducibility package
- `appendix/`: supporting appendix material and chain catalog
- `vendor/`: bundled fallback dependencies for the optional local Qwen backend

Directories that hold bulky local run outputs, transient paper build files, local preview renders, handoff packages, or archived working material are intentionally ignored by Git.

## Read the paper

Readers who only want the paper and frozen evidence can start with:

- `paper/selective-revocation-and-replay.pdf`
- `paper/usenix_security26/main.tex`
- `results/tables/`
- `results/figures/`
- `artifact/runbook.md`

The `Paper PDF` badge above points to GitHub's direct raw-file endpoint for the compiled PDF so readers can open or download the paper immediately.

## Quick start

1. Install the package and developer dependencies.
2. Materialize a workspace or let the run scripts do it automatically.
3. Run a base history through S1-S2.
4. Apply one or more recovery methods to the same compromised base history.
5. Run the full frozen matrix with the deterministic backend.
6. Generate tables and figures.
7. Build the paper with `powershell -ExecutionPolicy Bypass -File paper/usenix_security26/build.ps1`.
   This refreshes the public PDF at `paper/selective-revocation-and-replay.pdf`.

## Verification

- `python -m pytest -q`
- `python scripts/make_tables.py`
- `python scripts/make_figures.py`

The frozen aggregate outputs needed to inspect the current paper claims are already included in the repository.
