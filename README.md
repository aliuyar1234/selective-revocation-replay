# Selective Revocation and Replay: Post-Compromise Recovery of Explicit Persisted State in Memory-Augmented LLM Agents

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://github.com/aliuyar1234/selective-revocation-replay/raw/main/paper/selective-revocation-and-replay.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18967061.svg)](https://doi.org/10.5281/zenodo.18967061)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-USENIX%20source-1D4ED8?style=flat-square&logo=latex&logoColor=white)](https://github.com/aliuyar1234/selective-revocation-replay/tree/main/paper/usenix_security26)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Pytest](https://img.shields.io/badge/Pytest-25%20passed-2ea44f?style=flat-square&logo=pytest&logoColor=white)](https://github.com/aliuyar1234/selective-revocation-replay/tree/main/tests)
[![Scope](https://img.shields.io/badge/Scope-Post--Compromise%20Recovery-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *Selective Revocation and Replay: Post-Compromise Recovery of Explicit Persisted State in Memory-Augmented LLM Agents*

This repository contains the code, paper source, and frozen evaluation outputs for a methods paper on post-compromise recovery in memory-augmented LLM agents. It studies whether persisted descendants of a compromised root can be revoked and only dirty state-writing events replayed, so that explicit persisted state recovers without discarding benign history.

## Abstract

This project studies post-compromise recovery for explicit persisted state in memory-augmented LLM agents after persistent indirect prompt injection has already been written into durable state. The core mechanism revokes persisted descendants of suspicious roots and replays only dirty state-writing events when replay is sound, with fallback to coarse rollback otherwise. In a frozen deterministic matrix spanning 8 task chains, 2 memory architectures, 2 attack variants, 5 methods, and 1 ablation, selective replay is the only method that simultaneously achieves zero residual attack success and non-zero benign retention on explicit attacks. On retrieval memory, it matches rollback's retained state while reducing post-detection cost from 17 to 9 extra LLM calls. A focused live confirmation on `qwen2.5:14b` shows the same recovery wedge on repeated explicit retrieval-memory runs.

## Main Finding

Selective replay is the only evaluated method that combines zero residual explicit attack success with non-zero retained benign state, at lower post-detection cost than coarse rollback on retrieval memory.

- In the deterministic artifact, selective replay is the only method that combines zero residual explicit attack success with non-zero retained benign state.
- On retrieval memory, selective replay matches rollback's retained state while reducing post-detection cost from 17 to 9 extra LLM calls.
- In the focused live confirmation, all six unrecovered reruns keep the attack active while all six selective-replay reruns restore clean follow-up behavior and retained remembered facts.
- The strongest end-to-end live case succeeds with only four revocations and two replayed writer events.

## Contributions

1. A formal definition of *post-compromise recovery* for explicit persisted state in memory-augmented LLM agents, targeting durable state written by persistent indirect prompt injection.
2. A recovery mechanism that revokes persisted descendants of suspicious roots and replays only dirty state-writing events when replay is sound, with coarse rollback as a safe fallback.
3. A frozen deterministic evaluation matrix over 8 task chains, 2 memory architectures, 2 attack variants, 5 methods, and 1 ablation, with append-only provenance logging.
4. A cost analysis showing that on retrieval memory selective replay matches rollback's retained benign state at 9 extra LLM calls instead of 17.
5. A focused live confirmation on `qwen2.5:14b` that reproduces the recovery wedge on repeated explicit retrieval-memory runs.

## Scope

This project studies post-compromise recovery in a narrow but fully reproducible setting.

- one single-agent runtime
- text-only local files workspace
- two tools: `search_docs` and `read_doc`
- two persisted-state architectures: episodic retrieval memory and rolling summaries
- append-only provenance logging
- selective descendant revocation
- writer-only replay with coarse rollback fallback
- a frozen evaluation matrix over 8 chains, 2 architectures, 2 attack variants, 5 methods, and 1 ablation

The contribution is not breadth. It is a deterministic, artifact-backed recovery mechanism and an honest head-to-head against coarse rollback and unrecovered baselines.

## Paper

- Compiled PDF: [`paper/selective-revocation-and-replay.pdf`](paper/selective-revocation-and-replay.pdf)
- LaTeX source: [`paper/usenix_security26/main.tex`](paper/usenix_security26/main.tex)
- Frozen result tables: [`results/tables/`](results/tables/)
- Frozen result figures: [`results/figures/`](results/figures/)
- Artifact runbook: [`artifact/runbook.md`](artifact/runbook.md)

Build the paper with:

```powershell
powershell -ExecutionPolicy Bypass -File paper/usenix_security26/build.ps1
```

This refreshes the public PDF at `paper/selective-revocation-and-replay.pdf`.

## Repository Layout

- [`configs/`](configs/) — runtime, evaluation, and backend settings
- [`prompts/`](prompts/) — prompt templates
- [`src/`](src/) — runtime, recovery, scoring, and plotting modules
- [`scripts/`](scripts/) — command-line entry points
- [`data/workspace/`](data/workspace/) — chain workspaces and source files
- [`results/`](results/) — frozen aggregate outputs and paper-ready artifacts
- [`paper/`](paper/) — public compiled PDF and standalone LaTeX source package
- [`artifact/`](artifact/) — compact reproducibility package
- [`appendix/`](appendix/) — supporting appendix material and chain catalog
- [`vendor/`](vendor/) — bundled fallback dependencies for the optional local Qwen backend

Directories that hold bulky local run outputs, transient paper build files, local preview renders, handoff packages, or archived working material are intentionally ignored by Git.

## Reproducibility

The default backend is `heuristic_artifact`, which makes the frozen evaluation deterministic and fully reproducible from the repository.

Main entry points:

- `python scripts/build_workspace.py --chain c01_travel --attack explicit`
- `python scripts/run_base_history.py --chain c01_travel --architecture retrieval --attack explicit`
- `python scripts/apply_method.py --base-history <run_id> --method selective_replay`
- `python scripts/run_eval_matrix.py`
- `python scripts/run_live_confirmation_batch.py`
- `python scripts/make_tables.py`
- `python scripts/make_figures.py`
- `python scripts/build_paper_artifacts.py`

Quick start:

1. Install the package and developer dependencies.
2. Materialize a workspace or let the run scripts do it automatically.
3. Run a base history through S1-S2.
4. Apply one or more recovery methods to the same compromised base history.
5. Run the full frozen matrix with the deterministic backend.
6. Generate tables and figures.
7. Build the paper with `powershell -ExecutionPolicy Bypass -File paper/usenix_security26/build.ps1`.

Verification:

- `python -m pytest -q`
- `python scripts/make_tables.py`
- `python scripts/make_figures.py`

The frozen aggregate outputs needed to inspect the current paper claims are already included in the repository.

Optional model-backed configurations are available through `configs/models.yaml`:

- `qwen2_5_14b_ollama`
- `qwen2_5_32b_ollama`
- `qwen2_5_7b_instruct`
- `qwen3_5_27b_instruct`

For local Qwen checkpoints, set a local model path in the corresponding config or provide `LOCAL_QWEN_MODEL_PATH` in your environment before running the backend. If `transformers` and its runtime dependencies are already installed in your environment, the local-Qwen path uses them directly. The vendored `vendor/` tree is a fallback for that optional path, not a requirement for the deterministic artifact.

## Citation

```bibtex
@misc{uyar2026selectiverevocationreplay,
  author = {Uyar, Ali},
  title  = {Selective Revocation and Replay: Post-Compromise Recovery of Explicit Persisted State in Memory-Augmented {LLM} Agents},
  year   = {2026},
  doi    = {10.5281/zenodo.18967061},
  url    = {https://doi.org/10.5281/zenodo.18967061},
  note   = {Independent research}
}
```

Machine-readable citation metadata is also available in [`CITATION.cff`](CITATION.cff).
