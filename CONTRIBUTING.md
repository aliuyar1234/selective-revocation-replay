# Contributing

Thanks for your interest in improving this research artifact.

## Scope

This repository is intentionally narrow and paper-aligned. The most useful contributions are:

- correctness fixes in the runtime, recovery logic, scoring, or figure/table generation
- reproducibility improvements
- documentation improvements that help readers rebuild the artifact
- narrowly scoped extensions that do not blur the paper's main claim

## Before opening large changes

Please avoid broadening the repository into a generic prompt-injection benchmark or a large framework. The paper and artifact are designed around one narrow question: post-compromise recovery for explicit persisted state in memory-augmented LLM agents.

## Suggested workflow

1. Read [README.md](README.md) and [artifact/runbook.md](artifact/runbook.md).
2. Keep the public artifact surface clean and reader-oriented.
3. Run the fastest relevant verification for your change:
   - `python -m pytest -q`
   - `python scripts/build_paper_artifacts.py`
   - `powershell -ExecutionPolicy Bypass -File paper/usenix_security26/build.ps1`

## Citation

If you use this repository in academic work, please use the GitHub citation metadata or the compiled paper at `paper/selective-revocation-and-replay.pdf`.
