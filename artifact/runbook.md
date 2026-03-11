# Runbook

## 1. Materialize one workspace

```powershell
python scripts/build_workspace.py --chain c01_travel --attack explicit
```

## 2. Create one attacked base history

```powershell
python scripts/run_base_history.py --chain c01_travel --architecture retrieval --attack explicit
```

This writes one attacked S1-S2 history under `results/base_histories/` in a locally generated output directory.

## 3. Fork and evaluate one recovery method

```powershell
python scripts/apply_method.py --base-history <base_history_run_id> --method selective_replay
```

Supported methods:

- `no_recovery`
- `root_delete`
- `full_reset`
- `coarse_rollback`
- `selective_replay`
- `revoke_no_replay`

## 4. Run the full frozen matrix

```powershell
python scripts/run_eval_matrix.py
```

Expected outcome:

- 32 attacked base histories
- 192 attacked method runs
- `results/raw/eval_results.jsonl`
- `results/raw/eval_results.csv`

## 5. Generate tables and figures

```powershell
python scripts/make_tables.py
python scripts/make_figures.py
```

Expected outputs:

- `results/tables/tbl1_main_results.csv`
- `results/tables/tbl2_attack_breakdown.csv`
- `results/tables/tbl3_ablation.csv`
- `results/tables/tbl4_fallback_and_cost.csv`
- `results/tables/tbl5_live_confirmation.csv` after running the live confirmation batch
- `results/figures/fig1_motivating_example.pdf`
- `results/figures/fig2_system_overview.pdf`
- `results/figures/fig3_recovery_paths.pdf`
- `results/figures/fig4_cost_vs_retention.pdf`
- `results/figures/fig5_live_case_study.pdf` after running the live confirmation batch

## 5a. Generate the focused live-model confirmation section

```powershell
$env:SRR_ACTIVE_MODEL="qwen2_5_14b_ollama"
python scripts/run_live_confirmation_batch.py
```

Expected outputs:

- `results/model_pilots/live_confirmation_latest.json`
- `results/model_pilots/live_confirmation_latest.csv`
- `results/model_pilots/live_confirmation_latest.md`

This default batch intentionally targets the strongest live-model settings for the paper:

- `c01_travel`, retrieval, explicit
- `c05_training`, retrieval, explicit

## 6. Assemble paper and artifact assets

```powershell
python scripts/build_paper_artifacts.py
```

This copies generated tables and figures into `paper/` and writes an artifact manifest with repository-relative paths.

## Public repo note

The public GitHub-facing repository keeps aggregate outputs, paper assets, and the focused live-confirmation summary. Large per-run directories are generated locally and ignored by default.

## Sanity checks

- `no_recovery` should retain explicit attack success on at least part of the matrix
- `full_reset` should remove retained benign state and usually fail S4
- `selective_replay` should preserve more S4 retention than reset-like baselines
- summary-memory selective replay may fall back to rollback under the conservative replay-safe rule
