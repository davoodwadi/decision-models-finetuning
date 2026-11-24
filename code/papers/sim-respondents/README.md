# sim-respondents: code for "Training decision models for simulating human respondents"

The manuscript is `manuscript/sim-respondents/manuscript-training.qmd`. This folder holds everything that produces its
numbers, tables and figures.

## What is shared and what is this paper's own

| From `code/` (shared by every paper, never copied) | This paper's own copy (`src/`) |
|---|---|
| `kev/` (Kev model library) | `training/` |
| `models/` (`KevPointerModel`) | `beta_ols/` |
| | `beta_train/` |
| | `paper_beta/` (regime trainer and run summaries) |
| | `microdata_analysis.py` (loads saved runs and `test_best/` results for `assets/`) |
| | `data/`: this paper's copy of the loader code (`base.py`, `persona.py`, `marginals.py`, `grummon2023.py`, `grummon2025.py`) and of the dataset folders it uses, raw files included (berke2023, wu2025, Kriner2026, kantorowicz2026, covid, magistro2025, nutrition) |

Every notebook finds `code/` by walking up to the first folder with `kev/`, `data/` and `models/`, then puts `src/`
ahead of `code/` on `sys.path`, so `training`, `beta_ols`, `beta_train`, `paper_beta` and `data` load from `src/` and
`kev` and `models` load from `code/`. Environment: the repository's `.venv-cu13` (locally) and `~/envs/kev-cu13` (on
the clusters).

Data and cache: `src/data/` is this paper's own copy, so a change to personas or datasets in another paper never changes
this one. Its default cache, and every notebook's `CACHE_DIR`, is this paper's `.cache/` (not `code/.cache/`). The
persona-record cache (`build_persona_dataset`) is keyed on row count, columns and question IDs, not row values, so clear
`.cache/` after re-exporting a dataset. `code/data/` is a frozen archive read only by the history folders.

## Folders

| Folder | Contents |
|---|---|
| `src/` | The copied modules above |
| `train/train-800m/`, `train/train-4b/`, `train/train-9b/` | One notebook per regime and model (Rows 2b, 3, 5; `02-train-paper-4b-row5-continue.ipynb` continues the Kev-4B Row 5 run) |
| `analysis/test-best/` | Best checkpoint of every run on the held-out test datasets; writes each run's `test_best/` (details in its README) |
| `analysis/llm-baseline/` | Off-the-shelf LLMs vs the untrained model on Vaccines; writes `results_summary.csv` (details in its README) |
| `assets/` | One notebook per manuscript figure or table, named by its Quarto label (details in its README) |
| `runs-rorqual/`, `runs-nibi/` | Run folders pulled from each cluster (run files and `checkpoints/best.pt`, plus `test_best/`) |
| `slurm/` | Batch jobs for training and test-best on Rorqual and Nibi |
| `sync/` | rsync scripts between this folder and the clusters |

`manuscript/sim-respondents/manuscript-training-assets` is a symlink to `assets/`.

## Rows

| Row | Trains on | Epochs | Steps | Run folder |
|---|---|---:|---:|---|
| 1 | nothing (untrained model) | 0 | 0 | `row1_base_untrained` (holds only `test_best/`) |
| 2b | Wu choices (CE) | 3 | 1,395 | `row2b_wu_ce_oracle` |
| 3 | Berke choices (CE) | 3 | 2,730 | `row3_berke_ce_only` |
| 5 | Wu and Berke choices (CE) | 3 | 4,125 | `row5_full_berke_wu_ce_oracle` |

Rows 2, 4 and their ablations (beta loss) belong to the beta-loss paper and are not here.

| Model | Base checkpoint | Run folders |
|---|---|---|
| Kev-0.8B | `jaredpalmer/kev-0.8b@bf75a6a8` | `runs-rorqual/kev-0.8b/` |
| Kev-4B | `jaredpalmer/kev-4b@6cfce5c2` | `runs-rorqual/kev-4b/` |
| Kev-9B | `jaredpalmer/kev-9b@db029f08` | `runs-rorqual/kev-9b/` (Row 1), `runs-nibi/kev-9b/` (Rows 2b, 3, 5) |

Shared settings (every row and model): peak LR `5e-6` (`PEAK_LR`), S = 40 tasks per step, OneCycle, weight decay 0.01,
gradient clip 1.0, LoRA rank 64, seed 42. Validation on Kriner 2026 and Grummon 2025, never trained on; `best.pt` is
picked by `kriner2026/beta_mse`. Test datasets: Grummon 2025, COVID (Kreps 2020), Kantorowicz 2026, Magistro 2025.

## Commands

Run from this folder (`code/papers/sim-respondents/`).

Push code to a cluster (repository root scripts, they push all of `code/`): `../../../sync-rorqual.sh`,
`../../../sync-nibi.sh`.

Train one regime, or every notebook of a folder, on a cluster:

```bash
sbatch --job-name=02-train-paper-4b-row3 slurm/train-notebook.sbatch train/train-4b/02-train-paper-4b-row3.ipynb
```

```bash
slurm/submit-folder.sh train/train-4b
```

New training runs are written to `runs/<model>/` on the cluster. Pull them with `sync/pull-rorqual-runs.sh` or
`sync/pull-nibi-runs.sh` (into `runs-rorqual/` and `runs-nibi/`). The existing runs were trained before this folder
existed; on the clusters they are still under `code/train-notebooks/beta-training/ms_paper_beta_loss/runs/`.

Test best checkpoints on Nibi: `sync/push-nibi-test-runs.sh`, then

```bash
slurm/submit-test-best.sh analysis/test-best/04-test-best-9b.ipynb kev-9b
```

and `sync/pull-nibi-test-best.sh` to bring the `test_best/` results back.

Rebuild the manuscript: run the notebooks in `assets/` (CPU only), then `quarto render manuscript-training.qmd` in
`manuscript/sim-respondents/`.
