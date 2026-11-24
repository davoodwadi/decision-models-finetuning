# beta-loss: code for "Training decision models from choice microdata and published regression tables"

The manuscript folder is `manuscript/beta-loss/`. Its `manuscript-beta-loss.qmd` is an older draft kept as starting text
(its figures still come from `code/analysis/wang2026/` and `code/analysis/grummon2025/`); the argument is drafted in
`manuscript/beta-loss/paper-positioning-kl-vs-ols-beta.md`. This folder holds the code and runs behind the paper.

## What is shared and what is this paper's own

| From `code/` (shared by every paper, never copied) | This paper's own copy (`src/`) |
|---|---|
| `kev/` (Kev model library) | `training/` |
| `models/` (`KevPointerModel`) | `beta_ols/` (dataset OLS modules and the beta loss) |
| | `beta_train/` |
| | `paper_beta/` (regime trainer and run summaries) |
| | `data/`: this paper's copy of the loader code (`base.py`, `persona.py`, `marginals.py`, `grummon2023.py`, `grummon2025.py`) and of the dataset folders it uses, raw files included (berke2023, wu2025, Kriner2026, kantorowicz2026, covid, magistro2025, nutrition) |

Every notebook finds `code/` by walking up to the first folder with `kev/`, `data/` and `models/`, then puts `src/`
ahead of `code/` on `sys.path`, so `data` and this paper's modules load from `src/`. Environment: the repository's `.venv-cu13` (locally) and `~/envs/kev-cu13` (on the
clusters).

Data and cache: `src/data/` is this paper's own copy, so a change to personas or datasets in another paper never changes
this one. Its default cache, and every notebook's `CACHE_DIR`, is this paper's `.cache/` (not `code/.cache/`). The
persona-record cache (`build_persona_dataset`) is keyed on row count, columns and question IDs, not row values, so clear
`.cache/` after re-exporting a dataset. `code/data/` is a frozen archive read only by the history folders. The
`exploration/` notebooks keep reading `code/data/`.

## Folders

| Folder | Contents |
|---|---|
| `src/` | The copied modules above |
| `train/02-train-paper-9b.ipynb` | Rows 2, 2 ablation, 4, 4 ablation on Kev-9B in one notebook (it trained Rows 2 and 2 ablation) |
| `train/train-800m/`, `train/train-4b/`, `train/train-9b/` | One notebook per regime and model (Rows 2, 2 ablation, 4, 4 ablation; 9B: Rows 4, 4 ablation) |
| `train/kl-anchor/` | KL-anchor α sweeps on Wu's table (Kev-0.8B: `00-validate-kl-anchor-800m`; Kev-9B: `00-validate-kl-anchor-9b` and one notebook per α) |
| `analysis/03-analysis-paper.ipynb` | Paper tables and figures from all rows at `best.pt` (CPU only; notes in `03-analysis-paper.md`) |
| `analysis/03-analysis-paper-last.ipynb` | The same analysis at the final training step instead of `best.pt` |
| `analysis/test-best/` | Best checkpoint of this paper's runs on the held-out test datasets (details in its README) |
| `assets/` | One notebook per manuscript figure or table, once the paper is written (empty) |
| `runs-rorqual/`, `runs-nibi/`, `runs/` | This paper's run folders (run files and `checkpoints/best.pt`, plus `test_best/`) |
| `slurm/` | Batch jobs for training, the KL-anchor sweep and test-best on Rorqual and Nibi |
| `sync/` | rsync scripts between this folder and the clusters |
| `exploration/` | The beta-training v1–v14 notebooks and their runs (35 GB), with their own frozen `src/`; history, not used by the paper |

`manuscript/beta-loss/assets` is a symlink to `assets/`.

## Rows

| Row | Trains on | λ | α | Steps | Run folder | Where |
|---|---|---:|---:|---:|---|---|
| 1 | nothing (untrained model) | — | — | 0 | `row1_base_untrained` | sim-respondents |
| 2 | Wu table (8 fitted OLS slopes), beta loss | 1 | 0.005 | 1,395 | `row2_wu_table_anchored` | this paper |
| 2 ablation | Wu table, beta loss, no KL anchor | 1 | 0 | 1,395 | `row2_ablation_wu_table_no_kl` | this paper |
| 2b | Wu choices (CE) | 0 | 0 | 1,395 | `row2b_wu_ce_oracle` | sim-respondents |
| 3 | Berke choices (CE) | 0 | 0 | 2,730 | `row3_berke_ce_only` | sim-respondents |
| 4 | Berke choices (CE) + Wu table (beta loss) | 1 | 0.005 | 4,125 | `row4_hybrid_berke_ce_wu_table_anchored` | this paper |
| 4 ablation | Berke choices (CE) + Wu table, no KL anchor | 1 | 0 | 4,125 | `row4_hybrid_berke_ce_wu_table_no_kl` | this paper |
| 5 | Wu and Berke choices (CE) | 0 | 0 | 4,125 | `row5_full_berke_wu_ce_oracle` | sim-respondents |

- **Beta loss:** population beta loss against Wu's 8 fitted OLS slopes; Wu's choices never enter it.
- **KL anchor:** KL(p0 ‖ p) toward the untrained model on Wu's tasks, weight α.
- Rows 1, 2b, 3 and 5 belong to the sim-respondents paper. This paper reads their runs and `test_best/` results from
  `code/papers/sim-respondents/runs-rorqual/` and `runs-nibi/` and never writes there; `analysis/test-best/` tests only
  this paper's rows.

| Model | Base checkpoint | This paper's run folders |
|---|---|---|
| Kev-0.8B | `jaredpalmer/kev-0.8b@bf75a6a8` | `runs-rorqual/kev-0.8b/` |
| Kev-4B | `jaredpalmer/kev-4b@6cfce5c2` | `runs-rorqual/kev-4b/` |
| Kev-9B | `jaredpalmer/kev-9b@db029f08` | `runs-rorqual/kev-9b/` (Rows 2, 2 ablation), `runs-nibi/kev-9b/` (Rows 4, 4 ablation) |

Shared settings (every row and model): peak LR `5e-6` (`PEAK_LR`), S = 40 tasks per step, 3 epochs, weight decay 0.01,
LoRA rank 64, seed 42. Validation on Kriner 2026 and Grummon 2025, never trained on; `best.pt` is picked by
`kriner2026/beta_mse`. Test datasets: Grummon 2025, COVID (Kreps 2020), Kantorowicz 2026, Magistro 2025.

KL-anchor sweeps (Wu table only, λ = 1, 3 epochs, 1,395 steps):

| Model | Peak LR | α values | Run folders |
|---|---|---|---|
| Kev-0.8B | `1e-5` | 0, 0.02, 0.1, 0.5 (0.5 interrupted) | `runs/kl_anchor_800m/` |
| Kev-0.8B | `1e-5` | 0.002, 0.005 | `runs-rorqual/kl_anchor_800m/` |
| Kev-9B | `5e-6` | 0, 0.002, 0.005, 0.01, 0.05 | `runs-rorqual/kl_anchor_9b/` |

## Commands

Run from this folder (`code/papers/beta-loss/`).

Push code to a cluster (repository root scripts, they push all of `code/`): `../../../sync-rorqual.sh`,
`../../../sync-nibi.sh`.

Train one regime, or every notebook of a folder, on a cluster:

```bash
sbatch --job-name=02-train-paper-4b-row4 slurm/train-notebook.sbatch train/train-4b/02-train-paper-4b-row4.ipynb
```

```bash
slurm/submit-folder.sh train/train-4b
```

New training runs are written to `runs/<model>/` on the cluster. Pull them with `sync/pull-rorqual-runs.sh` or
`sync/pull-nibi-runs.sh` (into `runs-rorqual/` and `runs-nibi/`). The existing runs were trained before this folder
existed; on the clusters they are still under `code/train-notebooks/beta-training/ms_paper_beta_loss/runs/`, and every
one of them is already in this folder.

Test best checkpoints on Nibi: `sync/push-nibi-test-runs.sh`, then

```bash
slurm/submit-test-best.sh analysis/test-best/04-test-best-9b.ipynb kev-9b
```

and `sync/pull-nibi-test-best.sh` to bring the `test_best/` results back.
