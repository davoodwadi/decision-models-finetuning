# ms_paper_beta_loss: the paper's training regimes

What each row trains on. "—" means that data is not used. Each row's design (datasets, λ, α, epochs) is set in its own
`train_paper_regime(...)` cell in the training notebook; `PEAK_LR` is in the settings cell.

| Row | Wu choices (18,591) | Wu table (8 fitted OLS slopes) | Berke choices (36,297) | Epochs | Steps | Question |
|---|---|---|---|---:|---:|---|
| 1 | — | — | — | 0 | 0 | How good is the untrained model? |
| 2 | — | beta loss + KL (α = 0.02) | — | 3 | 1,395 | Can a table alone improve the model? |
| 2 ablation | — | beta loss, no KL (α = 0) | — | 3 | 1,395 | Does the KL term help? |
| 2b | CE | — | — | 3 | 1,395 | Table vs the choices it was fitted from |
| 3 | — | — | CE | 1 | 910 | Own microdata only (baseline) |
| 4 | — | beta loss + KL (α = 0.02) | CE | 1 | 1,375 | Own microdata + another study's table |
| 5 | CE | — | CE | 1 | 1,375 | Microdata for both studies (reference) |

- **Beta loss:** population beta loss with λ = 1, against the 8 Wu slopes. Wu's choices never enter it.
- **KL:** KL(p0 ‖ p) toward the untrained model on Wu's tasks, weight α.
- **CE:** cross-entropy on respondents' choices.
- Row 2 vs Row 4: the same Wu table term; Row 4 adds Berke's choices.
- Row 2 vs Row 2b: the same Wu tasks and settings; the table vs the choices.
- Row 3 vs Row 4: the same Berke data; Row 4 adds the Wu table.

## Shared settings

- Peak LR, the same for every row: `1e-5` on Kev 800M, `5e-6` on Kev 4B (`PEAK_LR` in the training notebook).
- S = 40 tasks per step, OneCycle, weight decay 0.01, gradient clip 1.0, seed 42.
- Validation: Kriner 2026 and Grummon 2025, never trained on. best.pt is picked by `kriner2026/beta_mse`.

## Folders and files

| Row | Run folder (`runs/kev-0.8b/`, `runs/kev-4b/`) |
|---|---|
| 1 | step 0 of any run |
| 2 | `row2_wu_table_anchored` |
| 2 ablation | `row2_ablation_wu_table_no_kl` |
| 2b | `row2b_wu_ce_oracle` |
| 3 | `row3_berke_ce_only` |
| 4 | `row4_hybrid_berke_ce_wu_table` |
| 5 | `row5_full_berke_wu_ce_oracle` |

- `00-validate-kl-anchor-800m.ipynb`: pre-flight sweep on Kev 800M comparing pure beta loss (`α = 0`) against KL anchoring (`α ∈ {0.02, 0.1, 0.5}`) on Wu's table, saved separately under `runs/kl_anchor_800m/`.
- `00-validate-kl-anchor-9b.ipynb`: the same sweep on Kev-9B (`jaredpalmer/kev-9b@db029f08`) at `PEAK_LR = 5e-6` (the Kev-4B rate), with the micro-chunk search on (`max_chunk = 41`), saved under `runs/kl_anchor_9b/`.
- `01-train-paper-800m.ipynb`: trains every row on Kev 800M.
- `02-train-paper-4b-<row>.ipynb` (`row2`, `row2-ablation`, `row2b`, `row3`, `row4`, `row4-ablation`, `row5`): one regime each on Kev-4B (`jaredpalmer/kev-4b@6cfce5c2`), with the same settings as `02-train-paper-9b.ipynb`, so each regime runs as its own job; saved under `runs/kev-4b/`.
- `02-train-paper-9b.ipynb`: every row on Kev-9B (`jaredpalmer/kev-9b@db029f08`) at `PEAK_LR = 5e-6`, gradient checkpointing off, the micro-chunk search on (`max_chunk = 41`), `α = 0.005`, 3 epochs for every row, and Row 4 split into `row4_hybrid_berke_ce_wu_table_anchored` (`α = 0.005`) and `row4_hybrid_berke_ce_wu_table_no_kl` (`α = 0`); saved under `runs/kev-9b/`.
- `slurm/*.sbatch`: Rorqual batch jobs that execute a notebook top to bottom on one H100 and save the executed copy as `<notebook>.rorqual.ipynb`.
- `slurm/train-notebook.sbatch <notebook>`: the same for any one notebook given as the argument (used for the one-regime 4B notebooks; the submit commands are in the script's header).
- `03-analysis-paper.ipynb`: paper tables and figures from the saved runs (CPU only).
- `paper_beta/`: loss, trainer, and run summaries (`regimes.py`). Each row is one `train_paper_regime(...)` cell in the training notebook (run folder name, paper row code, datasets, λ, α, epochs); the summary finds the saved runs on disk and orders them by row code. To add a row, copy a cell and change its values.
