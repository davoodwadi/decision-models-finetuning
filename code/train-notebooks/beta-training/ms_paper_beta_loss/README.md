# ms_paper_beta_loss: the paper's training regimes

What each row trains on. "—" means that data is not used. The design of every row (datasets, λ, α, epochs, LR) is set
in `REGIME_DESIGNS` and `PEAK_LR` in the training notebook's settings cell.

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

- `01-train-paper-800m.ipynb`, `02-train-paper-4b.ipynb`: train every row.
- `03-analysis-paper.ipynb`: paper tables and figures from the saved runs (CPU only).
- `paper_beta/`: loss, trainer, and the rows' labels (`regimes.py`, labels only).
