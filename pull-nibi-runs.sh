#!/usr/bin/env bash
# Pull the paper notebooks' run folders from Nibi into a separate local folder for the run viewer and local evaluation
# (the 9B rows trained on Nibi; Rows 2 and 2 ablation of 9B came from Rorqual and are in runs-rorqual).
# Nibi -> laptop only; nothing is deleted locally.
# From checkpoints/ only best.pt is copied (LoRA + pointer head at the best validation, for local evaluation);
# last.pt (resume state) and its temporary files stay on Nibi.
# Extra arguments go to rsync, e.g. `./pull-nibi-runs.sh -n` for a dry run.
# Rows 1, 2b, 3 and 5 belong to the sim-respondents paper (code/papers/sim-respondents) and are excluded here,
# so a pull never copies them back into ms_paper_beta_loss.
set -euo pipefail

REMOTE_HOST="nibi"
REMOTE_RUNS="projects/def-yuma/wadi/decision-models-finetuning/code/train-notebooks/beta-training/ms_paper_beta_loss/runs"
LOCAL_RUNS="$(dirname "$0")/code/train-notebooks/beta-training/ms_paper_beta_loss/runs-nibi"

mkdir -p "${LOCAL_RUNS}"
rsync -av --exclude '/kev-*/row1_base_untrained/' --exclude '/kev-*/row2b_wu_ce_oracle/' --exclude '/kev-*/row3_berke_ce_only/' --exclude '/kev-*/row5_full_berke_wu_ce_oracle/' \
    --include 'checkpoints/' --include 'checkpoints/best.pt' --exclude 'checkpoints/*' "${REMOTE_HOST}:${REMOTE_RUNS}/" "${LOCAL_RUNS}/" "$@"
