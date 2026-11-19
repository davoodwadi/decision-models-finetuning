#!/usr/bin/env bash
# Pull the paper notebooks' run folders from Nibi into a separate local folder for the run viewer and local evaluation
# (the 9B rows trained on Nibi; Rows 2 and 2 ablation of 9B came from Rorqual and are in runs-rorqual).
# Nibi -> laptop only; nothing is deleted locally.
# From checkpoints/ only best.pt is copied (LoRA + pointer head at the best validation, for local evaluation);
# last.pt (resume state) and its temporary files stay on Nibi.
# Extra arguments go to rsync, e.g. `./pull-nibi-runs.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="nibi"
REMOTE_RUNS="projects/def-yuma/wadi/decision-models-finetuning/code/train-notebooks/beta-training/ms_paper_beta_loss/runs"
LOCAL_RUNS="$(dirname "$0")/code/train-notebooks/beta-training/ms_paper_beta_loss/runs-nibi"

mkdir -p "${LOCAL_RUNS}"
rsync -av --include 'checkpoints/' --include 'checkpoints/best.pt' --exclude 'checkpoints/*' "${REMOTE_HOST}:${REMOTE_RUNS}/" "${LOCAL_RUNS}/" "$@"
