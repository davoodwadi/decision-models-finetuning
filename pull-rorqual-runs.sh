#!/usr/bin/env bash
# Pull the paper notebooks' run folders from Rorqual into a separate local folder for the run viewer
# (source "rorqual" in code/run-viewer/.env.local). Rorqual -> laptop only; nothing is deleted locally.
# checkpoints/ (model weights) is skipped: the viewer reads only run.json, steps.csv, val_history.csv and betas/.
# Extra arguments go to rsync, e.g. `./pull-rorqual-runs.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="rorqual"
REMOTE_RUNS="links/projects/def-yuma/wadi/decision-models-finetuning/code/train-notebooks/beta-training/ms_paper_beta_loss/runs"
LOCAL_RUNS="$(dirname "$0")/code/train-notebooks/beta-training/ms_paper_beta_loss/runs-rorqual"

mkdir -p "${LOCAL_RUNS}"
rsync -av --exclude 'checkpoints/' "${REMOTE_HOST}:${REMOTE_RUNS}/" "${LOCAL_RUNS}/" "$@"
