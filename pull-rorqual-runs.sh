#!/usr/bin/env bash
# Pull the paper notebooks' run folders from Rorqual into a separate local folder for the run viewer
# (source "rorqual" in code/run-viewer/.env.local). Rorqual -> laptop only; nothing is deleted locally.
# From checkpoints/ only best.pt is copied (LoRA + pointer head at the best validation, for local evaluation);
# last.pt (resume state) and its temporary files stay on Rorqual.
# Extra arguments go to rsync, e.g. `./pull-rorqual-runs.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="rorqual"
REMOTE_RUNS="links/projects/def-yuma/wadi/decision-models-finetuning/code/train-notebooks/beta-training/ms_paper_beta_loss/runs"
LOCAL_RUNS="$(dirname "$0")/code/train-notebooks/beta-training/ms_paper_beta_loss/runs-rorqual"

mkdir -p "${LOCAL_RUNS}"
rsync -av --include 'checkpoints/' --include 'checkpoints/best.pt' --exclude 'checkpoints/*' "${REMOTE_HOST}:${REMOTE_RUNS}/" "${LOCAL_RUNS}/" "$@"
