#!/usr/bin/env bash
# Pull this paper's run folders from Rorqual (code/papers/beta-loss/runs, written by the train/ notebooks) into
# runs-rorqual/ in the paper folder. Rorqual -> laptop only; nothing is deleted locally.
# From checkpoints/ only best.pt is copied (LoRA + pointer head at the best validation, for local evaluation);
# last.pt (resume state) and its temporary files stay on Rorqual.
# Run from anywhere: code/papers/beta-loss/sync/pull-rorqual-runs.sh
# Extra arguments go to rsync, e.g. `sync/pull-rorqual-runs.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="rorqual"
REMOTE_RUNS="links/projects/def-yuma/wadi/decision-models-finetuning/code/papers/beta-loss/runs"
LOCAL_RUNS="$(dirname "$0")/../runs-rorqual"

mkdir -p "${LOCAL_RUNS}"
rsync -av --include 'checkpoints/' --include 'checkpoints/best.pt' --exclude 'checkpoints/*' "${REMOTE_HOST}:${REMOTE_RUNS}/" "${LOCAL_RUNS}/" "$@"
