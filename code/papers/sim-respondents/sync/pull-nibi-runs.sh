#!/usr/bin/env bash
# Pull this paper's run folders from Nibi (code/papers/sim-respondents/runs, written by the train/ notebooks) into
# runs-nibi/ in the paper folder. Nibi -> laptop only; nothing is deleted locally.
# From checkpoints/ only best.pt is copied (LoRA + pointer head at the best validation, for local evaluation);
# last.pt (resume state) and its temporary files stay on Nibi.
# Run from anywhere: code/papers/sim-respondents/sync/pull-nibi-runs.sh
# Extra arguments go to rsync, e.g. `sync/pull-nibi-runs.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="nibi"
REMOTE_RUNS="projects/def-yuma/wadi/decision-models-finetuning/code/papers/sim-respondents/runs"
LOCAL_RUNS="$(dirname "$0")/../runs-nibi"

mkdir -p "${LOCAL_RUNS}"
rsync -av --include 'checkpoints/' --include 'checkpoints/best.pt' --exclude 'checkpoints/*' "${REMOTE_HOST}:${REMOTE_RUNS}/" "${LOCAL_RUNS}/" "$@"
