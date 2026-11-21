#!/usr/bin/env bash
# Pull the test-best results computed on Nibi back into the local run folders: only each run's test_best/
# (metrics.csv, betas.csv, test.json) from ms_paper_beta_loss/runs-rorqual and ms_paper_beta_loss/runs-nibi on Nibi,
# the copies pushed by push-nibi-test-runs.sh. Nothing else is pulled (best.pt and run files are already local).
# Nibi -> laptop only. Nothing is deleted locally. --update skips files that are newer locally.
# Extra arguments go to rsync, e.g. `./pull-nibi-test-best.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="nibi"
REMOTE_PAPER="projects/def-yuma/wadi/decision-models-finetuning/code/train-notebooks/beta-training/ms_paper_beta_loss"
LOCAL_PAPER="$(dirname "$0")/code/train-notebooks/beta-training/ms_paper_beta_loss"

for runs_folder in runs-rorqual runs-nibi; do
    rsync -av --update --prune-empty-dirs \
        --include '/kev-*/' --include '/kev-*/*/' --include '/kev-*/*/test_best/***' --exclude '*' \
        "${REMOTE_HOST}:${REMOTE_PAPER}/${runs_folder}/" "${LOCAL_PAPER}/${runs_folder}/" "$@"
done
