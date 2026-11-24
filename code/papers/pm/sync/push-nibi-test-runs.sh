#!/usr/bin/env bash
# Push the local run folders that the test-best notebooks in analysis/ read to Nibi, so
# they can run there in JupyterLab or as a batch job with the same paths as locally:
#   runs-rorqual/kev-0.8b, kev-4b, kev-9b
#   runs-nibi/kev-9b
# Each run folder goes as it is locally: run files, checkpoints/best.pt, and test_best/ (so finished test datasets are
# reused and only missing ones are evaluated).
# They land in code/papers/pm/runs-rorqual and runs-nibi on Nibi: real copies, separate from Nibi's own
# runs/ (the training output), so test results never end up in runs/ and pull-nibi-runs.sh never brings them back
# over local files.
# Laptop -> Nibi only. Nothing is deleted on Nibi. --update skips files that are newer on Nibi (e.g. test_best/ written
# there after a previous push).
# Push the code first with ./sync-nibi.sh from the repository root. Results come back with sync/pull-nibi-test-best.sh.
# Run from anywhere: code/papers/pm/sync/push-nibi-test-runs.sh
# Extra arguments go to rsync, e.g. `sync/push-nibi-test-runs.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="nibi"
REMOTE_PAPER="projects/def-yuma/wadi/decision-models-finetuning/code/papers/pm"
LOCAL_PAPER="$(dirname "$0")/.."

# runs-rorqual: the three model folders
rsync -av --update \
    --include '/kev-0.8b/***' --include '/kev-4b/***' --include '/kev-9b/***' --exclude '*' \
    --rsync-path="mkdir -p ${REMOTE_PAPER}/runs-rorqual && rsync" \
    "${LOCAL_PAPER}/runs-rorqual/" "${REMOTE_HOST}:${REMOTE_PAPER}/runs-rorqual/" "$@"

# runs-nibi: the 9B folder
rsync -av --update \
    --include '/kev-9b/***' --exclude '*' \
    --rsync-path="mkdir -p ${REMOTE_PAPER}/runs-nibi && rsync" \
    "${LOCAL_PAPER}/runs-nibi/" "${REMOTE_HOST}:${REMOTE_PAPER}/runs-nibi/" "$@"
