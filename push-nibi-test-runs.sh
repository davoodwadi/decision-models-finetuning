#!/usr/bin/env bash
# Push the local run folders that the test-best notebooks read (test-best/04-test-best-*.ipynb) to Nibi, so they can
# run there in JupyterLab or as a batch job with the same paths as locally:
#   runs-rorqual/kev-0.8b, kev-4b, kev-9b   (trained on Rorqual, pulled by pull-rorqual-runs.sh)
#   runs-nibi/kev-9b                        (trained on Nibi, pulled by pull-nibi-runs.sh)
# Each run folder goes as it is locally: run files, checkpoints/best.pt, and test_best/ (so finished test datasets are
# reused and only missing ones are evaluated). Row 1's test_best/ (row1_base_untrained) goes too. kl_anchor_* folders
# are not pushed (the test-best notebooks do not read them).
# They land in ms_paper_beta_loss/runs-rorqual and ms_paper_beta_loss/runs-nibi on Nibi: real copies, separate from
# Nibi's own runs/ (the training output), so test results never end up in runs/ and pull-nibi-runs.sh never brings
# them back over local files.
# Laptop -> Nibi only. Nothing is deleted on Nibi. --update skips files that are newer on Nibi (e.g. test_best/ written
# there after a previous push).
# Push the code first with ./sync-nibi.sh (notebooks, beta_ols, data parquets). Results come back with
# ./pull-nibi-test-best.sh.
# Extra arguments go to rsync, e.g. `./push-nibi-test-runs.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="nibi"
REMOTE_PAPER="projects/def-yuma/wadi/decision-models-finetuning/code/train-notebooks/beta-training/ms_paper_beta_loss"
LOCAL_PAPER="$(dirname "$0")/code/train-notebooks/beta-training/ms_paper_beta_loss"

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
