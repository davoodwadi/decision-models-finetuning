#!/usr/bin/env bash
# Push code/ to Nibi for the 9B paper runs: every file git can see (committed, modified, untracked; ignored files
# excluded) plus all data parquets outside raw/. Laptop -> Nibi only. Nothing is deleted on Nibi.
# --update skips files that are newer on Nibi (e.g. a notebook run there keeps its outputs).
# Nibi has ~/projects/def-yuma -> /project/6028335 (no ~/links as on Rorqual); ~/links/projects -> ~/projects was
# added on Nibi so the sbatch script and the kev-cu13 kernel find the same paths on both clusters.
# Extra arguments go to rsync, e.g. `./sync-nibi.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="nibi"
REMOTE_DIR="projects/def-yuma/wadi/decision-models-finetuning/code"

cd "$(dirname "$0")/code"

{
    git ls-files --cached --others --exclude-standard
    find data -name '*.parquet' -not -path '*/raw/*'
} | rsync -av --update --ignore-missing-args --files-from=- \
    --rsync-path="mkdir -p ${REMOTE_DIR} && rsync" \
    ./ "${REMOTE_HOST}:${REMOTE_DIR}/" "$@"
