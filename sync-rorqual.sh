#!/usr/bin/env bash
# Push code/ to Rorqual: every file git can see (committed, modified, untracked; ignored files excluded)
# plus all data parquets outside raw/. Laptop -> Rorqual only. Nothing is deleted on Rorqual.
# --update skips files that are newer on Rorqual (e.g. a notebook run there keeps its outputs).
# Extra arguments go to rsync, e.g. `./sync-rorqual.sh -n` for a dry run.
set -euo pipefail

REMOTE_HOST="rorqual"
REMOTE_DIR="links/projects/def-yuma/wadi/decision-models-finetuning/code"

cd "$(dirname "$0")/code"

{
    git ls-files --cached --others --exclude-standard
    find data -name '*.parquet' -not -path '*/raw/*'
} | rsync -av --update --ignore-missing-args --files-from=- \
    --rsync-path="mkdir -p ${REMOTE_DIR} && rsync" \
    ./ "${REMOTE_HOST}:${REMOTE_DIR}/" "$@"
