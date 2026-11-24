#!/bin/bash
# Submits one slurm/train-notebook.sbatch job per notebook in a folder of one-run notebooks, each job named after its
# notebook (log: <notebook>-<jobid>.out in the paper folder). Executed copies (*.rorqual.ipynb) are skipped.
# Extra arguments go to sbatch and override the script's defaults (48G, 16 h).
# Run on Rorqual from anywhere:
#   slurm/submit-folder.sh train/train-800m
#   slurm/submit-folder.sh train/train-4b
#   slurm/submit-folder.sh train/train-9b --mem=64G --time=48:00:00
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: slurm/submit-folder.sh <folder> [sbatch options...]" >&2
    exit 1
fi
FOLDER="${1%/}"
shift

# sbatch runs the job from the folder it is called in: the paper folder
cd "$(dirname "$0")/.."
if [ ! -d "$FOLDER" ]; then
    echo "No folder $FOLDER in $PWD" >&2
    exit 1
fi

submitted=0
for nb in "$FOLDER"/*.ipynb; do
    case "$nb" in
        *.rorqual.ipynb) continue ;;
    esac
    name="$(basename "$nb" .ipynb)"
    sbatch --job-name="$name" "$@" slurm/train-notebook.sbatch "$nb"
    submitted=$((submitted + 1))
done
if [ "$submitted" -eq 0 ]; then
    echo "No notebooks in $FOLDER" >&2
    exit 1
fi
echo "Submitted $submitted jobs from $FOLDER"
