#!/bin/bash
#SBATCH --account=def-yuma
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=%x-%j.out
# Runs one notebook top to bottom on one H100 and saves the executed copy next to it as <notebook>.rorqual.ipynb (the
# source notebook is not modified). The argument is the notebook's path relative to the paper folder (code/papers/beta-loss).
# Meant for the one-run notebooks (train/train-800m/, train/train-4b/, train/train-9b/, train/kl-anchor/), so each run is its own job.
# Submit from the paper folder, naming the job after the notebook so the log in that folder is
# <notebook>-<jobid>.out:
#   sbatch --job-name=02-train-paper-4b-row2 slurm/train-notebook.sbatch train/train-4b/02-train-paper-4b-row2.ipynb
# Every notebook of a folder at once (skips executed copies): slurm/submit-folder.sh <folder> [sbatch options]
# --time=16:00:00 is a guess for the longest regime (4,125 steps); override with sbatch --time=... . Resubmitting
# continues an unfinished regime from its run folder.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch --job-name=<name> slurm/train-notebook.sbatch <folder>/<notebook>.ipynb" >&2
    exit 1
fi
NOTEBOOK_PATH="$1"

module load python/3.13.2 arrow/25.0.0
source ~/envs/kev-cu13/bin/activate
export HF_HOME=$HOME/links/projects/def-yuma/wadi/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# lets the CUDA allocator grow segments instead of fragmenting (4B row5 ran out of memory with 4.78 GiB reserved but unused)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"
if [ ! -d src/paper_beta ]; then
    echo "Submit from the paper folder (no src/paper_beta/ in $PWD)" >&2
    exit 1
fi
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Notebook $NOTEBOOK_PATH not found in $PWD" >&2
    exit 1
fi
NOTEBOOK_NAME="$(basename "$NOTEBOOK_PATH" .ipynb)"

echo "job $SLURM_JOB_ID ($NOTEBOOK_PATH) on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
jupyter nbconvert --version

# --output is relative to the notebook's own folder, so the executed copy lands next to the notebook
jupyter nbconvert --to notebook --execute "$NOTEBOOK_PATH" \
    --output "${NOTEBOOK_NAME}.rorqual.ipynb" \
    --ExecutePreprocessor.kernel_name=kev-cu13 \
    --ExecutePreprocessor.timeout=-1

echo "finished $(date)"
