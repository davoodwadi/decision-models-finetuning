#!/bin/bash
#SBATCH --account=def-yuma
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=%x-%j.out
# Runs one notebook (name without .ipynb, given as the argument) top to bottom on one H100 and saves the executed copy
# as <notebook>.rorqual.ipynb (the source notebook is not modified). Meant for the one-regime notebooks
# (02-train-paper-4b-row*.ipynb), so each regime is its own job.
# Submit from the ms_paper_beta_loss folder, naming the job after the notebook so the log is <notebook>-<jobid>.out:
#   sbatch --job-name=02-train-paper-4b-row2 slurm/train-notebook.sbatch 02-train-paper-4b-row2
# All 4B regimes at once:
#   for nb in 02-train-paper-4b-row*.ipynb; do sbatch --job-name="${nb%.ipynb}" slurm/train-notebook.sbatch "${nb%.ipynb}"; done
# --time=16:00:00 is a guess for the longest regime (4,125 steps); override with sbatch --time=... . Resubmitting
# continues an unfinished regime from its run folder.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch --job-name=<notebook> slurm/train-notebook.sbatch <notebook>  (name without .ipynb)" >&2
    exit 1
fi
NOTEBOOK="$1"

module load python/3.13.2 arrow/25.0.0
source ~/envs/kev-cu13/bin/activate
export HF_HOME=$HOME/links/projects/def-yuma/wadi/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
if [ ! -f "${NOTEBOOK}.ipynb" ]; then
    echo "Submit from the ms_paper_beta_loss folder (${NOTEBOOK}.ipynb not found in $PWD)" >&2
    exit 1
fi

echo "job $SLURM_JOB_ID ($NOTEBOOK) on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
jupyter nbconvert --version

jupyter nbconvert --to notebook --execute "${NOTEBOOK}.ipynb" \
    --output "${NOTEBOOK}.rorqual.ipynb" \
    --ExecutePreprocessor.kernel_name=kev-cu13 \
    --ExecutePreprocessor.timeout=-1

echo "finished $(date)"
