#!/bin/bash
#SBATCH --job-name=01-train-paper-800m
#SBATCH --account=def-yuma
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
# Runs 01-train-paper-800m.ipynb top to bottom on one H100 and saves the executed copy as
# 01-train-paper-800m.rorqual.ipynb (the source notebook is not modified).
# Submit from the ms_paper_beta_loss folder:  sbatch slurm/01-train-paper-800m.sbatch
# The log goes to 01-train-paper-800m-<jobid>.out in that folder.
# Resubmitting continues unfinished regimes from their run folders and loads finished ones from disk.
set -euo pipefail

module load python/3.13.2 arrow/25.0.0
source ~/envs/kev-cu13/bin/activate
export HF_HOME=$HOME/links/projects/def-yuma/wadi/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
if [ ! -f 01-train-paper-800m.ipynb ]; then
    echo "Submit from the ms_paper_beta_loss folder (01-train-paper-800m.ipynb not found in $PWD)" >&2
    exit 1
fi

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
jupyter nbconvert --version

jupyter nbconvert --to notebook --execute 01-train-paper-800m.ipynb \
    --output 01-train-paper-800m.rorqual.ipynb \
    --ExecutePreprocessor.kernel_name=kev-cu13 \
    --ExecutePreprocessor.timeout=-1

echo "finished $(date)"
