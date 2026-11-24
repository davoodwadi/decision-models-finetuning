#!/bin/bash
#SBATCH --job-name=00-validate-kl-anchor-9b
#SBATCH --account=def-yuma
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# Runs train/kl-anchor/00-validate-kl-anchor-9b.ipynb top to bottom on one H100 (micro-chunk search on, max_chunk = 41) and saves the
# executed copy next to it as 00-validate-kl-anchor-9b.rorqual.ipynb (the source notebook is not modified).
# Submit from the paper folder (code/papers/beta-loss):  sbatch slurm/00-validate-kl-anchor-9b.sbatch
# The log goes to 00-validate-kl-anchor-9b-<jobid>.out in that folder.
# Resubmitting continues unfinished alpha runs from their run folders and loads finished ones from disk.
set -euo pipefail

module load python/3.13.2 arrow/25.0.0
source ~/envs/kev-cu13/bin/activate
export HF_HOME=$HOME/links/projects/def-yuma/wadi/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
if [ ! -f train/kl-anchor/00-validate-kl-anchor-9b.ipynb ]; then
    echo "Submit from the paper folder (train/kl-anchor/00-validate-kl-anchor-9b.ipynb not found in $PWD)" >&2
    exit 1
fi

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
jupyter nbconvert --version

jupyter nbconvert --to notebook --execute train/kl-anchor/00-validate-kl-anchor-9b.ipynb \
    --output 00-validate-kl-anchor-9b.rorqual.ipynb \
    --ExecutePreprocessor.kernel_name=kev-cu13 \
    --ExecutePreprocessor.timeout=-1

echo "finished $(date)"
