#!/bin/bash
#SBATCH --account=def-yuma
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
# Runs one test-best notebook for ONE run (TEST_RUN) on a 40 GB MIG slice of an H100 (3/8 of the GPU's compute,
# Nibi's h100_3g.40gb; inference only, 9B bf16 weights ~18 GB), so several runs of the same model can be tested at
# the same time, each in its own job. Only about half of Nibi's GPU nodes are MIG-configured. A run's test datasets
# already in its test_best/ are reused; only the missing ones are evaluated. The executed copy is saved next to the notebook as
# <notebook>.<run>.nibi.ipynb (one per run, so parallel jobs never write the same file; the source notebook is not
# modified). Arguments: the notebook's path relative to the paper folder (code/papers/pm), and the run: a run folder
# name.
# Submit every run of a notebook at once: slurm/submit-test-best.sh <notebook> <model slug> [sbatch options]
# One run by hand, from the paper folder:
#   sbatch --job-name=<notebook>-<run> slurm/test-best.sbatch analysis/<notebook>.ipynb <run folder name>
# --time=1:00:00 is a guess for one 9B checkpoint on the missing test datasets (a 3/8 slice is slower than a full
# H100); override with sbatch --time=... . --cpus-per-task=4 and --mem=48G are guesses for inference only (a CPU or
# memory request above the slice's share is billed too); raise them if the log shows an out-of-memory kill.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch --job-name=<name> slurm/test-best.sbatch <folder>/<notebook>.ipynb <run folder name>" >&2
    exit 1
fi
NOTEBOOK_PATH="$1"
export TEST_RUN="$2"

module load python/3.13.2 arrow/25.0.0
source ~/envs/kev-cu13/bin/activate
export HF_HOME=$HOME/links/projects/def-yuma/wadi/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# lets the CUDA allocator grow segments instead of fragmenting
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

# The notebook must pick its run from TEST_RUN. A copy that does not (an old copy left by sync-nibi.sh --update, or
# the 0.8B / 4B notebooks) would test every run in every job, with parallel jobs writing the same test_best/ folders.
python3 - "$NOTEBOOK_PATH" <<'PYEOF'
import json
import sys

notebook = json.load(open(sys.argv[1]))
code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
if 'os.environ["TEST_RUN"]' not in code:
    raise SystemExit(f"{sys.argv[1]} does not read TEST_RUN (an old copy on Nibi? push it again)")
PYEOF

echo "job $SLURM_JOB_ID ($NOTEBOOK_PATH, TEST_RUN=$TEST_RUN) on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --output is relative to the notebook's own folder, so the executed copy lands next to the notebook
jupyter nbconvert --to notebook --execute "$NOTEBOOK_PATH" \
    --output "${NOTEBOOK_NAME}.${TEST_RUN}.nibi.ipynb" \
    --ExecutePreprocessor.kernel_name=kev-cu13 \
    --ExecutePreprocessor.timeout=-1

echo "finished $(date)"
