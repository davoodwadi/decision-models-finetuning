#!/bin/bash
# Submits one slurm/test-best.sbatch job per run of a test-best notebook, all at once: one per finished run folder
# (a folder with run.json whose status is "finished") in the notebook's runs folders.
# Runs already tested on every dataset in the notebook's TEST_DATASETS get no job (no GPU time spent on a model load
# that would only skip).
# Each job is named <notebook>-<run> (log: <notebook>-<run>-<jobid>.out in the paper folder).
# Meant for 04-test-best-9b.ipynb, whose runs are in runs-rorqual/kev-9b and runs-nibi/kev-9b (pushed by
# sync/push-nibi-test-runs.sh). The model slug is the second argument.
# Extra arguments go to sbatch and override the script's defaults (h100_3g.40gb, 4 CPUs, 48G, 1 h).
# Run on Nibi from anywhere:
#   slurm/submit-test-best.sh analysis/test-best/04-test-best-9b.ipynb kev-9b
#   slurm/submit-test-best.sh analysis/test-best/04-test-best-9b.ipynb kev-9b --time=0:45:00
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: slurm/submit-test-best.sh <folder>/<notebook>.ipynb <model slug, e.g. kev-9b> [sbatch options...]" >&2
    exit 1
fi
NOTEBOOK_PATH="$1"
MODEL_SLUG="$2"
shift 2

# sbatch runs the job from the folder it is called in: the paper folder
cd "$(dirname "$0")/.."
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Notebook $NOTEBOOK_PATH not found in $PWD" >&2
    exit 1
fi
NOTEBOOK_NAME="$(basename "$NOTEBOOK_PATH" .ipynb)"

for runs_dir in "runs-rorqual/${MODEL_SLUG}" "runs-nibi/${MODEL_SLUG}"; do
    if [ ! -d "$runs_dir" ]; then
        echo "No runs folder $runs_dir in $PWD (push it with sync/push-nibi-test-runs.sh)" >&2
        exit 1
    fi
done

# Which runs need a job: every finished run, minus those whose test_best/test.json already holds every
# dataset in TEST_DATASETS for this checkpoint step and model (the notebook's own already_tested rule). TEST_DATASETS and the model
# key (KEV_BASE, LORA_RANK, WEIGHTS_DTYPE, HEAD_TEMPERATURE) are read from the notebook itself, so they cannot drift.
# A run's checkpoint step is run.json's summary.best_step (the step of its best.pt). Row 1 belongs to the
# sim-respondents paper and gets no job here.
# Prints "submit <run>" or "skip <run>: <reason>" per run.
DECISIONS="$(python3 - "$NOTEBOOK_PATH" "runs-rorqual/${MODEL_SLUG}" "runs-nibi/${MODEL_SLUG}" <<'PYEOF'
import ast
import json
import re
import sys
from pathlib import Path

notebook_path = sys.argv[1]
runs_dirs = [Path(p) for p in sys.argv[2:]]

notebook = json.load(open(notebook_path))
code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
# A notebook that ignores TEST_RUN would test every run in every job (parallel jobs writing the same folders)
if 'os.environ["TEST_RUN"]' not in code:
    raise SystemExit(f"{notebook_path} does not read TEST_RUN (an old copy on Nibi? push it again)")


def notebook_value(name):
    # The literal assigned to `name` at the start of a line; it must be assigned exactly once
    matches = re.findall(rf"^{name} = (.+)$", code, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{notebook_path}: expected one '{name} = ...' line, found {len(matches)}")
    return ast.literal_eval(matches[0])


test_datasets = notebook_value("TEST_DATASETS")
model_key = {
    "base": notebook_value("KEV_BASE"),
    "lora_rank": notebook_value("LORA_RANK"),
    "weights_dtype": notebook_value("WEIGHTS_DTYPE"),
    "head_temperature": notebook_value("HEAD_TEMPERATURE"),
}


def decide(name, test_dir, checkpoint_step):
    test_json = test_dir / "test.json"
    if not test_json.exists():
        return f"submit {name}"
    stored = json.load(open(test_json))
    if stored["checkpoint_step"] != checkpoint_step:
        return f"submit {name}"
    if stored["model"] != model_key:
        return f"submit {name}"
    # Same rule as the notebook's already_tested: every TEST_DATASETS dataset saved (other saved datasets do not matter)
    missing = [d for d in test_datasets if d not in stored["test_datasets"]]
    if missing:
        return f"submit {name}"
    return f"skip {name}: all of {test_datasets} already tested at step {checkpoint_step}"


for runs_dir in runs_dirs:
    for run_json in sorted(runs_dir.glob("*/run.json")):
        run_dir = run_json.parent
        run_info = json.load(open(run_json))
        if run_info["status"] != "finished":
            print(f"skip {run_dir.name}: status {run_info['status']}")
            continue
        print(decide(run_dir.name, run_dir / "test_best", int(run_info["summary"]["best_step"])))
PYEOF
)"

RUNS=()
while read -r action run rest; do
    if [ "$action" = "submit" ]; then
        RUNS+=("$run")
    else
        echo "skip $run $rest"
    fi
done <<< "$DECISIONS"

if [ "${#RUNS[@]}" -eq 0 ]; then
    echo "nothing to submit: every run is already tested on the notebook's TEST_DATASETS"
    exit 0
fi

for run in "${RUNS[@]}"; do
    sbatch --job-name="${NOTEBOOK_NAME}-${run}" "$@" slurm/test-best.sbatch "$NOTEBOOK_PATH" "$run"
done
echo "submitted ${#RUNS[@]} jobs"
