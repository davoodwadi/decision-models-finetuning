# pm: code for the pm paper

The manuscript is `manuscript/pm/manuscript-pm.qmd`. This folder holds everything that produces its numbers, tables
and figures. Nothing is written yet.

## Rules

| From `code/` (shared by every paper, never copied) | This paper's own copy (`src/`) |
|---|---|
| `kev/` (Kev model library) | Training and analysis modules, copied in when a notebook first needs them (e.g. `training/`, `beta_ols/`, `beta_train/` from another paper's `src/`); empty for now |
| `data/` (dataset files and loaders) | |
| `models/` (`KevPointerModel`) | |

- Notebooks and modules are copied, not shared: a change here never changes another paper, and the reverse.
- Large files are not copied: a notebook here may read another paper's checkpoints and run folders
  (`code/papers/<paper>/runs-*/`), but never writes into another paper's folder.
- Environment: the repository's `.venv-cu13` (locally) and `~/envs/kev-cu13` (on the clusters).

## Folders

| Folder | Contents |
|---|---|
| `src/` | This paper's modules |
| `train/` | Training notebooks |
| `analysis/` | Evaluation and analysis notebooks |
| `assets/` | One notebook per manuscript figure or table, named by its Quarto label; writes `<label>.png`/`.pdf` or `<label>.md` |
| `slurm/` | Batch jobs for running a notebook on Rorqual or Nibi (submitted from this folder) |
| `sync/` | rsync scripts between this folder and the clusters (`runs/` on the cluster → `runs-rorqual/`, `runs-nibi/` here) |

`manuscript/pm/assets` is a symlink to `assets/`.

## Standard setup cell

The first code cell of every notebook in this paper. It finds `code/` by walking up from the notebook's folder, then
puts `src/` ahead of `code/` on `sys.path`, so this paper's modules load from `src/` and `kev`, `data` and `models`
load from `code/`.

```python
import os
import sys
from pathlib import Path

# Walk up from the notebook folder to the repository's `code/` root
CODE_DIR = next(
    (c for p in (Path.cwd(), *Path.cwd().parents) for c in (p, p / "code") if all((c / d).is_dir() for d in ("kev", "data", "models"))),
    None,
)
if CODE_DIR is None:
    raise RuntimeError(f"No code/ root containing kev/, data/, and models/ at or above {Path.cwd()}")

PAPER_DIR = CODE_DIR / "papers" / "pm"
SRC_DIR = PAPER_DIR / "src"

# Run from `code/` so data loaders and shared disk caches resolve as in the other papers
os.chdir(CODE_DIR)
for _p in (str(CODE_DIR), str(CODE_DIR / "kev"), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"CODE_DIR : {CODE_DIR}")
print(f"PAPER_DIR: {PAPER_DIR}")
```
