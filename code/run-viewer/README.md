# run-viewer

Read-only web viewer for training runs: `<run folder>/<notebook>/<run>/`, for every run folder listed below.
It reads `run.json`, `steps.csv`, `val_history.csv` and `betas/` on every request; it never lists or opens `checkpoints/`.

```bash
pnpm install
pnpm dev   # http://localhost:3210
```

Or start the `run-viewer` entry in `.claude/launch.json`.

Built with shadcn/ui (Radix + Tailwind) for components, TanStack Table for sortable tables and Apache ECharts for plots.
Charts: drag to pan, ctrl + scroll to zoom, toolbar for box zoom / reset / PNG, click a legend entry to hide a series, log-y switch.
Validation charts of one run are linked (shared crosshair, zoom and legend). Add components with `pnpm dlx shadcn@4.23.0 add <name>`.

- Run folders (each a section on the home page, URLs `/s/<folder>/n/<notebook>/r/<run>`):
  - `beta-loss-exploration`: `../papers/beta-loss/exploration/runs`
  - `beta-loss`, `beta-loss-rorqual`, `beta-loss-nibi`: `../papers/beta-loss/runs`, `runs-rorqual`, `runs-nibi`
  - `sim-respondents-rorqual`, `sim-respondents-nibi`: `../papers/sim-respondents/runs-rorqual`, `runs-nibi`
  
  Paths are relative to this folder. Set `RUNS_SOURCES="name=path;name=path"` (for example in `.env.local`) to replace the list,
  such as the main checkout's runs from a git worktree. A missing folder is an error.
- Validators and metrics come from the `val_history.csv` header (`<validator>/<metric>` columns).
- A last CSV line without a newline is a partial write and is skipped; an empty or header-only file means no rows yet. Any other malformed row is shown as an error.
- Running runs have no `summary` in `run.json`, so total steps show as `?` and the best step/value is computed from `val_history.csv` ("so far").
- Run pages poll every 10 s while the run's status is `running`.
- Older run folders without `identity` in `run.json` (v5) are listed with an error instead of being read.
- Use `pnpm dev`: `next start` replaces server error messages on the error page with a generic text.
