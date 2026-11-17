# run-viewer

Read-only web viewer for `code/train-notebooks/beta-training/runs/<notebook>/<run>/`.
It reads `run.json`, `steps.csv` and `val_history.csv` on every request; it never lists or opens `checkpoints/` or `betas/`.

```bash
pnpm install
pnpm dev   # http://localhost:3210
```

Or start the `run-viewer` entry in `.claude/launch.json`.

- Runs folder: `../train-notebooks/beta-training/runs` relative to this folder; set `RUNS_DIR` (for example in `.env.local`) to read another one, such as the main checkout's runs from a git worktree.
- Validators and metrics come from the `val_history.csv` header (`<validator>/<metric>` columns).
- A last CSV line without a newline is a partial write and is skipped; an empty or header-only file means no rows yet. Any other malformed row is shown as an error.
- Running runs have no `summary` in `run.json`, so total steps show as `?` and the best step/value is computed from `val_history.csv` ("so far").
- Run pages poll every 10 s while the run's status is `running`.
- Older run folders without `identity` in `run.json` (v5) are listed with an error instead of being read.
- Use `pnpm dev`: `next start` replaces server error messages on the error page with a generic text.
