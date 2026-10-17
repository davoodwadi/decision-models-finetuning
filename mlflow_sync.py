"""Sync runs/issp-* folders (run_info.json, training_log.json, metrics.json) to a local MLflow SQLite store.

Called automatically by train.py after each validation and at run completion, and runnable standalone to sync
runs using the held-out ISSP calibration set (v9+):
  kev/.venv/bin/python mlflow_sync.py
Then view all runs in the browser:
  kev/.venv/bin/mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
"""
import datetime
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities import Metric, Param, RunStatus, RunTag  # noqa: E402
from mlflow.tracking import MlflowClient                      # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "runs"
DB_URI = f"sqlite:///{(RUNS_DIR / 'mlflow.db').resolve()}"
EXPERIMENT = "issp"


def _client():
    client = MlflowClient(tracking_uri=DB_URI)
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)
    return client, exp_id


def sync_run(folder, client=None, exp_id=None):
    """Create or update the MLflow run for `folder` (e.g. runs/issp-all-0.8b-v9) from its JSON files."""
    folder = Path(folder)
    log_path = folder / "training_log.json"
    if not log_path.exists():
        return
    tlog = json.loads(log_path.read_text(encoding="utf-8"))
    info_path = folder / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    metrics_path = folder / "metrics.json"
    final_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    if client is None or exp_id is None:
        client, exp_id = _client()

    existing = client.search_runs([exp_id], f" attributes.run_name = '{folder.name}' ")
    now_ms = int(time.time() * 1000)
    if existing:
        run_id = existing[0].info.run_id
    else:
        started = info.get("started")
        start_ms = int(datetime.datetime.fromisoformat(started).timestamp() * 1000) if started else now_ms
        run = client.create_run(exp_id, start_time=start_ms, run_name=folder.name,
                                tags={"mlflow.runName": folder.name, "folder": str(folder)})
        run_id = run.info.run_id

    # Parameters (logged once when the run has no params yet)
    current_run = client.get_run(run_id)
    last_step = int(current_run.data.tags.get("last_synced_step", -1))
    if not current_run.data.params:
        args = info.get("settings") or tlog.get("args") or {}
        recipe = args.get("recipe")   # runs with a data recipe (--data); earlier runs have the dataset flags instead
        source = next(iter(recipe["source"].values())) if recipe else {}
        trains = info.get("train_datasets") or args.get("train", [])
        train_str = f"{len(trains)} yrs" if len(trains) > 3 else ",".join(trains)
        params = {
            "base": args.get("base", ""),
            "data": args.get("data", ""),
            "construction": next(iter(recipe["construction"])) if recipe else "",
            "train": train_str,
            "epochs": args.get("epochs", ""),
            "lr": args.get("lr", ""),
            "batch": args.get("batch", ""),
            "single_respondent": ("single_respondent" in json.dumps(recipe["construction"]) if recipe
                                  else args.get("single_respondent", False)),
            "calibration_frac": source.get("calibration_frac", "") if recipe else args.get("calibration_frac", ""),
            # runs before 2026-10-14 had no --augment: they trained on every variation (full), p_reverse 0.5
            "augment": (" ".join(args["augment"]) or "none") if "augment" in args else "full",
            "p_reverse": args.get("p_reverse", ""),
            "seed": args.get("seed", ""),
        }
        train_set = info.get("sets", {}).get("train", {})
        if "slots" in train_set:
            params["slots"] = train_set["slots"]
        by_resp = train_set.get("drawn_records", {}).get("targets_by_respondents", {})
        if "1" in by_resp:
            params["share_n1"] = by_resp["1"]
        client.log_batch(run_id, params=[Param(k, str(v)) for k, v in params.items() if v != ""])

    # Step & validation metrics (only new steps since last sync)
    metric_list = []
    max_step = last_step
    for s in tlog.get("steps", []):
        if not isinstance(s, dict) or "step" not in s:
            continue
        step = int(s["step"])
        if step <= last_step:
            continue
        max_step = max(max_step, step)
        for k in ("kl_bits", "loss", "grad_norm", "lr"):
            if k in s and s[k] is not None:
                metric_list.append(Metric(f"train/{k}", float(s[k]), now_ms, step))

    for v in tlog.get("validations", []):
        step = int(v["step"])
        if step <= last_step:
            continue
        max_step = max(max_step, step)
        cal = v.get("calibration", {})
        for qtype, vals in cal.items():
            if "kl_divergence" in vals:
                metric_list.append(Metric(f"cal/kl_{qtype}", float(vals["kl_divergence"]), now_ms, step))
            if "wasserstein" in vals:
                metric_list.append(Metric(f"cal/w_{qtype}", float(vals["wasserstein"]), now_ms, step))

    best_step = int(info.get("best_step") or tlog.get("best_step") or 0)
    if tlog.get("best_step") is not None:
        metric_list.append(Metric("best_step", float(best_step), now_ms, best_step))
    if "best_calibration_kl_divergence" in info:
        metric_list.append(Metric("best/cal_kl_all", float(info["best_calibration_kl_divergence"]), now_ms, best_step))

    for set_name, by_model in final_metrics.items():
        clean_set = re.sub(r"[^A-Za-z0-9_.\- :/]", "", set_name).strip()
        for model_label, by_type in by_model.items():
            if not isinstance(by_type, dict):
                continue
            prefix = f"{clean_set}_step0" if "step 0" in model_label else clean_set
            for qtype, vals in by_type.items():
                if not isinstance(vals, dict):
                    continue
                if "kl_divergence" in vals:
                    metric_list.append(Metric(f"{prefix}/kl_{qtype}", float(vals["kl_divergence"]), now_ms, best_step))
                if "wasserstein" in vals:
                    metric_list.append(Metric(f"{prefix}/w_{qtype}", float(vals["wasserstein"]), now_ms, best_step))

    for i in range(0, len(metric_list), 500):
        client.log_batch(run_id, metrics=metric_list[i:i + 500])
    if max_step > last_step:
        client.set_tag(run_id, "last_synced_step", str(max_step))

    if info.get("finished") or final_metrics:
        client.set_terminated(run_id, status=RunStatus.to_string(RunStatus.FINISHED))


def main():
    client, exp_id = _client()
    folders = []
    for p in sorted(RUNS_DIR.glob("issp-*")):
        info_path = p / "run_info.json"
        if not p.is_dir() or not info_path.exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if "calibration_slots" in info:
            folders.append(p)
            sync_run(p, client, exp_id)
    print(f"Synced {len(folders)} runs to {DB_URI} (experiment '{EXPERIMENT}')")


if __name__ == "__main__":
    main()
