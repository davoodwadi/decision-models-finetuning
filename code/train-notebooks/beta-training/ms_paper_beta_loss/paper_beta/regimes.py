"""Paper row labels and step-0 / cross-run comparison helpers for `ms_paper_beta_loss`.

Labels only: each regime's design (datasets, beta_lambda, kl_anchor_alpha, epochs, LR) is declared in the training
notebooks and saved in each run's run.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from beta_train import runs as run_store
from beta_train.validation import METRIC_NAMES
from .trainer import load_paper_run


def _sign_accuracy(beta_df: pd.DataFrame) -> float:
    model_sign = np.sign(beta_df["model_pop_beta"].to_numpy(dtype=float))
    human_sign = np.sign(beta_df["human_pop_beta"].to_numpy(dtype=float))
    return float(np.mean(model_sign == human_sign))


@dataclass(frozen=True)
class PaperRegimeSpec:
    slug: str           # run folder name
    row_code: str       # row in the paper table
    paper_label: str    # regime name shown in tables
    data_regime: str    # data the regime trains on, in words


PAPER_REGIMES: dict[str, PaperRegimeSpec] = {
    spec.slug: spec
    for spec in (
        PaperRegimeSpec(
            slug="row2_wu_table_anchored",
            row_code="Row 2",
            paper_label="Table Only (Wu fitted OLS beta* + KL Anchor)",
            data_regime="Wu Table (8 slopes, 0 choices)",
        ),
        PaperRegimeSpec(
            slug="row2_ablation_wu_table_no_kl",
            row_code="Row 2 (Ablation)",
            paper_label="Table Only (Wu fitted OLS beta*, No KL Anchor)",
            data_regime="Wu Table (8 slopes, 0 choices)",
        ),
        PaperRegimeSpec(
            slug="row2b_wu_ce_oracle",
            row_code="Row 2b",
            paper_label="Single-Study Microdata Reference (Wu y CE)",
            data_regime="Wu Microdata (18,591 choices)",
        ),
        PaperRegimeSpec(
            slug="row3_berke_ce_only",
            row_code="Row 3",
            paper_label="In-House Microdata Only (Berke y CE)",
            data_regime="Berke Microdata (36,297 choices)",
        ),
        PaperRegimeSpec(
            slug="row4_hybrid_berke_ce_wu_table",
            row_code="Row 4",
            paper_label="Hybrid: In-House Microdata + External Table (Berke y + Wu beta*)",
            data_regime="Berke Microdata + Wu Table (Wu y withheld)",
        ),
        PaperRegimeSpec(
            slug="row5_full_berke_wu_ce_oracle",
            row_code="Row 5",
            paper_label="Full-Microdata Reference (Wu + Berke y CE)",
            data_regime="Wu + Berke Microdata (54,888 choices)",
        ),
    )
}

MAIN_TABLE_ORDER: tuple[str, ...] = (
    "row2_wu_table_anchored",
    "row3_berke_ce_only",
    "row4_hybrid_berke_ce_wu_table",
    "row5_full_berke_wu_ce_oracle",
)

ALL_REGIME_ORDER: tuple[str, ...] = (
    "row2_wu_table_anchored",
    "row2_ablation_wu_table_no_kl",
    "row2b_wu_ce_oracle",
    "row3_berke_ce_only",
    "row4_hybrid_berke_ce_wu_table",
    "row5_full_berke_wu_ce_oracle",
)


def extract_step0_from_run_dir(run_dir: Path, validators: tuple[str, ...] = ("kriner2026", "grummon2025")) -> dict:
    """Extract Row 1 (Untrained Base LLM, step 0) metrics and term-level betas from any saved run directory."""
    run_dir = Path(run_dir)
    val_history_df = pd.read_csv(run_dir / "val_history.csv").set_index("step")
    if 0 not in val_history_df.index:
        raise ValueError(f"Step 0 missing from {run_dir / 'val_history.csv'}")
    step0_row = val_history_df.loc[0]
    step0_betas = run_store.read_betas(run_dir, 0)
    metrics_out = {}
    for validator_name in validators:
        for metric in (*METRIC_NAMES, "std_beta_mse"):
            col = f"{validator_name}/{metric}"
            metrics_out[col] = float(step0_row[col])
        metrics_out[f"{validator_name}/sign_acc"] = _sign_accuracy(step0_betas[validator_name])
    return {
        "row_code": "Row 1",
        "paper_label": "Untrained Base LLM (Zero-Shot)",
        "data_regime": "None (Zero-Shot)",
        "status": "finished",
        "best_step": 0,
        "total_steps": 0,
        "metrics": metrics_out,
        "betas_by_validator": {v: step0_betas[v] for v in validators},
    }


def summarize_paper_runs_dir(
    runs_dir: Path,
    validators: tuple[str, ...] = ("kriner2026", "grummon2025"),
    metrics: tuple[str, ...] = ("beta_mse", "std_beta_mse", "beta_mae", "spearman_rho", "sign_acc", "kl_loss"),
) -> pd.DataFrame:
    """Build a clean summary DataFrame across all paper rows saved in `runs_dir` (including Row 1 step 0 if any run exists)."""
    runs_dir = Path(runs_dir)
    rows = []
    first_existing_run_dir: Path | None = None
    for slug in ALL_REGIME_ORDER:
        candidate = runs_dir / slug
        if (candidate / "run.json").exists():
            first_existing_run_dir = candidate
            break

    if first_existing_run_dir is not None:
        base_info = extract_step0_from_run_dir(first_existing_run_dir, validators)
        base_row = {
            "row": base_info["row_code"],
            "slug": "row1_base_untrained",
            "regime": base_info["paper_label"],
            "status": "cached_step0",
            "best_step": 0,
            "total_steps": 0,
        }
        for v in validators:
            for m in metrics:
                base_row[f"{v}/{m}"] = base_info["metrics"][f"{v}/{m}"]
        rows.append(base_row)

    for slug in ALL_REGIME_ORDER:
        spec = PAPER_REGIMES[slug]
        run_dir = runs_dir / slug
        if not (run_dir / "run.json").exists():
            missing_row = {
                "row": spec.row_code,
                "slug": slug,
                "regime": spec.paper_label,
                "status": "not_run_yet",
                "best_step": None,
                "total_steps": None,
            }
            for v in validators:
                for m in metrics:
                    missing_row[f"{v}/{m}"] = float("nan")
            rows.append(missing_row)
            continue

        run_json = run_store.read_run_json(run_dir)
        result = load_paper_run(run_dir)
        total_steps = (
            int(result.summary["total_steps"])
            if "total_steps" in result.summary.index
            else int(result.step_df["step"].max()) if len(result.step_df) > 0 else 0
        )
        row = {
            "row": spec.row_code,
            "slug": slug,
            "regime": spec.paper_label,
            "status": run_json["status"],
            "best_step": int(result.best["step"]),
            "total_steps": total_steps,
        }
        for v in validators:
            for m in metrics:
                col = f"{v}/{m}"
                if m == "sign_acc":
                    row[col] = _sign_accuracy(result.best_validation[v][1]) if v in result.best_validation else float("nan")
                else:
                    row[col] = float(result.best_row[col])
        rows.append(row)

    return pd.DataFrame(rows).set_index("slug")
