"""Population OLS evaluation for any dataset module in `beta_ols`, using the module's own coding.

The model's betas on the full dataset are computed exactly like the human replication and the training loss:
the module's `respondent_beta_part` gives every outcome row, and betas = B @ p with B = pinv(X)[targets] @ W
(unstandardized, in the paper's units). Only the probabilities differ: one-hot human choices vs model predictions.

Usage:
    eval_ops = build_population_eval(dataset_module, choice_df, specs, eval_records)
    probs_flat = flat_probs_from_predictions(eval_ops, preds)        # preds in eval_records order
    metrics, beta_df = compare_population_betas(eval_ops, probs_flat)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from beta_ols.core import RegressionSpec, iter_respondents, raw_beta_operator


def build_population_eval(dataset_module, choice_df: pd.DataFrame, specs: dict[str, RegressionSpec], eval_records: list[dict]) -> dict:
    # 1. Flat option vector over the whole dataset, respondent by respondent, task by task (module order)
    y_parts = []
    rows_by_group: dict[str, list[tuple[list[float], list[int], list[float]]]] = {}
    task_offset_by_key: dict[tuple[str, str], tuple[int, int]] = {}
    offset = 0
    for caseid, r_df in iter_respondents(choice_df):
        part = dataset_module.respondent_beta_part(r_df)
        y_parts.append(part.y_flat)
        for row in part.outcome_rows:
            if row.group not in rows_by_group:
                rows_by_group[row.group] = []
            rows_by_group[row.group].append((row.x, [offset + i for i in row.option_idx], row.weights))
        # Each task is identified by (CASEID, qid); its options start at `task_start` in the flat vector
        task_start = offset
        for qid, n_options in zip(r_df["qid"], part.num_options, strict=True):
            key = (caseid, str(qid))
            if key in task_offset_by_key:
                raise ValueError(f"{caseid}: task qid {qid!r} appears twice; tasks cannot be matched to records")
            task_offset_by_key[key] = (task_start, n_options)
            task_start += n_options
        offset += part.y_flat.shape[0]
    total_options = offset
    y_flat = torch.cat(y_parts)

    # 2. Unstandardized population operators, one per regression (the full data must be full rank)
    operators = []
    for group in specs:
        B = raw_beta_operator(rows_by_group[group], specs[group], total_options)
        if B is None:
            raise ValueError(f"{group}: full-data design is rank-deficient")
        operators.append({"group": group, "B": B})

    # 3. Where each evaluation record's options go in the flat vector (records may be in any order)
    if len(eval_records) != len(task_offset_by_key):
        raise ValueError(f"Expected {len(task_offset_by_key)} evaluation records (one per task), got {len(eval_records)}")
    record_positions = []
    for rec in eval_records:
        key = (str(rec["CASEID"]), str(rec["qid"]))
        if key not in task_offset_by_key:
            raise KeyError(f"Evaluation record {key} has no matching task in the module's choice rows")
        record_positions.append(task_offset_by_key[key])
    if len(set(record_positions)) != len(task_offset_by_key):
        raise ValueError("Evaluation records must cover every task exactly once")

    human_betas = {}
    for op in operators:
        human_betas[op["group"]] = op["B"] @ y_flat
    return {
        "specs": specs,
        "operators": operators,
        "total_options": total_options,
        "record_positions": record_positions,
        "human_betas": human_betas,
    }


def flat_probs_from_predictions(eval_ops: dict, preds: list) -> torch.Tensor:
    # Place each record's predicted option probabilities at its task's position in the flat option vector
    if len(preds) != len(eval_ops["record_positions"]):
        raise ValueError(f"Expected {len(eval_ops['record_positions'])} predictions, got {len(preds)}")
    probs_flat = torch.full((eval_ops["total_options"],), float("nan"), dtype=torch.float64)
    for (task_start, n_options), pred in zip(eval_ops["record_positions"], preds, strict=True):
        pred_t = torch.as_tensor(np.asarray(pred, dtype=np.float64))
        if pred_t.shape != (n_options,):
            raise ValueError(f"Expected {n_options} option probabilities, got shape {tuple(pred_t.shape)}")
        probs_flat[task_start : task_start + n_options] = pred_t
    if bool(torch.isnan(probs_flat).any()):
        raise ValueError("Some tasks received no prediction")
    return probs_flat


def compare_population_betas(eval_ops: dict, probs_flat: torch.Tensor) -> tuple[dict[str, float], pd.DataFrame]:
    # Model vs human population betas over every target of every regression (paper units)
    beta_rows = []
    for op in eval_ops["operators"]:
        spec = eval_ops["specs"][op["group"]]
        model_betas = op["B"] @ probs_flat
        human_betas = eval_ops["human_betas"][op["group"]]
        for label, human_beta, model_beta in zip(spec.target_labels, human_betas.tolist(), model_betas.tolist(), strict=True):
            beta_rows.append({"model": op["group"], "term": label, "human_pop_beta": human_beta, "model_pop_beta": model_beta})
    beta_df = pd.DataFrame(beta_rows).set_index("term")
    beta_df["diff"] = beta_df["model_pop_beta"] - beta_df["human_pop_beta"]
    beta_df["abs_diff"] = beta_df["diff"].abs()

    if bool((beta_df["human_pop_beta"] == 0).any()):
        raise ValueError("Zero human beta encountered; MAPE denominator would be undefined")
    metrics = {
        "beta_mse": float((beta_df["diff"] ** 2).mean()),
        "beta_mae": float(beta_df["abs_diff"].mean()),
        "beta_mape_pct": float((beta_df["abs_diff"] / beta_df["human_pop_beta"].abs()).mean() * 100.0),
        "spearman_rho": float(spearmanr(beta_df["model_pop_beta"], beta_df["human_pop_beta"]).statistic),
    }
    return metrics, beta_df
