"""Shared building blocks for every dataset module in `beta_ols`.

Vocabulary used throughout the package
- regression (group): one OLS model we replicate, e.g. "wu2025" or "berke2023/v2". A dataset can have several.
- outcome row: one observation of a regression. It has a design row `x` and an outcome that is a fixed linear
  combination of the respondent's option probabilities (or, for humans, of their one-hot choices).
- respondent beta part: everything one respondent contributes: option counts per task, flat one-hot choices,
  and outcome rows. Every dataset module builds it with ONE function (`respondent_beta_part`), and both the human
  replication and the training loss are built from it, so the two can never use different codings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

# Relative singular-value threshold below which a design matrix counts as rank-deficient
OLS_SVD_RCOND = 1e-8


@dataclass(frozen=True)
class RegressionSpec:
    """One regression we replicate."""

    name: str                   # group key, used by OutcomeRow.group
    design_columns: list[str]   # column order of every OutcomeRow.x in this group, "Intercept" first
    target_columns: list[str]   # the coefficients that are beta targets (a subset of design_columns)
    target_labels: list[str]    # human-readable names of the target coefficients, same order
    sigma: float                # fixed SD of this regression's human outcome on the full dataset
    cov_type: str               # "cluster" (clustered by respondent) or "HC1"

    def target_indices(self) -> list[int]:
        # Positions of the target coefficients in the design columns
        return [self.design_columns.index(col) for col in self.target_columns]


@dataclass(frozen=True)
class OutcomeRow:
    """One observation of a regression."""

    group: str                  # which regression (RegressionSpec.name)
    x: list[float]              # design row, in RegressionSpec.design_columns order
    option_idx: list[int]       # positions within the respondent's flat option vector (task by task, prompt order)
    weights: list[float]        # outcome = sum(weights[k] * p[option_idx[k]])


@dataclass(frozen=True)
class RespondentBetaPart:
    """Everything one respondent contributes to the regressions."""

    num_options: list[int]      # options per task, in prompt order
    y_flat: torch.Tensor        # human one-hot choices, task by task, flattened (float64, CPU)
    outcome_rows: list[OutcomeRow]


def check_one_hot(values: list[float], expected_options: int, where: str) -> list[float]:
    # Every respondent-level choice must be exactly one chosen option out of `expected_options`
    if len(values) != expected_options or sorted(values) != [0.0] * (expected_options - 1) + [1.0]:
        raise ValueError(f"{where}: expected a {expected_options}-option one-hot choice, got {values}")
    return values


def parse_treatment_term(term: str) -> tuple[str, int]:
    # "C(airline_n)[T.2]" -> ("airline_n", 2): a published AMCE term = indicator that `column` equals `level`
    if not (term.startswith("C(") and ")[T." in term and term.endswith("]")):
        raise ValueError(f"Unexpected treatment term {term!r}")
    column = term[2 : term.index(")[T.")]
    level = int(term[term.index(")[T.") + 4 : -1])
    return column, level


def iter_respondents(choice_df: pd.DataFrame):
    # Yield (caseid, one respondent's task rows) in first-appearance order, keeping each respondent's task order.
    # groupby would silently drop rows with a missing CASEID, so those raise instead.
    if bool(choice_df["CASEID"].isna().any()):
        raise ValueError("choice_df has rows without a CASEID")
    for caseid, r_group in choice_df.groupby("CASEID", sort=False):
        yield str(caseid), r_group.reset_index(drop=True)


def regression_rows_from_parts(choice_df: pd.DataFrame, beta_part_fn, design_columns_by_group: dict) -> pd.DataFrame:
    # Build the regression table (one row per outcome row) from every respondent's beta part.
    # Human outcome y = the outcome row's weights applied to the respondent's one-hot choices.
    # `task` = "<CASEID>#<task index>": the data row (answered task) an outcome row comes from; a task's stacked
    # option rows share it (the unit training samples, see beta_noise_by_sample_size).
    table_rows = []
    for caseid, r_df in iter_respondents(choice_df):
        part = beta_part_fn(r_df)
        # Task of each position in the respondent's flat option vector
        task_of_option = []
        for task_idx, n_options in enumerate(part.num_options):
            task_of_option.extend([task_idx] * n_options)
        for row in part.outcome_rows:
            row_tasks = {task_of_option[option_idx] for option_idx in row.option_idx}
            if len(row_tasks) != 1:
                raise ValueError(f"{caseid}: an outcome row of {row.group} spans tasks {sorted(row_tasks)}")
            design_columns = design_columns_by_group[row.group]
            table_row = {"CASEID": caseid, "task": f"{caseid}#{row_tasks.pop()}", "group": row.group}
            for col, value in zip(design_columns, row.x, strict=True):
                table_row[col] = value
            human_y = 0.0
            for option_idx, weight in zip(row.option_idx, row.weights, strict=True):
                human_y += weight * float(part.y_flat[option_idx])
            table_row["y"] = human_y
            table_rows.append(table_row)
    return pd.DataFrame(table_rows)


def outcome_sigma(y: np.ndarray) -> float:
    # Population SD (ddof=0) of a regression's human outcome; must be finite and positive to standardize by it
    if len(y) == 0:
        raise ValueError("No outcomes to compute an SD from")
    sigma = float(np.std(y, ddof=0))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"Outcome SD must be finite and positive, got {sigma}")
    return sigma


def split_by_task(choice_df: pd.DataFrame, dataset_name: str) -> dict[str, pd.DataFrame]:
    # One pseudo dataset per task variable `v`, named "<dataset>/<v>", in first-appearance order; a dataset with a
    # single task variable stays whole under its own name. Each piece keeps every respondent's rows of that task in order.
    task_names = list(dict.fromkeys(choice_df["v"]))
    if len(task_names) == 1:
        return {dataset_name: choice_df}
    pieces = {}
    for task_name in task_names:
        pieces[f"{dataset_name}/{task_name}"] = choice_df[choice_df["v"] == task_name].reset_index(drop=True)
    return pieces


def fit_human_ols(rows: pd.DataFrame, specs: dict[str, RegressionSpec]) -> pd.DataFrame:
    # Fit every regression on its own rows with statsmodels, using the module's own design columns (no formulas).
    # Returns a tidy table: one row per target coefficient.
    tidy_rows = []
    for group, spec in specs.items():
        group_rows = rows[rows["group"] == group]
        if len(group_rows) == 0:
            raise ValueError(f"No regression rows for group {group!r}")
        X = group_rows[spec.design_columns].astype(float)
        y = group_rows["y"].astype(float)
        if spec.cov_type == "cluster":
            fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": group_rows["CASEID"]})
        elif spec.cov_type == "HC1":
            fit = sm.OLS(y, X).fit(cov_type="HC1")
        else:
            raise ValueError(f"{group}: unknown cov_type {spec.cov_type!r}")
        for col, label in zip(spec.target_columns, spec.target_labels, strict=True):
            beta = float(fit.params[col])
            tidy_rows.append(
                {
                    "model": group,
                    "term": label,
                    "human_beta": beta,
                    "se": float(fit.bse[col]),
                    "p_value": float(fit.pvalues[col]),
                    "sigma_y": spec.sigma,
                    "std_beta": beta / spec.sigma,
                    "n_rows": len(group_rows),
                }
            )
    return pd.DataFrame(tidy_rows)


def wide_ols_table(tidy: pd.DataFrame) -> pd.DataFrame:
    # Display form: one column group per regression (model), rows = target terms, so separate models are visible.
    # Term labels are stripped of a leading "<model> | " prefix when present.
    columns = {}
    for model, model_rows in tidy.groupby("model", sort=False):
        term_labels = [str(t).split(" | ")[-1] for t in model_rows["term"]]
        columns[(model, "beta")] = pd.Series(model_rows["human_beta"].to_numpy(), index=term_labels)
        columns[(model, "se")] = pd.Series(model_rows["se"].to_numpy(), index=term_labels)
        columns[(model, "std_beta")] = pd.Series(model_rows["std_beta"].to_numpy(), index=term_labels)
    table = pd.DataFrame(columns)
    sigma_row = {}
    n_row = {}
    for model, model_rows in tidy.groupby("model", sort=False):
        sigma_row[(model, "beta")] = float(model_rows["sigma_y"].iloc[0])
        n_row[(model, "beta")] = float(model_rows["n_rows"].iloc[0])
    table.loc["sigma_y"] = pd.Series(sigma_row)
    table.loc["n_rows"] = pd.Series(n_row)
    return table


def ols_projector(X: torch.Tensor, target_indices: list[int]) -> torch.Tensor | None:
    # A = pinv(X)[target rows], the fixed linear map from outcomes to target betas; None when X is rank-deficient
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D design matrix, got shape {tuple(X.shape)}")
    if X.shape[0] < X.shape[1]:
        return None
    X64 = X.to(dtype=torch.float64)
    svals = torch.linalg.svdvals(X64)
    if float(svals[-1]) <= OLS_SVD_RCOND * float(svals[0]):
        return None
    return torch.linalg.pinv(X64)[target_indices]


def raw_beta_operator(
    group_rows: list[tuple[list[float], list[int], list[float]]],
    spec: RegressionSpec,
    total_options: int,
) -> torch.Tensor | None:
    # Unstandardized B = pinv(X)[target rows] @ W for one regression, so that betas = B @ (option probabilities).
    # `group_rows` = (x, option positions in the flat option vector, weights) per outcome row; None if X is rank-deficient.
    X = torch.tensor([x for x, _, _ in group_rows], dtype=torch.float64)
    A = ols_projector(X, spec.target_indices())
    if A is None:
        return None
    # W is sparse (an outcome row uses a few options), so add each entry's contribution A[:, row] * weight
    # into B's column for that option instead of building W densely
    entry_rows = []
    entry_options = []
    entry_weights = []
    for row_idx, (_, option_idx, weights) in enumerate(group_rows):
        for opt, weight in zip(option_idx, weights, strict=True):
            entry_rows.append(row_idx)
            entry_options.append(opt)
            entry_weights.append(weight)
    entry_rows_t = torch.tensor(entry_rows, dtype=torch.long)
    entry_options_t = torch.tensor(entry_options, dtype=torch.long)
    entry_weights_t = torch.tensor(entry_weights, dtype=torch.float64)
    B = torch.zeros(A.shape[0], total_options, dtype=torch.float64)
    B.index_add_(1, entry_options_t, A[:, entry_rows_t] * entry_weights_t)
    return B


def beta_noise_by_sample_size(
    rows: pd.DataFrame,
    specs: dict[str, RegressionSpec],
    sample_sizes: list[int],
    num_draws: int,
    seed: int,
) -> pd.DataFrame:
    # Monte Carlo: for each sample size, draw that many tasks (data rows, the training sample unit), fit each
    # regression on their outcome rows, and compare the standardized sample betas with the full-data standardized
    # betas (target noise). Signal = mean squared full-data standardized beta. Rank-deficient draws are counted and
    # skipped.
    rng = np.random.default_rng(seed)
    tasks = rows["task"].unique()
    result_rows = []
    for group, spec in specs.items():
        group_rows = rows[rows["group"] == group].reset_index(drop=True)
        X_full = torch.tensor(group_rows[spec.design_columns].to_numpy(dtype=np.float64))
        y_full = torch.tensor(group_rows["y"].to_numpy(dtype=np.float64)) / spec.sigma
        A_full = ols_projector(X_full, spec.target_indices())
        if A_full is None:
            raise ValueError(f"{group}: full-data design is rank-deficient")
        full_betas = A_full @ y_full
        signal = float(full_betas.pow(2).mean())

        # Outcome-row positions of each task in this group (tasks of other groups have none)
        row_idx_by_task = {}
        for task, task_rows in group_rows.groupby("task", sort=False):
            row_idx_by_task[task] = task_rows.index.to_numpy()

        for sample_size in sample_sizes:
            noise_values = []
            rank_deficient = 0
            for _ in range(num_draws):
                drawn = rng.choice(tasks, size=sample_size, replace=False)
                drawn_idx_parts = []
                for task in drawn:
                    if task in row_idx_by_task:
                        drawn_idx_parts.append(row_idx_by_task[task])
                if not drawn_idx_parts:
                    rank_deficient += 1
                    continue
                drawn_idx = torch.tensor(np.concatenate(drawn_idx_parts))
                A_s = ols_projector(X_full[drawn_idx], spec.target_indices())
                if A_s is None:
                    rank_deficient += 1
                    continue
                noise_values.append(float((A_s @ y_full[drawn_idx] - full_betas).pow(2).mean()))
            mean_noise = float(np.mean(noise_values)) if noise_values else float("nan")
            result_rows.append(
                {
                    "model": group,
                    "sample_size": sample_size,
                    "rank_deficient_pct": 100.0 * rank_deficient / num_draws,
                    "std_signal_ms": signal,
                    "std_target_noise_mse": mean_noise,
                    "snr": signal / mean_noise,
                }
            )
    return pd.DataFrame(result_rows)
