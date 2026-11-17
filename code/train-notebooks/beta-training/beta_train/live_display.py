"""Live training display: one validation table and one figure, each redrawn in place (no growing output)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from beta_train.validation import METRIC_NAMES

# Shading of the best row's cells (semi-transparent, so it reads in light and dark notebook themes)
BEST_ROW_STYLE = "background-color: rgba(0, 160, 0, 0.18)"
# The one cell that picked the checkpoint (best metric at the best step): darker shading and bold text
BEST_CELL_STYLE = "background-color: rgba(0, 160, 0, 0.40); font-weight: bold"


def table_styles(val_df: pd.DataFrame, styles: pd.DataFrame) -> pd.DataFrame:
    # Styler.apply(axis=None) callback: the CSS of every cell, computed in update_table
    if not styles.index.equals(val_df.index) or not styles.columns.equals(val_df.columns):
        raise ValueError("Cell styles do not match the validation table")
    return styles


def step_label_style(step_labels: pd.Series, styles: pd.Series) -> list[str]:
    # Styler.apply_index callback: the CSS of every step label (the table's index, passed as a Series of labels),
    # computed in update_table
    if list(step_labels) != list(styles.index):
        raise ValueError("Step label styles do not match the validation table")
    return list(styles)


# Per-step gradient-split columns (beta_train.step.gradient_split_stats), and the lines of the gradient-norm panel
GRADIENT_SPLIT_STEP_COLUMNS = [
    "grad_ce_norm",
    "grad_beta_norm",
    "grad_lambda_beta_norm",
    "grad_total_norm",
    "grad_cos_ce_lambda_beta",
    "grad_ce_projection",
    "grad_lambda_beta_projection",
]
GRADIENT_NORM_LINES = [
    ("grad_ce_norm", "|CE gradient|"),
    ("grad_beta_norm", "|beta gradient|"),
    ("grad_lambda_beta_norm", "|lambda x beta gradient|"),
    ("grad_total_norm", "|total gradient|"),
]

# (metric, title) for the six validation panels; every validator gets one line per panel
VALIDATION_PANELS = [
    ("val_loss", "validation loss (KL + lambda x std beta MSE)"),
    ("kl_loss", "validation KL loss"),
    ("std_beta_mse", "validation std beta MSE (training units)"),
    ("beta_mse", "validation beta MSE"),
    ("beta_mae", "validation beta MAE"),
    ("beta_mape_pct", "validation beta MAPE (%)"),
]


def make_val_row(
    step: int,
    epoch: int,
    rows_seen: int,
    train_loss: float,
    validator_names: list[str],
    validation: dict[str, tuple[dict[str, float], pd.DataFrame]],
    losses: dict[str, float],
    lr: float,
    train_elapsed_sec: float,
) -> dict:
    # Column order: ids, train loss, each validator's val_loss / std beta MSE and metrics (up to spearman_rho), lr,
    # eval times, train time last. Every validator has its columns; one that did not run at this step (not in
    # `validation`) gets NaN. `losses` = beta_train.runs.validation_losses(...)
    nan = float("nan")
    row = {"step": step, "epoch": epoch, "rows_seen": rows_seen, "train_loss": train_loss}
    for name in validator_names:
        if name in validation:
            metrics = validation[name][0]
            row[f"{name}/val_loss"] = losses[f"{name}/val_loss"]
            row[f"{name}/std_beta_mse"] = losses[f"{name}/std_beta_mse"]
            for metric in METRIC_NAMES:
                row[f"{name}/{metric}"] = metrics[metric]
        else:
            row[f"{name}/val_loss"] = nan
            row[f"{name}/std_beta_mse"] = nan
            for metric in METRIC_NAMES:
                row[f"{name}/{metric}"] = nan
    row["lr"] = lr
    for name in validator_names:
        if name in validation:
            row[f"{name}/eval_wall_sec"] = validation[name][0]["eval_wall_sec"]
        else:
            row[f"{name}/eval_wall_sec"] = nan
    row["train_elapsed_sec"] = train_elapsed_sec
    return row


class LiveDisplay:
    """The validation table and a 2x4 figure: train loss by dataset, 6 validation metrics, learning rate.
    With gradient_split, a third row shows every step's gradient split (CE vs lambda x beta)."""

    def __init__(self, validator_names: list[str], steps_per_epoch: int, rolling_steps: int, gradient_split: bool):
        if not isinstance(gradient_split, bool):
            raise TypeError(f"gradient_split must be a bool, got {gradient_split!r}")
        self.validator_names = list(validator_names)
        self.steps_per_epoch = steps_per_epoch
        self.rolling_steps = rolling_steps
        self.gradient_split = gradient_split
        self.table_handle = display(pd.DataFrame(), display_id=True)
        if gradient_split:
            self.fig, self.axes = plt.subplots(3, 4, figsize=(20, 10.5))
        else:
            self.fig, self.axes = plt.subplots(2, 4, figsize=(20, 7))
        plt.close(self.fig)
        self.plot_handle = display(self.fig, display_id=True)

    def update_table(self, val_rows: list[dict], best: dict) -> None:
        # best: the run's best validation so far ({metric, mode, step, value}); its whole row is shaded green and the
        # cell that picked it (best metric at the best step) is darker and bold
        val_df = pd.DataFrame(val_rows).set_index("step").round(5)
        if best["step"] not in val_df.index:
            raise KeyError(f"Best step {best['step']} is not a validation row")
        cell_styles = pd.DataFrame("", index=val_df.index, columns=val_df.columns)
        cell_styles.loc[best["step"], :] = BEST_ROW_STYLE
        if best["metric"] not in val_df.columns:
            raise KeyError(f"Best metric {best['metric']!r} is not a column of the validation table")
        cell_styles.loc[best["step"], best["metric"]] = BEST_CELL_STYLE
        step_label_styles = pd.Series("", index=val_df.index)
        step_label_styles.loc[best["step"]] = BEST_ROW_STYLE
        styled_df = val_df.style.apply(table_styles, styles=cell_styles, axis=None)
        styled_df = styled_df.apply_index(step_label_style, styles=step_label_styles)
        styled_df = styled_df.format(precision=5, na_rep="NaN")
        self.table_handle.update(styled_df)

    def redraw(self, step_rows: list[dict], val_rows: list[dict], best: dict) -> None:
        # best: the run's best validation so far ({metric, mode, step, value}); marked on every validation panel
        for ax in self.axes.flat:
            ax.clear()

        # Train total loss, one rolling line per training dataset
        train_ax = self.axes[0, 0]
        if step_rows:
            step_df = pd.DataFrame(step_rows)
            for dataset_name, dataset_steps in step_df.groupby("dataset"):
                rolling_loss = dataset_steps["total_loss"].rolling(self.rolling_steps, min_periods=1).mean()
                train_ax.plot(dataset_steps["step"], rolling_loss, label=f"{dataset_name} (rolling {self.rolling_steps})")
            train_ax.legend(fontsize=8)
            self.axes[1, 3].plot(step_df["step"], step_df["lr"])
        train_ax.set_title("train total loss by dataset")
        self.axes[1, 3].set_title("learning rate")

        # Validation metrics, one line per validator
        panel_axes = [self.axes[0, 1], self.axes[0, 2], self.axes[0, 3], self.axes[1, 0], self.axes[1, 1], self.axes[1, 2]]
        for ax, (metric, title) in zip(panel_axes, VALIDATION_PANELS, strict=True):
            for name in self.validator_names:
                # Only the steps this validator ran at (validators can have different eval intervals)
                validator_steps = []
                validator_values = []
                for row in val_rows:
                    value = row[f"{name}/{metric}"]
                    if not math.isnan(value):
                        validator_steps.append(row["step"])
                        validator_values.append(value)
                ax.plot(validator_steps, validator_values, marker="o", label=name)
            ax.axvline(best["step"], color="green", linestyle=":", linewidth=1.2, label=f"best {best['metric']} (step {best['step']})")
            ax.set_title(title)
            ax.legend(fontsize=8)

        if self.gradient_split:
            self.draw_gradient_split(step_rows)

        # Dashed lines at epoch boundaries
        last_step = step_rows[-1]["step"] if step_rows else 0
        for ax in self.axes.flat:
            ax.set_xlabel("step")
            for boundary in range(self.steps_per_epoch, last_step + 1, self.steps_per_epoch):
                ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
        self.fig.tight_layout()
        self.plot_handle.update(self.fig)

    def draw_gradient_split(self, step_rows: list[dict]) -> None:
        # Third row, one point per step (steps alternate between training datasets, so lines zigzag between them):
        # the gradient norms together (log scale), CE vs lambda x beta cosine, each part's projection onto the total
        # gradient, and the lambda x beta / CE norm ratio
        norms_ax = self.axes[2, 0]
        cosine_ax = self.axes[2, 1]
        projection_ax = self.axes[2, 2]
        ratio_ax = self.axes[2, 3]
        if step_rows:
            step_df = pd.DataFrame(step_rows)
            for column in GRADIENT_SPLIT_STEP_COLUMNS:
                if column not in step_df.columns:
                    raise KeyError(f"gradient_split is on but steps have no {column!r} column")
            for column, label in GRADIENT_NORM_LINES:
                norms_ax.plot(step_df["step"], step_df[column], linewidth=0.8, label=label)
            norms_ax.set_yscale("log")
            norms_ax.legend(fontsize=8)
            cosine_ax.plot(step_df["step"], step_df["grad_cos_ce_lambda_beta"], linewidth=0.8)
            cosine_ax.axhline(0.0, color="gray", linewidth=0.8)
            cosine_ax.set_ylim(-1.05, 1.05)
            projection_ax.plot(step_df["step"], step_df["grad_ce_projection"], linewidth=0.8, label="CE")
            projection_ax.plot(step_df["step"], step_df["grad_lambda_beta_projection"], linewidth=0.8, label="lambda x beta")
            projection_ax.axhline(0.0, color="gray", linewidth=0.8)
            projection_ax.legend(fontsize=8)
            ratio = step_df["grad_lambda_beta_norm"] / step_df["grad_ce_norm"]
            ratio_ax.plot(step_df["step"], ratio, linewidth=0.8)
        norms_ax.set_title("gradient norms (before clipping)")
        cosine_ax.set_title("cosine(CE gradient, lambda x beta gradient)")
        projection_ax.set_title("projection onto the total gradient (sums to total)")
        ratio_ax.set_title("|lambda x beta gradient| / |CE gradient|")
