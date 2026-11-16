"""Live training display: one validation table and one figure, each redrawn in place (no growing output)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from beta_train.validation import METRIC_NAMES

# (metric, title) for the four validation panels; every validator gets one line per panel
VALIDATION_PANELS = [
    ("kl_loss", "validation KL loss"),
    ("beta_mse", "validation beta MSE"),
    ("beta_mae", "validation beta MAE"),
    ("beta_mape_pct", "validation beta MAPE (%)"),
]


def make_val_row(
    step: int,
    epoch: int,
    respondents_seen: int,
    train_loss: float,
    validation: dict[str, tuple[dict[str, float], pd.DataFrame]],
    lr: float,
    train_elapsed_sec: float,
) -> dict:
    # Column order: ids, train loss, each validator's metrics (up to spearman_rho), lr, eval times, train time last
    row = {"step": step, "epoch": epoch, "respondents_seen": respondents_seen, "train_loss": train_loss}
    for name, (metrics, _) in validation.items():
        for metric in METRIC_NAMES:
            row[f"{name}/{metric}"] = metrics[metric]
    row["lr"] = lr
    for name, (metrics, _) in validation.items():
        row[f"{name}/eval_wall_sec"] = metrics["eval_wall_sec"]
    row["train_elapsed_sec"] = train_elapsed_sec
    return row


class LiveDisplay:
    """The validation table and a 2x3 figure: train loss by dataset, 4 validation metrics, learning rate."""

    def __init__(self, validator_names: list[str], steps_per_epoch: int, rolling_steps: int):
        self.validator_names = list(validator_names)
        self.steps_per_epoch = steps_per_epoch
        self.rolling_steps = rolling_steps
        self.table_handle = display(pd.DataFrame(), display_id=True)
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 7))
        plt.close(self.fig)
        self.plot_handle = display(self.fig, display_id=True)

    def update_table(self, val_rows: list[dict]) -> None:
        self.table_handle.update(pd.DataFrame(val_rows).set_index("step").round(5))

    def redraw(self, step_rows: list[dict], val_rows: list[dict]) -> None:
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
            self.axes[1, 2].plot(step_df["step"], step_df["lr"])
        train_ax.set_title("train total loss by dataset")
        self.axes[1, 2].set_title("learning rate")

        # Validation metrics, one line per validator
        panel_axes = [self.axes[0, 1], self.axes[0, 2], self.axes[1, 0], self.axes[1, 1]]
        val_steps = [row["step"] for row in val_rows]
        for ax, (metric, title) in zip(panel_axes, VALIDATION_PANELS, strict=True):
            for name in self.validator_names:
                ax.plot(val_steps, [row[f"{name}/{metric}"] for row in val_rows], marker="o", label=name)
            ax.set_title(title)
            ax.legend(fontsize=8)

        # Dashed lines at epoch boundaries
        last_step = step_rows[-1]["step"] if step_rows else 0
        for ax in self.axes.flat:
            ax.set_xlabel("step")
            for boundary in range(self.steps_per_epoch, last_step + 1, self.steps_per_epoch):
                ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
        self.fig.tight_layout()
        self.plot_handle.update(self.fig)
