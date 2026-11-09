import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display


def fmt_time(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


class TrainingPlotter:
    """Live Jupyter status bar, training/validation KL curves, and summary tables.

    Decoupled from the trainer via composition: receives only step numbers, KL values,
    and evaluation summary dicts. Display handles are created lazily via `init_display()`
    so instantiating a trainer or running VRAM probes does not emit empty figures.
    """

    PALETTE = ["tab:orange", "tab:green", "tab:purple", "tab:red", "tab:brown"]

    def __init__(self, total_steps, eval_sets, val_name, run_name="train"):
        self.run_name = run_name
        self.total_steps = total_steps
        self.val_name = val_name
        self.start_time = time.perf_counter()
        self.train_start_time = None
        self.train_start_step = 0
        self.last_lr = None
        self.eval_specs = [
            {
                "name": s["name"],
                "valid": bool(s["valid"]),
                "n_records": len(s["records"]),
                "n_states": len(s["packed_views"]),
                "has_ols": s["eval_fn"] is not None,
            }
            for s in eval_sets
        ]
        self.ols_names = [sp["name"] for sp in self.eval_specs if sp["has_ols"]]
        self.val_rows = []
        self.latest_ols_df = {}
        self.latest_ols_summary = {}
        self.history = {
            "step": [],
            "step_kl": [],
            "lr": [],
            "val_step": [],
            "eval_kls": {sp["name"]: [] for sp in self.eval_specs},
        }

        self._display_initialized = False
        self.status_handle = None
        self.fig = None
        self.ax_train = None
        self.ax_lr = None
        self.ax_evals = {}
        self.ax_ols = {}
        self.handle = None
        self.table_handle = None
        self.ols_table_handle = None

    def init_display(self, force=False):
        """Create Jupyter display handles once at the start of training (or recreate when `force=True`)."""
        if self._display_initialized and not force:
            return
        self.start_time = time.perf_counter()
        self.train_start_time = None
        self.train_start_step = 0
        self.status_handle = display(
            HTML(f"<div style='font-family: monospace;'>{self.run_name}: starting…</div>"), display_id=True
        )
        n_plots = 2 + len(self.eval_specs) + len(self.ols_names)
        self.fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        self.ax_train = axes[0]
        self.ax_evals = {
            sp["name"]: axes[1 + i] for i, sp in enumerate(self.eval_specs)
        }
        for ax in self.ax_evals.values():
            ax.sharex(self.ax_train)
        self.ax_lr = axes[1 + len(self.eval_specs)]
        self.ax_lr.sharex(self.ax_train)
        self.ax_ols = {
            name: axes[2 + len(self.eval_specs) + j] for j, name in enumerate(self.ols_names)
        }
        self.handle = display(self.fig, display_id=True)
        plt.close(self.fig)
        self.table_handle = display(pd.DataFrame(), display_id=True)
        self.ols_table_handle = (
            display(pd.DataFrame(), display_id=True) if self.ols_names else None
        )
        self._display_initialized = True
        self.redraw(0, eval_updated=True)

    def state_dict(self):
        return {
            "history": self.history,
            "val_rows": self.val_rows,
            "latest_ols_df": self.latest_ols_df,
            "latest_ols_summary": self.latest_ols_summary,
        }

    def load_state_dict(self, state, step):
        self.init_display()
        self.history = state["history"]
        if "lr" not in self.history:
            self.history["lr"] = [np.nan] * len(self.history["step"])
        self.val_rows = state["val_rows"]
        self.latest_ols_df = state["latest_ols_df"]
        self.latest_ols_summary = state["latest_ols_summary"]
        self.redraw(step)
        self.redraw_table()

    def update_status(self, step, epoch=0, status="training", lr=None):
        self.init_display()
        if lr is not None:
            self.last_lr = lr
        now = time.perf_counter()
        elapsed = now - self.start_time
        pct = 100.0 * step / max(1, self.total_steps)
        if self.train_start_time is not None and step > self.train_start_step:
            train_elapsed = now - self.train_start_time
            steps_done = step - self.train_start_step
            rate = steps_done / max(1e-6, train_elapsed)
            eta = (self.total_steps - step) / max(1e-6, rate)
            rate_str = f"{rate:.2f} step/s"
            eta_str = fmt_time(eta)
        else:
            rate_str = "-- step/s"
            eta_str = "--:--"

        kl_str = f"{self.history['step_kl'][-1]:.4f}" if self.history["step_kl"] else "—"
        lr_str = f"{self.last_lr:.2e}" if self.last_lr is not None else "—"
        bar_color = "#f59e0b" if status.startswith("eval") else "#2563eb"
        html = (
            f"<div style='font-family: monospace; font-size: 13px; padding: 8px 10px; "
            f"border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; color: #0f172a; margin: 4px 0;'>"
            f"<div style='margin-bottom: 5px;'>"
            f"<b>{self.run_name}</b> &nbsp;|&nbsp; Epoch <b>{epoch}</b> &nbsp;|&nbsp; "
            f"Step <b>{step:,} / {self.total_steps:,}</b> (<b>{pct:.2f}%</b>) &nbsp;|&nbsp; "
            f"Elapsed: {fmt_time(elapsed)} &lt; ETA: {eta_str} ({rate_str}) &nbsp;|&nbsp; "
            f"train_kl: <b>{kl_str}</b> &nbsp;|&nbsp; lr: {lr_str} &nbsp;|&nbsp; "
            f"<span style='color: {bar_color}; font-weight: bold;'>[{status}]</span>"
            f"</div>"
            f"<div style='width: 100%; height: 12px; background: #e2e8f0; border-radius: 4px; overflow: hidden;'>"
            f"<div style='width: {min(100.0, max(0.2 if step > 0 else 0.0, pct)):.3f}%; height: 100%; background: {bar_color};'></div>"
            f"</div></div>"
        )
        if self.status_handle is not None:
            self.status_handle.update(HTML(html))

    def log_step(self, step, step_kls, epoch=1, lr=None):
        self.init_display()
        if self.train_start_time is None:
            self.train_start_time = time.perf_counter() - 0.25
            self.train_start_step = step - 1
        step_kl = sum(step_kls) / len(step_kls)
        self.history["step"].append(step)
        self.history["step_kl"].append(step_kl)
        self.history["lr"].append(float(lr) if lr is not None else np.nan)
        self.redraw(step, eval_updated=False)
        self.update_status(step, epoch=epoch, status="training", lr=lr)
        return step_kl

    def log_eval(self, step, eval_summaries, eval_extra_dfs, epoch=0):
        self.init_display()
        self.history["val_step"].append(step)
        train_kl = self.history["step_kl"][-1] if self.history["step_kl"] else np.nan
        row = {
            "epoch": epoch,
            "step": step,
            "train_kl": train_kl,
        }
        for sp in self.eval_specs:
            name = sp["name"]
            if name not in eval_summaries:
                self.history["eval_kls"][name].append(np.nan)
                row[f"{name}_kl"] = np.nan
                if sp["has_ols"]:
                    for k in ("OLS MAPE (%)", "OLS MAE", "OLS MSE", "Spearman rho"):
                        row[f"{name}_{k}"] = np.nan
                continue
            s = eval_summaries[name]
            kl = float(s["dist_kl"])
            self.history["eval_kls"][name].append(kl)
            row[f"{name}_kl"] = kl
            if sp["has_ols"]:
                for k in ("OLS MAPE (%)", "OLS MAE", "OLS MSE", "Spearman rho"):
                    row[f"{name}_{k}"] = float(s[k])
                self.latest_ols_summary[name] = s
                if eval_extra_dfs[name] is None:
                    raise ValueError(f"Expected non-None extra_df for OLS evaluation set {name!r}")
                self.latest_ols_df[name] = eval_extra_dfs[name]

        self.val_rows.append(row)
        self.redraw(step, eval_updated=True)
        self.redraw_table()

    def redraw_table(self):
        self.init_display()
        if not self.val_rows:
            return
        df = pd.DataFrame(self.val_rows)
        val_col = f"{self.val_name}_kl"
        best_idx = df[val_col].idxmin() if df[val_col].notna().any() else None
        fmt = {c: "{:.4f}" for c in df.columns if c not in ("epoch", "step")}
        for c in df.columns:
            if "MAPE" in c:
                fmt[c] = "{:.2f}"
        styled = (
            df.style
            .apply(
                lambda r: [
                    "background-color: #d4edda; color: #155724; font-weight: bold"
                    if best_idx is not None and r.name == best_idx else ""
                    for _ in r
                ],
                axis=1,
            )
            .format(fmt, na_rep="—")
        )
        if self.table_handle is not None:
            self.table_handle.update(styled)

        if self.ols_table_handle is not None and self.latest_ols_df:
            name = self.ols_names[0]
            if name in self.latest_ols_df:
                ols_df = self.latest_ols_df[name].copy()
                self.ols_table_handle.update(ols_df.style.format("{:.4f}"))

    def redraw(self, step, *, eval_updated=True):
        self.init_display()
        self.ax_train.clear()
        if self.history["step"]:
            self.ax_train.plot(
                self.history["step"],
                self.history["step_kl"],
                color="tab:blue",
                linewidth=1.5,
                label="train KL",
            )
            self.ax_train.set_yscale("log")
            self.ax_train.legend(loc="upper right")
        self.ax_train.set_ylabel("Train KL (bits)")
        self.ax_train.set_title(f"Training KL — step {step}/{self.total_steps}")
        self.ax_train.set_xlabel("Step")

        self.ax_lr.clear()
        lr_pts = [
            (st, lr_val)
            for st, lr_val in zip(self.history["step"], self.history["lr"])
            if math.isfinite(lr_val)
        ]
        if lr_pts:
            lr_xs, lr_ys = zip(*lr_pts)
            self.ax_lr.plot(
                lr_xs,
                lr_ys,
                color="tab:red",
                linewidth=1.8,
                label=f"lr (cur={lr_ys[-1]:.2e})",
            )
            self.ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            self.ax_lr.legend(loc="upper right")
            self.ax_lr.set_title(f"Learning Rate — {lr_ys[-1]:.2e}")
        else:
            self.ax_lr.set_title("Learning Rate")
        self.ax_lr.set_ylabel("Learning Rate")
        self.ax_lr.set_xlabel("Step")

        if eval_updated:
            for i, sp in enumerate(self.eval_specs):
                name = sp["name"]
                ax = self.ax_evals[name]
                ax.clear()
                kls = self.history["eval_kls"][name]
                color = self.PALETTE[i % len(self.PALETTE)]
                pts = [
                    (st, kl)
                    for st, kl in zip(self.history["val_step"], kls)
                    if math.isfinite(kl)
                ]
                if pts:
                    xs, ys = zip(*pts)
                    best_kl = min(ys)
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        marker="o",
                        linewidth=2,
                        label=f"{name} KL (best={best_kl:.4f} bits, last={ys[-1]:.4f})",
                    )
                    ax.set_yscale("log")
                    ax.legend(loc="upper right")
                tag = " [VALID — saves best]" if sp["valid"] else ""
                ax.set_ylabel(f"{name} KL (bits)")
                ax.set_xlabel("Step")
                ax.set_title(f"{name}{tag} ({sp['n_records']:,} records, {sp['n_states']:,} states)")

            for name in self.ols_names:
                ax = self.ax_ols[name]
                ax.clear()
                if name in self.latest_ols_df:
                    bdf = self.latest_ols_df[name]
                    s = self.latest_ols_summary[name]
                    ols_step = [
                        st
                        for st, kl in zip(self.history["val_step"], self.history["eval_kls"][name])
                        if math.isfinite(kl)
                    ][-1]
                    arms = list(bdf.index)
                    x = np.arange(len(arms))
                    width = 0.36
                    ax.bar(x - width / 2, bdf["human"].values, width, label="Human RCT", color="tab:gray")
                    ax.bar(x + width / 2, bdf["model"].values, width, label=f"Model (step {ols_step})", color="tab:green")
                    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
                    ax.set_xticks(x)
                    ax.set_xticklabels(arms, rotation=15, ha="right")
                    ax.set_ylabel("OLS Treatment Effect (vs. Positive)")
                    mape = s["OLS MAPE (%)"]
                    mae = s["OLS MAE"]
                    rho = s["Spearman rho"]
                    ax.set_title(
                        f"{name} Current Model OLS Betas — MAPE={mape:.2f}%, MAE={mae:.4f}, Spearman rho={rho:.3f}"
                    )
                    ax.legend(loc="upper right")

            self.fig.tight_layout()

        if self.handle is not None:
            self.handle.update(self.fig)
