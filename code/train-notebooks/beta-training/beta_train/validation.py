"""Population validation on any `beta_ols` dataset: model predictions on every task, then KL and population betas.

The betas use `beta_ols.evaluation`, i.e. the dataset module's own coding (same as its human replication and the
training loss). Two validators with the same interface (`.validate(model) -> (metrics, beta_df)`):
- PopulationValidator ("fused"): merges the LoRA into the weights, swaps in fused kernels and CUDA graphs, and runs
  respondents that share a persona state together. Fast per token; each call merges, offloads, and recaptures.
- TrainingPathValidator ("training_path"): the training forward (`beta_train.step.predict_chunk`) in eval mode with
  the LoRA unmerged, over the dataset's training items. No merge, offload, or graph capture; memory set by `eval_chunk`.
Build one per validation dataset once; call `.validate(model)` any time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import data.base
import kev.cuda_graphs
import kev.fused_qwen35
import kev.shared_prefix
from data.base import build_long_records, fused_eval_context, group_records_by_state, make_view, predict_probs_grouped
from training import free_gpu

import beta_ols.core
import beta_ols.evaluation
from beta_ols.core import iter_respondents
from beta_ols.evaluation import build_population_eval, compare_population_betas, flat_probs_from_predictions

import beta_train.items
import beta_train.step
from beta_train.cache import dataframe_fingerprint, load_or_compute, source_fingerprint
from beta_train.model_state import reset_trainable
from beta_train.batching import split_into_chunks
from beta_train.step import predict_chunk

# This file's own module, so its source is part of the cache keys it writes
THIS_MODULE = sys.modules[__name__]

# Code a step-0 result depends on beyond the encoding: the shared-prefix forward and the fused eval kernels / graphs
FORWARD_MODULES = (kev.shared_prefix, kev.fused_qwen35, kev.cuda_graphs)


METRIC_NAMES = ("kl_loss", "beta_mse", "beta_mae", "beta_mape_pct", "spearman_rho")


def human_beta_series(name: str, eval_ops: dict) -> pd.Series:
    # Human population betas with their labels (all regressions, in spec order)
    values = []
    labels = []
    for op in eval_ops["operators"]:
        spec = eval_ops["specs"][op["group"]]
        values.extend(eval_ops["human_betas"][op["group"]].tolist())
        labels.extend(spec.target_labels)
    return pd.Series(values, index=labels, name=f"{name}_human_pop_beta")


def check_training_mode_restored(model, was_training: bool) -> None:
    # After validation the model must be back in its previous mode with every trainable parameter intact
    if model.training != was_training:
        raise AssertionError(f"Expected model.training={was_training} after validation, got {model.training}")
    if any((not p.requires_grad) or p.numel() == 0 for p in model.trainable_parameters()):
        raise AssertionError("Trainable parameters were not cleanly restored after validation")


class PopulationValidator:
    """Fused validation of one dataset: holds its state-grouped inference bundle and population OLS operators."""

    KIND = "fused"

    def __init__(self, dataset_module, choice_df: pd.DataFrame, specs: dict, max_questions: int, max_chars: int):
        # max_questions / max_chars: per-GPU-batch budgets for inference (questions, padded characters); whichever is
        # hit first closes a batch. Lower them to avoid OOM on bigger models, raise them for fewer, fuller batches.
        for budget_name, budget in (("max_questions", max_questions), ("max_chars", max_chars)):
            if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
                raise ValueError(f"{budget_name} must be a positive integer, got {budget!r}")
        t_start = time.perf_counter()
        self.name = dataset_module.NAME
        self.dataset_module = dataset_module
        self.choice_fingerprint = dataframe_fingerprint(choice_df)
        self.max_questions = max_questions
        self.max_chars = max_chars

        # 1. State-grouped bundle: respondents sharing a persona state are run together (fast inference)
        records = build_long_records(
            choice_df, persona_cols=dataset_module.PERSONA_COLS, include_country=False, year=dataset_module.NAME
        )
        self.bundle = group_records_by_state(records, view_fn=make_view, name=f"{self.name}_validation", valid=True)

        # 2. Human one-hot choice per record (bundle order), for the KL
        self.human_choices = []
        for rec_idx, rec in enumerate(self.bundle["records"]):
            h_dist = [float(v) for v in rec["human_dist"]]
            if sorted(h_dist) != [0.0] * (len(h_dist) - 1) + [1.0]:
                raise ValueError(f"{self.name} record {rec_idx}: expected a one-hot human choice, got {h_dist}")
            self.human_choices.append(h_dist.index(1.0))

        # 3. Population OLS operators from the module's coding; records matched to tasks by (CASEID, qid)
        self.eval_ops = build_population_eval(dataset_module, choice_df, specs, self.bundle["records"])
        self.prep_sec = time.perf_counter() - t_start

    def describe(self) -> dict:
        return {
            "validator": self.name,
            "kind": self.KIND,
            "tasks": len(self.bundle["records"]),
            "packed_persona_states": len(self.bundle["packed_views"]),
            "regressions": ", ".join(self.eval_ops["specs"]),
            "max_questions": self.max_questions,
            "max_chars": self.max_chars,
            "prep_sec": round(self.prep_sec, 3),
        }

    def human_beta_series(self) -> pd.Series:
        return human_beta_series(self.name, self.eval_ops)

    @torch.no_grad()
    def validate(self, model) -> tuple[dict[str, float], pd.DataFrame]:
        # Inference in fused eval mode (training mode and weights restored afterwards), then KL + population betas
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        was_training = model.training
        model.eval()
        free_gpu()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16), fused_eval_context(model, eval_dtype=torch.bfloat16):
                preds, stats = predict_probs_grouped(
                    model,
                    self.bundle,
                    max_questions=self.max_questions,
                    max_chars=self.max_chars,
                    show_progress=True,
                    return_stats=True,
                )
        finally:
            if was_training:
                model.train()
            free_gpu()
        check_training_mode_restored(model, was_training)
        if len(preds) != len(self.human_choices):
            raise ValueError(f"Expected {len(self.human_choices)} predictions, got {len(preds)}")

        # 1. KL(y || p) per task for one-hot y = -log p(chosen); mean over tasks
        chosen_log_probs = []
        for pred, chosen in zip(preds, self.human_choices, strict=True):
            chosen_log_probs.append(float(np.log(max(float(pred[chosen]), 1e-12))))
        kl_loss = -float(np.mean(chosen_log_probs))

        # 2. Model vs human population betas
        probs_flat = flat_probs_from_predictions(self.eval_ops, preds)
        beta_metrics, beta_df = compare_population_betas(self.eval_ops, probs_flat)
        eval_wall_sec = time.perf_counter() - t_start

        metrics = {
            "kl_loss": kl_loss,
            "beta_mse": beta_metrics["beta_mse"],
            "beta_mae": beta_metrics["beta_mae"],
            "beta_mape_pct": beta_metrics["beta_mape_pct"],
            "spearman_rho": beta_metrics["spearman_rho"],
            "eval_prep_sec": float(stats["prep_sec"]),
            "eval_gpu_sec": float(stats["gpu_sec"]),
            "eval_wall_sec": eval_wall_sec,
            "eval_ms_per_task": (eval_wall_sec * 1000.0) / len(preds),
        }
        return metrics, beta_df


class TrainingPathValidator:
    """Validation of one dataset through the training forward (LoRA unmerged, eval mode, no gradients)."""

    KIND = "training_path"

    def __init__(self, dataset_module, choice_df: pd.DataFrame, specs: dict, items: list[dict], eval_chunk: int):
        # items: this dataset's items (`beta_train.items`, one per data row), in the module's order.
        # eval_chunk: items (rows) per forward pass; no gradients are kept, so it can be far above the training micro-chunk.
        if not isinstance(eval_chunk, int) or isinstance(eval_chunk, bool) or eval_chunk < 1:
            raise ValueError(f"eval_chunk must be a positive integer, got {eval_chunk!r}")
        t_start = time.perf_counter()
        self.name = dataset_module.NAME
        self.dataset_module = dataset_module
        self.choice_fingerprint = dataframe_fingerprint(choice_df)
        self.items = items
        self.eval_chunk = eval_chunk

        # 1. The items must be this dataset's rows in the module's order (that order is the flat option vector)
        row_keys = []
        task_records = []
        for caseid, r_df in iter_respondents(choice_df):
            for qid in r_df["qid"]:
                row_keys.append((caseid, [str(qid)]))
                task_records.append({"CASEID": caseid, "qid": str(qid)})
        item_keys = [(item["caseid"], item["qids"]) for item in items]
        if item_keys != row_keys:
            raise ValueError(f"{self.name}: items are not this dataset's rows in module order")
        for item in items:
            if item["dataset"] != self.name:
                raise ValueError(f"{self.name}: got an item from dataset {item['dataset']!r}")

        # 2. Human choice per task (item order), for the KL
        self.human_choices = []
        for item in items:
            for task_y in torch.split(item["y_flat"], item["num_options"]):
                self.human_choices.append(int(torch.argmax(task_y)))

        # 3. Population OLS operators; tasks are listed in module order, so every task maps to its own position
        self.eval_ops = build_population_eval(dataset_module, choice_df, specs, task_records)
        self.prep_sec = time.perf_counter() - t_start

    def describe(self) -> dict:
        return {
            "validator": self.name,
            "kind": self.KIND,
            "tasks": len(self.human_choices),
            "items (rows)": len(self.items),
            "regressions": ", ".join(self.eval_ops["specs"]),
            "eval_chunk": self.eval_chunk,
            "prep_sec": round(self.prep_sec, 3),
        }

    def human_beta_series(self) -> pd.Series:
        return human_beta_series(self.name, self.eval_ops)

    @torch.no_grad()
    def validate(self, model) -> tuple[dict[str, float], pd.DataFrame]:
        # Training forward in eval mode (no dropout), chunk by chunk; then KL + population betas
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        was_training = model.training
        model.eval()
        preds = []
        chunks = split_into_chunks(self.items, self.eval_chunk)
        try:
            # One bar over the forward chunks (cleared when validation ends), like the fused validator's batch bar
            for chunk in tqdm(chunks, desc=f"validating {self.name} (training path)", unit="chunk", leave=False):
                flat_logits = predict_chunk(model, chunk)
                chunk_num_options = []
                for item in chunk:
                    chunk_num_options.extend(item["num_options"])
                for task_logits in torch.split(flat_logits, chunk_num_options):
                    preds.append(torch.softmax(task_logits, dim=-1).cpu().numpy())
            torch.cuda.synchronize()
        finally:
            if was_training:
                model.train()
            free_gpu()
        gpu_sec = time.perf_counter() - t_start
        check_training_mode_restored(model, was_training)
        if len(preds) != len(self.human_choices):
            raise ValueError(f"Expected {len(self.human_choices)} predictions, got {len(preds)}")

        # 1. KL(y || p) per task for one-hot y = -log p(chosen); mean over tasks
        chosen_log_probs = []
        for pred, chosen in zip(preds, self.human_choices, strict=True):
            chosen_log_probs.append(float(np.log(max(float(pred[chosen]), 1e-12))))
        kl_loss = -float(np.mean(chosen_log_probs))

        # 2. Model vs human population betas
        probs_flat = flat_probs_from_predictions(self.eval_ops, preds)
        beta_metrics, beta_df = compare_population_betas(self.eval_ops, probs_flat)
        eval_wall_sec = time.perf_counter() - t_start

        metrics = {
            "kl_loss": kl_loss,
            "beta_mse": beta_metrics["beta_mse"],
            "beta_mae": beta_metrics["beta_mae"],
            "beta_mape_pct": beta_metrics["beta_mape_pct"],
            "spearman_rho": beta_metrics["spearman_rho"],
            "eval_prep_sec": 0.0,
            "eval_gpu_sec": gpu_sec,
            "eval_wall_sec": eval_wall_sec,
            "eval_ms_per_task": (eval_wall_sec * 1000.0) / len(preds),
        }
        return metrics, beta_df


def validate_all(model, validators: dict) -> dict[str, tuple[dict[str, float], pd.DataFrame]]:
    # {validator name: (metrics, beta_df)} for every validator
    results = {}
    for name, validator in validators.items():
        results[name] = validator.validate(model)
    return results


def validate_all_cached(
    model,
    validators: dict,
    init_snapshot: dict,
    model_key: dict,
    cache_dir: Path,
) -> tuple[dict[str, tuple[dict[str, float], pd.DataFrame]], pd.DataFrame]:
    # Untrained (step-0) validation, cached on disk per validator. The model is first reset to `init_snapshot`, so the
    # result belongs to the untrained weights. The key covers `model_key` (declared in the notebook: base checkpoint,
    # LoRA rank, dtype, head temperature), the tokenizer and encoding settings, the validation rows and persona
    # columns, and the code that computes the result (encoding, forward, OLS).
    # Returns ({validator: (metrics, beta_df)}, one row per validator saying whether it was loaded or computed).
    reset_trainable(model, init_snapshot)
    results = {}
    report_rows = []
    for name, validator in validators.items():
        # Code the result depends on: OLS + encoding + forward for both kinds; the training-path validator also runs
        # the training forward (step) on training items (items), the fused one does not
        code_modules = [
            validator.dataset_module,
            beta_ols.core,
            beta_ols.evaluation,
            THIS_MODULE,
            *beta_train.items.ENCODING_MODULES,
            *FORWARD_MODULES,
        ]
        if validator.KIND == "training_path":
            code_modules.extend([beta_train.items, beta_train.step])
        key = {
            "kind": "step0_validation",
            "validator": name,
            "validator_kind": validator.KIND,
            "model": model_key,
            "tokenizer": str(model.tok.name_or_path),
            "option_isolation": bool(model.encoder.option_isolation),
            "persona_cols": list(validator.dataset_module.PERSONA_COLS),
            "choice_rows": validator.choice_fingerprint,
            "code": source_fingerprint(*code_modules),
        }

        def compute_one_validator():
            return validator.validate(model)

        results[name], loaded = load_or_compute(cache_dir, f"step0_{name}", key, compute_one_validator)
        report_rows.append({"validator": name, "source": "cache" if loaded else "computed"})
    return results, pd.DataFrame(report_rows).set_index("validator")


def validation_metrics_frame(validation: dict[str, tuple[dict[str, float], pd.DataFrame]]) -> pd.DataFrame:
    # One column per validator, one row per metric
    columns = {}
    for name, (metrics, _) in validation.items():
        columns[name] = pd.Series(metrics)
    return pd.DataFrame(columns)
