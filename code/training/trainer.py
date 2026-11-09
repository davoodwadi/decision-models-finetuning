import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import sys
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch

from data.base import (
    DEFAULT_CACHE_DIR,
    _record_chars,
    evaluate_model,
    fused_eval_context,
    group_train_records,
    make_grouped_view,
    release_cuda_graphs,
)
from models.kev_pointer import Checkpoint, KevPointerModel, LoadOptions
from training.plotter import TrainingPlotter

EVAL0_CACHE_VERSION = 2


def free_gpu():
    """Clear Python exception frames, run GC, and release cached CUDA blocks."""
    if hasattr(sys, "last_traceback") and sys.last_traceback is not None:
        traceback.clear_frames(sys.last_traceback)
        sys.last_traceback = None
    for attr in ("last_value", "last_type", "last_exc"):
        if hasattr(sys, attr):
            setattr(sys, attr, None)
    gc.collect()
    torch.cuda.empty_cache()


def save_checkpoint(model, save_dir, name, extra, *, trainer_state=None):
    """Atomically write checkpoint to `save_dir / name` via a temporary directory."""
    save_dir = Path(save_dir)
    folder, tmp, old = save_dir / name, save_dir / f"{name}-tmp", save_dir / f"{name}-old"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    model.save(tmp, extra=extra)
    (tmp / "metrics.json").write_text(json.dumps(extra, indent=2) + "\n")
    if trainer_state is not None:
        torch.save(trainer_state, tmp / "trainer_state.pt")
    if folder.exists():
        folder.rename(old)
    tmp.rename(folder)
    shutil.rmtree(old, ignore_errors=True)


def update_checkpoint_meta(model, save_dir, name, extra):
    """Update `head.pt` and `metrics.json` in `save_dir / name` without rewriting adapter/tokenizer files."""
    folder = Path(save_dir) / name
    head = {k: v.detach().cpu() for k, v in model.net.head.state_dict().items()}
    meta = dataclasses.replace(model.meta, head=head, temperature=1.0, extra={**model.meta.extra, **extra})
    tmp_head = folder / "head.pt.tmp"
    torch.save(meta.to_dict(), tmp_head)
    tmp_head.replace(folder / "head.pt")
    tmp_metrics = folder / "metrics.json.tmp"
    tmp_metrics.write_text(json.dumps(extra, indent=2) + "\n")
    tmp_metrics.replace(folder / "metrics.json")


def load_checkpoint(
    ckpt_dir,
    device="cuda",
    *,
    trainable=False,
    lora_rank=None,
    weights_dtype="bf16",
    gradient_checkpointing=False,
):
    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / "head.pt").exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir / 'head.pt'}")
    free_gpu()
    if trainable:
        m = KevPointerModel.build(
            str(ckpt_dir), device=device, lora_rank=lora_rank, weights_dtype=weights_dtype
        )
        m.net.head.temperature = 1.0
        if gradient_checkpointing:
            m.enable_checkpointing()
        return m
    ck = Checkpoint(str(ckpt_dir))
    tok, net = ck.load(device, LoadOptions(dtype=torch.bfloat16, temperature=1.0, cuda_graphs=True, fused=True))
    m = KevPointerModel(net, tok, ck.meta, "loaded")
    m.net.head.temperature = 1.0
    return m


def normalize_eval_sets(eval_records):
    """Normalize `eval_records` into shallow-copied grouped bundle dicts and verify exactly one has `valid=True`."""
    if isinstance(eval_records, dict) and "records" in eval_records:
        copied = dict(eval_records)
        copied["valid"] = True
        eval_sets = [copied]
    elif isinstance(eval_records, (list, tuple)):
        eval_sets = [dict(s) if isinstance(s, dict) else s for s in eval_records]
    else:
        raise ValueError("eval_records must be a grouped evaluation dict or a list of grouped evaluation dicts")

    if not eval_sets:
        raise ValueError("eval_records must not be empty")
    for s in eval_sets:
        if not isinstance(s, dict) or not s["name"]:
            raise ValueError("Each evaluation set must be a grouped dict with a non-empty 'name'")
        if "valid" not in s:
            raise KeyError(f"Evaluation set {s['name']!r} is missing required 'valid' boolean flag")

    valid_sets = [s for s in eval_sets if s["valid"]]
    if len(valid_sets) != 1:
        raise ValueError(
            f"Exactly one evaluation set in eval_records must have valid=True, got {len(valid_sets)}: "
            f"{[(s['name'], s['valid']) for s in eval_sets]}"
        )
    return eval_sets, valid_sets[0]["name"]


def normalize_val_every(val_every, eval_sets):
    """Normalize `val_every` (an `int`, a `list`/`tuple` matching `eval_sets`, or a `{name: int}` dict) into `{name: int}`."""
    if isinstance(val_every, int) and not isinstance(val_every, bool):
        if val_every <= 0:
            raise ValueError(f"val_every must be a positive int, got {val_every!r}")
        return {s["name"]: val_every for s in eval_sets}
    if isinstance(val_every, (list, tuple)):
        if len(val_every) != len(eval_sets):
            raise ValueError(
                f"val_every sequence length ({len(val_every)}) must match len(eval_records) ({len(eval_sets)}): "
                f"{[s['name'] for s in eval_sets]}"
            )
        out = {}
        for s, v in zip(eval_sets, val_every, strict=True):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"val_every for {s['name']!r} must be a positive int, got {v!r}")
            out[s["name"]] = v
        return out
    if isinstance(val_every, dict):
        expected = {s["name"] for s in eval_sets}
        if set(val_every.keys()) != expected:
            raise ValueError(
                f"val_every dict keys {sorted(val_every.keys())} must match eval_records names {sorted(expected)}"
            )
        out = {}
        for s in eval_sets:
            v = val_every[s["name"]]
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"val_every for {s['name']!r} must be a positive int, got {v!r}")
            out[s["name"]] = v
        return out
    raise ValueError(
        f"val_every must be a positive int, a list/tuple of positive ints matching eval_records, "
        f"or a dict mapping eval set names to positive ints; got {val_every!r}"
    )


class GroupedTrainer:
    """Grouped shared-prefix trainer with composed `TrainingPlotter` and automatic VRAM cleanup.

    Can either wrap an existing `KevPointerModel` (`model=...`) or build/load one automatically
    from `base_model` / `init_from` / `save_dir` (`resume=True`).
    """

    def __init__(
        self,
        train_records,
        eval_records,
        save_dir,
        *,
        model=None,
        base_model=None,
        lora_rank=None,
        init_from=None,
        device="cuda",
        weights_dtype="fp32",
        use_bf16=False,
        eval_bf16=None,
        gradient_checkpointing=False,
        epochs=5,
        lr=5e-5,
        min_lr=None,
        lr_schedule="onecycle",
        decay_pct=0.20,
        weight_decay=0.01,
        warmup_steps=2000,
        pct_start=None,
        batch_size=1,
        grad_accum_steps=12,
        max_questions_per_group=64,
        max_chars_per_group=20_000,
        eval_max_questions=None,
        eval_max_chars=None,
        val_every=300,
        save_every=None,
        eval_at_start=True,
        samples_per_epoch=None,
        seed=0,
        resume=False,
        plotter=None,
        free_model_on_exit=False,
    ):
        if weights_dtype not in ("fp32", "bf16"):
            raise ValueError(f"weights_dtype must be 'fp32' or 'bf16', got {weights_dtype!r}")

        self.train_records = train_records
        self.eval_sets, self.val_name = normalize_eval_sets(eval_records)
        self.save_dir = Path(save_dir)
        self.device = device
        if str(self.device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.weights_dtype = weights_dtype
        self.use_bf16 = bool(use_bf16)
        # eval autocast: None follows use_bf16; True runs eval under bf16 autocast even when training is fp32
        self.eval_bf16 = self.use_bf16 if eval_bf16 is None else bool(eval_bf16)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.epochs = epochs
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr!r}")
        self.lr = float(lr)
        if lr_schedule not in ("onecycle", "linear", "wsd", "constant"):
            raise ValueError(
                f"lr_schedule must be one of ('onecycle', 'linear', 'wsd', 'constant'), got {lr_schedule!r}"
            )
        self.lr_schedule = lr_schedule
        if min_lr is not None:
            min_lr = float(min_lr)
            if min_lr < 0 or min_lr > self.lr:
                raise ValueError(f"min_lr must satisfy 0 <= min_lr <= lr ({self.lr}), got {min_lr!r}")
            if self.lr_schedule == "onecycle" and min_lr == 0.0:
                raise ValueError("min_lr must be > 0 when lr_schedule='onecycle' (or pass min_lr=None for default)")
        self.min_lr = min_lr
        self.decay_pct = float(decay_pct)
        if self.lr_schedule == "wsd" and not (0.0 < self.decay_pct < 1.0):
            raise ValueError(f"decay_pct must satisfy 0 < decay_pct < 1 when lr_schedule='wsd', got {decay_pct!r}")
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self._explicit_pct_start = pct_start
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_questions_per_group = max_questions_per_group
        self.max_chars_per_group = max_chars_per_group
        self._explicit_eval_max_questions = eval_max_questions
        self._explicit_eval_max_chars = eval_max_chars
        self.eval_max_questions = (
            eval_max_questions if eval_max_questions is not None else max_questions_per_group * 2
        )
        self.eval_max_chars = (
            eval_max_chars if eval_max_chars is not None else max_chars_per_group * 2
        )
        self.val_every_by_name = normalize_val_every(val_every, self.eval_sets)
        self.val_every = (
            val_every
            if isinstance(val_every, int) and not isinstance(val_every, bool)
            else dict(self.val_every_by_name)
        )
        if save_every is not None and (not isinstance(save_every, int) or isinstance(save_every, bool) or save_every <= 0):
            raise ValueError(f"save_every must be a positive int or None, got {save_every!r}")
        self.save_every = save_every
        self.eval_at_start = bool(eval_at_start)
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.resume = bool(resume)
        self.free_model_on_exit = bool(free_model_on_exit)

        self.state_path = self.save_dir / "last" / "trainer_state.pt"
        if self.resume and not self.state_path.exists():
            raise FileNotFoundError(f"Cannot resume: {self.state_path} does not exist")

        # Load or configure model (fail-fast if `init_from` checkpoint path is missing)
        if model is None:
            free_gpu()
            torch.manual_seed(self.seed)
            if self.resume:
                self.init_from = str(self.save_dir / "last")
                self.model = load_checkpoint(
                    self.save_dir / "last",
                    device=self.device,
                    trainable=True,
                    lora_rank=None,
                    weights_dtype=self.weights_dtype,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
                if lora_rank is not None and self.model.meta.lora != lora_rank:
                    raise ValueError(
                        f"Cannot resume {self.save_dir.name}: checkpoint has lora_rank={self.model.meta.lora}, "
                        f"got lora_rank={lora_rank}"
                    )
            elif init_from is not None:
                self.init_from = str(init_from)
                self.model = load_checkpoint(
                    init_from,
                    device=self.device,
                    trainable=True,
                    lora_rank=lora_rank,
                    weights_dtype=self.weights_dtype,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
            elif base_model is not None:
                self.init_from = str(base_model)
                self.model = KevPointerModel.build(
                    str(base_model), self.device, lora_rank=lora_rank, weights_dtype=self.weights_dtype
                )
                self.model.net.head.temperature = 1.0
                if self.gradient_checkpointing:
                    self.model.enable_checkpointing()
            else:
                raise ValueError("Provide either `model`, `base_model`, or `init_from`")
        else:
            if lora_rank is not None and model.meta.lora != lora_rank:
                raise ValueError(
                    f"Provided model has lora_rank={model.meta.lora}, which does not match lora_rank={lora_rank}"
                )
            self.model = model
            if init_from is not None:
                self.init_from = str(init_from)
            elif base_model is not None:
                self.init_from = str(base_model)
            else:
                self.init_from = str(self.model.meta.base)
            self.model.net.head.temperature = 1.0
            if self.gradient_checkpointing:
                self.model.enable_checkpointing()

        self.lora_rank = self.model.meta.lora
        self._model_base = self.model.meta.base
        self._model_mode = self.model.mode
        self._custom_plotter = plotter
        self.plotter = None
        self.optimizer = None
        self.scheduler = None
        self.rng = None
        self.step = 0
        self.start_epoch = 0
        self.resume_start_idx = 0
        self.best_val_kl = None

        self._recompute_schedule()

    def _epoch_groups_for(self, epoch_idx):
        epoch_rng = random.Random(f"{self.seed}:epoch:{epoch_idx}")
        sampled = (
            epoch_rng.sample(self.train_records, self.n_per_epoch)
            if self.samples_per_epoch is not None
            else self.train_records
        )
        return group_train_records(
            sampled,
            max_questions=self.max_questions_per_group,
            max_chars=self.max_chars_per_group,
            rng=epoch_rng,
        )

    def _recompute_schedule(self):
        """Compute group counts, steps_per_epoch, total_steps, and warmup pct_start."""
        self.effective_groups_bs = self.batch_size * self.grad_accum_steps
        self.n_per_epoch = (
            min(self.samples_per_epoch, len(self.train_records))
            if self.samples_per_epoch is not None
            else len(self.train_records)
        )
        if self.samples_per_epoch is None:
            probe_groups = group_train_records(
                self.train_records,
                max_questions=self.max_questions_per_group,
                max_chars=self.max_chars_per_group,
                rng=None,
            )
            self.groups_per_epoch = len(probe_groups)
            self.steps_per_epoch = math.ceil(self.groups_per_epoch / self.effective_groups_bs)
            self.total_steps = self.epochs * self.steps_per_epoch
            del probe_groups
        else:
            epoch_group_counts = [len(self._epoch_groups_for(e)) for e in range(self.epochs)]
            self.groups_per_epoch = epoch_group_counts[0]
            self.steps_per_epoch = math.ceil(self.groups_per_epoch / self.effective_groups_bs)
            self.total_steps = sum(
                math.ceil(c / self.effective_groups_bs) for c in epoch_group_counts
            )
        self.mean_questions_per_group = self.n_per_epoch / max(1, self.groups_per_epoch)

        if self._explicit_pct_start is not None:
            self.pct_start = float(self._explicit_pct_start)
        elif self.warmup_steps is not None:
            self.pct_start = min(0.1, self.warmup_steps / max(1, self.total_steps))
        else:
            self.pct_start = 0.1

        if self.lr_schedule == "wsd" and self.pct_start + self.decay_pct > 1.0:
            raise ValueError(
                f"pct_start ({self.pct_start:.4f}) + decay_pct ({self.decay_pct:.4f}) cannot exceed 1.0 for lr_schedule='wsd'"
            )

        if self.lr_schedule == "onecycle":
            if self.min_lr is None:
                self.final_div_factor = 1e4
                self.effective_min_lr = self.lr / (25.0 * self.final_div_factor)
            else:
                self.effective_min_lr = self.min_lr
                self.final_div_factor = (self.lr / 25.0) / self.effective_min_lr
        elif self.lr_schedule == "constant":
            self.final_div_factor = None
            self.effective_min_lr = self.lr if self.min_lr is None else self.min_lr
        else:
            self.final_div_factor = None
            self.effective_min_lr = 0.0 if self.min_lr is None else self.min_lr
        self.min_lr_ratio = self.effective_min_lr / self.lr

        self.plotter = (
            self._custom_plotter
            if self._custom_plotter is not None
            else TrainingPlotter(
                self.total_steps,
                self.eval_sets,
                self.val_name,
                run_name=self.save_dir.name,
            )
        )

    def set_grouping(
        self,
        max_questions_per_group,
        max_chars_per_group,
        grad_accum_steps=None,
        *,
        eval_max_questions=None,
        eval_max_chars=None,
    ):
        """Update grouping parameters (e.g., after running `probe_vram`) and recompute schedule."""
        if self.resume:
            raise ValueError("Cannot call set_grouping() when resume=True; grouping must match the saved run")
        self.max_questions_per_group = max_questions_per_group
        self.max_chars_per_group = max_chars_per_group
        if grad_accum_steps is not None:
            self.grad_accum_steps = grad_accum_steps
        if eval_max_questions is not None:
            self._explicit_eval_max_questions = eval_max_questions
        if eval_max_chars is not None:
            self._explicit_eval_max_chars = eval_max_chars
        self.eval_max_questions = (
            self._explicit_eval_max_questions
            if self._explicit_eval_max_questions is not None
            else self.max_questions_per_group * 2
        )
        self.eval_max_chars = (
            self._explicit_eval_max_chars
            if self._explicit_eval_max_chars is not None
            else self.max_chars_per_group * 2
        )
        self._recompute_schedule()
        return self.describe()

    def _run_signature(self):
        sig = {
            "base": self._model_base,
            "lora_rank": self.lora_rank,
            "weights_dtype": self.weights_dtype,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "pct_start": round(self.pct_start, 8),
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "max_questions_per_group": self.max_questions_per_group,
            "max_chars_per_group": self.max_chars_per_group,
            "samples_per_epoch": self.samples_per_epoch,
            "n_train_records": len(self.train_records),
            "seed": self.seed,
            "total_steps": self.total_steps,
        }
        if self.lr_schedule != "onecycle":
            sig["lr_schedule"] = self.lr_schedule
        if self.min_lr is not None:
            sig["min_lr"] = self.min_lr
        if self.lr_schedule == "wsd":
            sig["decay_pct"] = round(self.decay_pct, 8)
        return sig

    def _verify_resume_signature(self, st):
        saved = st["signature"]
        current = self._run_signature()
        mismatches = [
            f"{k}: saved={saved[k]!r} != current={current[k]!r}"
            for k in current
            if saved[k] != current[k]
        ]
        if mismatches:
            raise ValueError(
                f"Cannot resume {self.save_dir.name}: configuration does not match saved checkpoint: "
                + "; ".join(mismatches)
            )

    def describe(self):
        return {
            "run_name": self.save_dir.name,
            "save_dir": str(self.save_dir),
            "init_from": self.init_from,
            "base": self._model_base,
            "lora_rank": self.lora_rank,
            "mode": self.model.mode if self.model is not None else f"{self._model_mode} (freed)",
            "resume": self.resume,
            "seed": self.seed,
            "epochs": self.epochs,
            "lr_schedule": self.lr_schedule,
            "lr": self.lr,
            "min_lr": self.effective_min_lr,
            "min_lr_ratio": round(self.min_lr_ratio, 6),
            "decay_pct": round(self.decay_pct, 5) if self.lr_schedule == "wsd" else None,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "effective_groups_bs": self.effective_groups_bs,
            "val_every": self.val_every,
            "save_every": self.save_every,
            "samples_per_epoch": self.samples_per_epoch,
            "weights_dtype": self.weights_dtype,
            "use_bf16": self.use_bf16,
            "eval_bf16": self.eval_bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_questions_per_group": self.max_questions_per_group,
            "max_chars_per_group": self.max_chars_per_group,
            "eval_max_questions": self.eval_max_questions,
            "eval_max_chars": self.eval_max_chars,
            "groups_per_epoch": self.groups_per_epoch,
            "mean_questions_per_group": round(self.mean_questions_per_group, 2),
            "mean_questions_per_step": round(self.effective_groups_bs * self.mean_questions_per_group, 1),
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "warmup_steps": int(round(self.pct_start * self.total_steps)),
            "pct_start": round(self.pct_start, 5),
        }

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _eval_autocast(self):
        if self.eval_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @property
    def current_lr(self):
        return self.scheduler.get_last_lr()[0]

    def _build_scheduler(self, optimizer):
        if self.lr_schedule == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.total_steps,
                pct_start=self.pct_start,
                div_factor=25.0,
                final_div_factor=self.final_div_factor,
            )

        warmup_steps = int(round(self.pct_start * self.total_steps))
        last_step = max(1, self.total_steps - 1)
        init_ratio = 1.0 / 25.0
        min_ratio = self.min_lr_ratio

        if self.lr_schedule == "constant":
            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return init_ratio + (1.0 - init_ratio) * (step / warmup_steps)
                return 1.0
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if self.lr_schedule == "linear":
            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return init_ratio + (1.0 - init_ratio) * (step / warmup_steps)
                pct = (step - warmup_steps) / max(1, last_step - warmup_steps)
                return max(min_ratio, 1.0 - (1.0 - min_ratio) * pct)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if self.lr_schedule == "wsd":
            stable_end = int(round((1.0 - self.decay_pct) * self.total_steps))
            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    pct = step / warmup_steps
                    return init_ratio + (1.0 - init_ratio) * (1.0 - math.cos(math.pi * pct)) / 2.0
                if step < stable_end:
                    return 1.0
                pct = min(1.0, (step - stable_end) / max(1, last_step - stable_end))
                return min_ratio + (1.0 - min_ratio) * (1.0 + math.cos(math.pi * pct)) / 2.0
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        raise ValueError(f"Unsupported lr_schedule: {self.lr_schedule!r}")

    def train_one_step(
        self,
        macro_batch,
        *,
        optimizer=None,
        scheduler=None,
        step=None,
        total_steps=None,
        rng=None,
    ):
        """Run one optimizer step over `macro_batch` of persona groups using shared-prefix forward/backward."""
        opt = optimizer if optimizer is not None else self.optimizer
        sched = scheduler if scheduler is not None else self.scheduler
        cur_step = step if step is not None else self.step
        max_steps = total_steps if total_steps is not None else self.total_steps
        step_rng = rng if rng is not None else self.rng

        total_questions = sum(len(g) for g in macro_batch)
        step_kls = []
        for sub_start in range(0, len(macro_batch), self.batch_size):
            batch_groups = macro_batch[sub_start : sub_start + self.batch_size]
            prepared = [self.model.prepare(make_grouped_view(g, rng=step_rng)) for g in batch_groups]
            with self._autocast():
                logits_batch = self.model.forward([enc for enc, _ in prepared], shared_prefix=True)

            losses = []
            for (_, targets), record_logits in zip(prepared, logits_batch, strict=True):
                for z, q in zip(record_logits, targets, strict=True):
                    t = q["target"].to(self.device)
                    logp = torch.log_softmax(z.float(), dim=-1)
                    losses.append(-(t * logp).sum())
                    nz = t > 0
                    step_kls.append(((t[nz] * (t[nz].log() - logp[nz])).sum() / math.log(2)).item())

            loss = torch.stack(losses).sum() / total_questions
            loss.backward()
            del prepared, logits_batch, losses, loss

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), 1.0)
        if torch.isfinite(grad_norm):
            opt.step()
        else:
            print(f"Skipping optimizer.step() at step {cur_step + 1}: non-finite grad_norm={grad_norm.item()}")
        if cur_step + 1 < max_steps:
            sched.step()
        opt.zero_grad(set_to_none=True)
        return step_kls

    def _model_eval0_digest(self):
        meta = self.model.meta
        h = hashlib.sha256()
        key = (
            f"v={EVAL0_CACHE_VERSION}|init={self.init_from}|base={meta.base}|rev={meta.base_revision}|"
            f"lora={meta.lora}|hdim={meta.head_dim}|temp={self.model.net.head.temperature}|"
            f"iso={meta.option_isolation}|wdtype={self.weights_dtype}|bf16={self.use_bf16}"
        )
        if self.eval_bf16 != self.use_bf16:   # only then, so existing step-0 caches stay valid
            key += f"|evalbf16={self.eval_bf16}"
        h.update(key.encode())
        for p in self.model.trainable_parameters():
            h.update(p.detach().cpu().float().numpy().tobytes())
        return h.hexdigest()[:12]

    @staticmethod
    def _eval_set_digest(s):
        h = hashlib.sha256()
        eval_fn_name = s["eval_fn"].__name__ if s["eval_fn"] is not None else ""
        h.update(
            f"name={s['name']}|fn={eval_fn_name}|views={len(s['packed_views'])}|recs={len(s['records'])}".encode()
        )
        for v in s["packed_views"]:
            h.update(json.dumps(v, sort_keys=True).encode())
        for r in s["records"]:
            arm = r["arm"] if "arm" in r else ""
            caseid = r["CASEID"] if "CASEID" in r else ""
            h.update(f"{r['qid']}|{r['type']}|{arm}|{caseid}|".encode())
            h.update(r["human_dist"].tobytes())
        return h.hexdigest()[:12]

    def evaluate_step0(self):
        """Evaluate step-0 baseline using `.cache/` per `(model, dtype, eval_set)` so repeated runs start immediately."""
        model_digest = self._model_eval0_digest()
        summaries = {}
        extra_dfs = {}
        uncached = []
        for s in self.eval_sets:
            eval_digest = self._eval_set_digest(s)
            cache_path = DEFAULT_CACHE_DIR / f"eval0_v{EVAL0_CACHE_VERSION}_{s['name']}_{model_digest}_{eval_digest}.pkl"
            if cache_path.exists():
                cached = pickle.loads(cache_path.read_bytes())
                summaries[s["name"]] = cached["summary"]
                extra_dfs[s["name"]] = cached["extra_df"]
                s["last_extra_df"] = cached["extra_df"]
            else:
                uncached.append((s, cache_path))

        if uncached:
            was_training = self.model.training
            self.model.eval()
            eval_dtype = torch.bfloat16 if self.eval_bf16 else None
            free_gpu()   # see evaluate(): graph capture cannot reuse or release the allocator's cached blocks
            with self._eval_autocast(), fused_eval_context(self.model, eval_dtype=eval_dtype):
                try:
                    for idx, (s, cache_path) in enumerate(uncached, start=1):
                        name = s["name"]
                        self.plotter.update_status(
                            0, epoch=0, status=f"eval step 0 ({idx}/{len(uncached)}): {name}..."
                        )
                        summary, _ = evaluate_model(
                            self.model,
                            s,
                            max_questions=self.eval_max_questions,
                            max_chars=self.eval_max_chars,
                            show_progress=False,
                        )
                        extra_df = s["last_extra_df"]
                        summaries[name] = summary
                        extra_dfs[name] = extra_df
                        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        tmp_path = cache_path.with_suffix(".tmp")
                        tmp_path.write_bytes(
                            pickle.dumps({"summary": summary, "extra_df": extra_df}, protocol=pickle.HIGHEST_PROTOCOL)
                        )
                        tmp_path.replace(cache_path)
                finally:
                    if was_training:
                        self.model.train()
                    free_gpu()

        return summaries, extra_dfs

    def _due_eval_sets(self, step, *, force_all=False):
        if force_all:
            return list(self.eval_sets)
        return [s for s in self.eval_sets if step % self.val_every_by_name[s["name"]] == 0]

    def evaluate(self, cur_epoch=0, *, show_progress=False, eval_sets=None):
        """Run evaluation sets sharing fused kernels + `CudaGraphs` and free temporary GPU buffers on exit."""
        target_sets = self.eval_sets if eval_sets is None else eval_sets
        summaries = {}
        extra_dfs = {}
        was_training = self.model.training
        self.model.eval()
        eval_dtype = torch.bfloat16 if self.eval_bf16 else None
        # Return training's cached-but-free blocks to the driver first: CudaGraphs capture into private pools that cannot
        # reuse them, and memory cannot be freed mid-capture, so a full cache makes captures OOM (worse after long runs).
        free_gpu()
        with self._eval_autocast(), fused_eval_context(self.model, eval_dtype=eval_dtype):
            try:
                for idx, s in enumerate(target_sets, start=1):
                    name = s["name"]
                    if not show_progress:
                        self.plotter.update_status(
                            self.step, epoch=cur_epoch, status=f"eval ({idx}/{len(target_sets)}): {name}..."
                        )
                    summary, _ = evaluate_model(
                        self.model,
                        s,
                        max_questions=self.eval_max_questions,
                        max_chars=self.eval_max_chars,
                        show_progress=show_progress,
                    )
                    summaries[name] = summary
                    extra_dfs[name] = s["last_extra_df"]
            finally:
                if was_training:
                    self.model.train()
                free_gpu()
        return summaries, extra_dfs

    def _build_extra(self, epoch_idx, train_kl, val_ce=None, val_kl=None, summaries=None):
        return {
            "init_from": self.init_from,
            "step": self.step,
            "epoch": epoch_idx + 1,
            "val_set": self.val_name,
            "train_kl": train_kl,
            "val_ce": val_ce,
            "val_dist_kl": val_kl,
            "eval_metrics": (
                {name: {k: float(v) for k, v in s.items()} for name, s in summaries.items()}
                if summaries is not None
                else None
            ),
            "settings": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "pct_start": self.pct_start,
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "effective_groups_per_step": self.effective_groups_bs,
                "max_questions_per_group": self.max_questions_per_group,
                "max_chars_per_group": self.max_chars_per_group,
                "weights_dtype": self.weights_dtype,
                "use_bf16": self.use_bf16,
                "eval_bf16": self.eval_bf16,
                "gradient_checkpointing": self.gradient_checkpointing,
                "epochs": self.epochs,
                "samples_per_epoch": self.samples_per_epoch,
                "seed": self.seed,
            },
        }

    def _build_trainer_state(
        self,
        epoch_idx,
        next_start,
        n_epoch_groups,
        *,
        eval_pending=False,
        pending_train_kl=None,
    ):
        at_epoch_end = next_start >= n_epoch_groups
        return {
            "init_from": self.init_from,
            "step": self.step,
            "epoch": epoch_idx + 1 if at_epoch_end else epoch_idx,
            "next_start": 0 if at_epoch_end else next_start,
            "best_val_kl": self.best_val_kl,
            "eval_pending": bool(eval_pending),
            "pending_epoch_idx": epoch_idx,
            "pending_next_start": next_start,
            "pending_n_epoch_groups": n_epoch_groups,
            "pending_train_kl": pending_train_kl,
            "rng_state": self.rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "signature": self._run_signature(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            **self.plotter.state_dict(),
        }

    def _save_trainer_state(
        self,
        epoch_idx,
        next_start,
        n_epoch_groups,
        *,
        eval_pending=False,
        pending_train_kl=None,
    ):
        state = self._build_trainer_state(
            epoch_idx,
            next_start,
            n_epoch_groups,
            eval_pending=eval_pending,
            pending_train_kl=pending_train_kl,
        )
        tmp_state = self.state_path.with_suffix(".tmp")
        torch.save(state, tmp_state)
        tmp_state.replace(self.state_path)

    def _complete_eval_and_save(self, epoch_idx, next_start, n_epoch_groups, train_kl, *, force_all=False):
        is_final_step = force_all or ((epoch_idx == self.epochs - 1) and (next_start >= n_epoch_groups))
        due_sets = self._due_eval_sets(self.step, force_all=is_final_step)
        summaries, extra_dfs = self.evaluate(cur_epoch=epoch_idx + 1, eval_sets=due_sets)
        self.plotter.log_eval(self.step, summaries, extra_dfs, epoch=epoch_idx + 1)

        if self.val_name in summaries:
            val_summary = summaries[self.val_name]
            val_ce = float(val_summary["ce"])
            val_kl = float(val_summary["dist_kl"])
            extra = self._build_extra(epoch_idx, train_kl, val_ce=val_ce, val_kl=val_kl, summaries=summaries)
            if val_kl < self.best_val_kl:
                self.best_val_kl = val_kl
                save_checkpoint(self.model, self.save_dir, "best", extra=extra)
        else:
            extra = self._build_extra(epoch_idx, train_kl, summaries=summaries)
        update_checkpoint_meta(self.model, self.save_dir, "last", extra=extra)
        self._save_trainer_state(epoch_idx, next_start, n_epoch_groups, eval_pending=False)
        self.plotter.update_status(self.step, epoch=epoch_idx + 1, status="training", lr=self.current_lr)

    def save_last(self, epoch_idx, next_start, n_epoch_groups):
        train_kl = float(self.plotter.history["step_kl"][-1])
        self.plotter.update_status(self.step, epoch=epoch_idx + 1, status="saving last...", lr=self.current_lr)
        extra = self._build_extra(epoch_idx, train_kl)
        state = self._build_trainer_state(
            epoch_idx,
            next_start,
            n_epoch_groups,
            eval_pending=False,
        )
        save_checkpoint(self.model, self.save_dir, "last", extra=extra, trainer_state=state)
        self.plotter.update_status(self.step, epoch=epoch_idx + 1, status="training", lr=self.current_lr)

    def validate_and_save(self, epoch_idx, next_start, n_epoch_groups, *, force_all=False):
        train_kl = float(self.plotter.history["step_kl"][-1])
        # Save `last/` + `trainer_state.pt` atomically with `eval_pending=True` before evaluation so progress is never lost
        pre_extra = self._build_extra(epoch_idx, train_kl)
        pre_state = self._build_trainer_state(
            epoch_idx,
            next_start,
            n_epoch_groups,
            eval_pending=True,
            pending_train_kl=train_kl,
        )
        save_checkpoint(self.model, self.save_dir, "last", extra=pre_extra, trainer_state=pre_state)
        self._complete_eval_and_save(epoch_idx, next_start, n_epoch_groups, train_kl, force_all=force_all)

    def _prepare_epoch(self, epoch_idx):
        epoch_groups = self._epoch_groups_for(epoch_idx)
        first_start = self.resume_start_idx if epoch_idx == self.start_epoch else 0
        return epoch_groups, first_start

    def cleanup(self, free_model=False):
        """Free optimizer, scheduler, gradients, CudaGraphs, and cached GPU blocks."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer = None
        self.scheduler = None
        if self.model is not None:
            for p in self.model.trainable_parameters():
                p.grad = None
            release_cuda_graphs(self.model)
            if free_model:
                self.model = None
        _, _, tb = sys.exc_info()
        if tb is not None:
            traceback.clear_frames(tb)
        del tb
        free_gpu()

    def train(self):
        """Execute the full training loop and guarantee VRAM cleanup on exit or interrupt."""
        if self.model is None:
            raise ValueError("self.model is None; re-instantiate GroupedTrainer before calling train()")

        release_cuda_graphs(self.model)
        free_gpu()

        self.plotter = (
            self._custom_plotter
            if self._custom_plotter is not None
            else TrainingPlotter(
                self.total_steps,
                self.eval_sets,
                self.val_name,
                run_name=self.save_dir.name,
            )
        )
        self.plotter.init_display(force=True)

        try:
            if str(self.device).startswith("cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self.rng = random.Random(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            self.optimizer = torch.optim.AdamW(
                self.model.trainable_parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            self.scheduler = self._build_scheduler(self.optimizer)

            if self.resume:
                st = torch.load(self.state_path, map_location="cpu", weights_only=False)
                self._verify_resume_signature(st)
                self.init_from = st["init_from"]
                self.step = st["step"]
                self.start_epoch = st["epoch"]
                self.resume_start_idx = st["next_start"]
                self.best_val_kl = st["best_val_kl"]
                self.rng.setstate(st["rng_state"])
                torch.set_rng_state(st["torch_rng_state"])
                torch.cuda.set_rng_state(st["cuda_rng_state"])
                self.optimizer.load_state_dict(st["optimizer"])
                self.scheduler.load_state_dict(st["scheduler"])
                self.plotter.load_state_dict(st, step=self.step)
                if st["eval_pending"]:
                    self.plotter.update_status(
                        self.step,
                        epoch=st["pending_epoch_idx"] + 1,
                        status="completing interrupted eval...",
                        lr=self.current_lr,
                    )
                    self._complete_eval_and_save(
                        st["pending_epoch_idx"],
                        st["pending_next_start"],
                        st["pending_n_epoch_groups"],
                        st["pending_train_kl"],
                    )
                self.plotter.update_status(
                    self.step, epoch=self.start_epoch + 1, status="resumed", lr=self.current_lr
                )
                print(
                    f"Resumed {self.save_dir.name} at epoch {self.start_epoch + 1}, "
                    f"step {self.step}/{self.total_steps} (best_{self.val_name}_kl={self.best_val_kl:.4f})"
                )
            else:
                self.step = 0
                self.start_epoch = 0
                self.resume_start_idx = 0
                if self.eval_at_start:
                    summaries_0, extra_dfs_0 = self.evaluate_step0()
                    self.best_val_kl = float(summaries_0[self.val_name]["dist_kl"])
                    self.plotter.log_eval(0, summaries_0, extra_dfs_0, epoch=0)
                else:
                    self.best_val_kl = float("inf")
                self.plotter.update_status(0, epoch=1, status="starting epoch 1...")

            if self.gradient_checkpointing:
                self.model.enable_checkpointing()
            self.model.train()

            ran_steps = False
            n_epoch_groups = 0
            for e in range(self.start_epoch, self.epochs):
                epoch_groups, first_start = self._prepare_epoch(e)
                n_epoch_groups = len(epoch_groups)

                for start in range(first_start, n_epoch_groups, self.effective_groups_bs):
                    macro_batch = epoch_groups[start : start + self.effective_groups_bs]
                    step_kls = self.train_one_step(macro_batch)
                    self.step += 1
                    ran_steps = True
                    self.plotter.log_step(self.step, step_kls, epoch=e + 1, lr=self.current_lr)

                    next_start = start + self.effective_groups_bs
                    is_final_step = (e == self.epochs - 1) and (next_start >= n_epoch_groups)
                    if is_final_step or self._due_eval_sets(self.step):
                        self.validate_and_save(e, next_start, n_epoch_groups, force_all=is_final_step)
                    elif self.save_every is not None and self.step % self.save_every == 0:
                        self.save_last(e, next_start, n_epoch_groups)

            self.plotter.update_status(self.step, epoch=self.epochs, status="completed")

        finally:
            self.cleanup(free_model=self.free_model_on_exit)

        return self.plotter

    def probe_vram(self, grid, *, stop_on_oom=True):
        """Sweep `(max_questions, max_chars)` pairs to measure steady-state peak VRAM without mutating weights."""
        if self.model is None:
            raise ValueError("self.model is None")

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_vram_gb:.1f} GiB total)")

        probe_records = (
            random.Random(f"{self.seed}:epoch:0").sample(self.train_records, self.n_per_epoch)
            if self.samples_per_epoch is not None
            else self.train_records
        )

        rows = []
        for mq, mc in grid:
            groups = group_train_records(
                probe_records, max_questions=mq, max_chars=mc, rng=None
            )
            by_padded_chars = sorted(
                groups,
                key=lambda g: len(g) * max(_record_chars(r) for r in g),
                reverse=True,
            )[:15]
            by_len = sorted(
                groups,
                key=lambda g: (len(g), max(_record_chars(r) for r in g)),
                reverse=True,
            )[:15]
            candidates = by_padded_chars + by_len

            worst_g, worst_tokens, worst_shape = None, -1, None
            for g in candidates:
                enc, _ = self.model.prepare(make_grouped_view(g))
                Ls = enc["seg"].count(0)
                q = max(enc["seg"])
                Lb = max(enc["seg"].count(s) for s in range(1, q + 1))
                padded_tokens = Ls + q * Lb
                if padded_tokens > worst_tokens:
                    worst_tokens = padded_tokens
                    worst_g = g
                    worst_shape = (q, Ls, Lb)

            # Use 2 micro-batches when grad_accum_steps > 1 so micro-batch 2 runs while p.grad from micro-batch 1 is allocated
            worst_macro = [worst_g] * (self.batch_size * min(2, self.grad_accum_steps))
            opt = torch.optim.AdamW(self.model.trainable_parameters(), lr=0.0, weight_decay=0.0)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
            if self.gradient_checkpointing:
                self.model.enable_checkpointing()
            self.model.train()
            release_cuda_graphs(self.model)
            free_gpu()
            torch.cuda.reset_peak_memory_stats()

            try:
                with torch.random.fork_rng(devices=[self.device]):
                    for s in range(2):
                        self.train_one_step(
                            worst_macro,
                            optimizer=opt,
                            scheduler=sched,
                            step=s,
                            total_steps=10,
                            rng=random.Random(0),
                        )
                peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                peak_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                status = "OK"
            except torch.OutOfMemoryError as exc:
                if exc.__traceback__ is not None:
                    traceback.clear_frames(exc.__traceback__)
                peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                peak_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                status = "OOM"
            finally:
                opt.zero_grad(set_to_none=True)
                del opt, sched
                for p in self.model.trainable_parameters():
                    p.grad = None
                free_gpu()

            q, Ls, Lb = worst_shape
            steps_per_epoch = math.ceil(len(groups) / (self.batch_size * self.grad_accum_steps))
            row = {
                "max_questions": mq,
                "max_chars": mc,
                "groups_per_epoch": len(groups),
                "mean_q_per_group": round(len(probe_records) / len(groups), 2),
                "steps_per_epoch": steps_per_epoch,
                f"total_steps_{self.epochs}ep": self.epochs * steps_per_epoch,
                "worst_(Q, Ls, Lb)": f"({q}, {Ls}, {Lb})",
                "worst_padded_toks": worst_tokens,
                "peak_alloc_GiB": round(peak_alloc_gb, 2),
                "peak_reserved_GiB": round(peak_res_gb, 2),
                "status": status,
            }
            rows.append(row)
            print(
                f"mq={mq:3d}, mc={mc:6d} -> {row['status']:3s} | "
                f"peak_alloc={row['peak_alloc_GiB']:5.2f} GiB (reserved={row['peak_reserved_GiB']:5.2f} GiB) | "
                f"worst={row['worst_(Q, Ls, Lb)']} ({row['worst_padded_toks']:,} toks) | "
                f"{self.epochs}ep_steps={row[f'total_steps_{self.epochs}ep']:,}"
            )
            if status == "OOM" and stop_on_oom:
                break

        return pd.DataFrame(rows)
