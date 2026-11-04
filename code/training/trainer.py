import contextlib
import gc
import json
import math
import random
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch

from data.base import (
    CudaGraphs,
    _record_chars,
    evaluate_model,
    group_train_records,
    make_grouped_view,
)
from models.kev_pointer import Checkpoint, KevPointerModel, LoadOptions
from training.plotter import TrainingPlotter


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


def save_checkpoint(model, save_dir, name, extra):
    """Atomically write checkpoint to `save_dir / name` via a temporary directory."""
    save_dir = Path(save_dir)
    folder, tmp = save_dir / name, save_dir / f"{name}-tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save(tmp, extra=extra)
    (tmp / "metrics.json").write_text(json.dumps(extra, indent=2) + "\n")
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)


def load_checkpoint(
    ckpt_dir,
    device="cuda",
    *,
    trainable=False,
    weights_dtype="bf16",
    gradient_checkpointing=False,
):
    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / "head.pt").exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir / 'head.pt'}")
    free_gpu()
    if trainable:
        m = KevPointerModel.build(str(ckpt_dir), device=device, weights_dtype=weights_dtype)
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
    """Normalize `eval_records` into a list of grouped bundle dicts and verify exactly one has `valid=True`."""
    if isinstance(eval_records, dict) and "records" in eval_records:
        eval_records["valid"] = True
        eval_sets = [eval_records]
    elif isinstance(eval_records, (list, tuple)):
        eval_sets = list(eval_records)
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
        lora_rank=64,
        init_from=None,
        device="cuda",
        weights_dtype="fp32",
        use_bf16=False,
        gradient_checkpointing=False,
        epochs=5,
        lr=5e-5,
        weight_decay=0.01,
        warmup_steps=2000,
        pct_start=None,
        batch_size=1,
        grad_accum_steps=12,
        max_questions_per_group=64,
        max_chars_per_group=20_000,
        eval_batch_size=128,
        log_every=20,
        val_every=300,
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
        self.weights_dtype = weights_dtype
        self.use_bf16 = bool(use_bf16)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self._explicit_pct_start = pct_start
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_questions_per_group = max_questions_per_group
        self.max_chars_per_group = max_chars_per_group
        self.eval_batch_size = eval_batch_size
        self.log_every = log_every
        self.val_every = val_every
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
            if self.resume:
                self.init_from = str(self.save_dir / "last")
                self.model = load_checkpoint(
                    self.save_dir / "last",
                    device=self.device,
                    trainable=True,
                    weights_dtype=self.weights_dtype,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
            elif init_from is not None:
                self.init_from = str(init_from)
                self.model = load_checkpoint(
                    init_from,
                    device=self.device,
                    trainable=True,
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
            self.model = model
            self.init_from = str(init_from or base_model or self.model.meta.base)
            self.model.net.head.temperature = 1.0
            if self.gradient_checkpointing:
                self.model.enable_checkpointing()

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

    def _recompute_schedule(self):
        """Compute group counts, steps_per_epoch, total_steps, and warmup pct_start."""
        self.effective_groups_bs = self.batch_size * self.grad_accum_steps
        self.n_per_epoch = (
            min(self.samples_per_epoch, len(self.train_records))
            if self.samples_per_epoch
            else len(self.train_records)
        )
        probe_records = (
            random.Random(self.seed).sample(self.train_records, self.n_per_epoch)
            if self.samples_per_epoch
            else self.train_records
        )
        probe_groups = group_train_records(
            probe_records,
            max_questions=self.max_questions_per_group,
            max_chars=self.max_chars_per_group,
            rng=None,
        )
        self.groups_per_epoch = len(probe_groups)
        self.mean_questions_per_group = self.n_per_epoch / max(1, self.groups_per_epoch)
        self.steps_per_epoch = math.ceil(self.groups_per_epoch / self.effective_groups_bs)
        self.total_steps = self.epochs * self.steps_per_epoch
        del probe_records, probe_groups

        if self._explicit_pct_start is not None:
            self.pct_start = float(self._explicit_pct_start)
        elif self.warmup_steps is not None:
            self.pct_start = min(0.1, self.warmup_steps / max(1, self.total_steps))
        else:
            self.pct_start = 0.1

        self.plotter = (
            self._custom_plotter
            if self._custom_plotter is not None
            else TrainingPlotter(
                self.total_steps,
                self.log_every,
                self.eval_sets,
                self.val_name,
                run_name=self.save_dir.name,
            )
        )

    def set_grouping(self, max_questions_per_group, max_chars_per_group, grad_accum_steps=None):
        """Update grouping parameters (e.g., after running `probe_vram`) and recompute schedule."""
        self.max_questions_per_group = max_questions_per_group
        self.max_chars_per_group = max_chars_per_group
        if grad_accum_steps is not None:
            self.grad_accum_steps = grad_accum_steps
        self._recompute_schedule()
        return self.describe()

    def describe(self):
        return {
            "run_name": self.save_dir.name,
            "base": self._model_base,
            "mode": self.model.mode if self.model is not None else f"{self._model_mode} (freed)",
            "weights_dtype": self.weights_dtype,
            "use_bf16": self.use_bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_questions_per_group": self.max_questions_per_group,
            "max_chars_per_group": self.max_chars_per_group,
            "groups_per_epoch": self.groups_per_epoch,
            "mean_questions_per_group": round(self.mean_questions_per_group, 2),
            "effective_groups_bs": self.effective_groups_bs,
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

    @property
    def current_lr(self):
        return self.scheduler.get_last_lr()[0]

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
            for (_, targets), record_logits in zip(prepared, logits_batch):
                for z, q in zip(record_logits, targets):
                    t = q["target"].to(self.device)
                    logp = torch.log_softmax(z.float(), dim=-1)
                    losses.append(-(t * logp).sum())
                    nz = t > 0
                    step_kls.append(((t[nz] * (t[nz].log() - logp[nz])).sum() / math.log(2)).item())

            loss = torch.stack(losses).sum() / total_questions
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), 1.0)
        if torch.isfinite(grad_norm):
            opt.step()
        else:
            print(f"Skipping optimizer.step() at step {cur_step + 1}: non-finite grad_norm={grad_norm.item()}")
        if cur_step + 1 < max_steps:
            sched.step()
        opt.zero_grad(set_to_none=True)
        return step_kls

    def evaluate(self, cur_epoch=0):
        """Run all evaluation sets sharing a single `CudaGraphs` instance and free its buffers on exit."""
        summaries = {}
        extra_dfs = {}
        was_training = self.model.training
        with self._autocast():
            self.model.net.graphs = CudaGraphs(self.model.net.lm, self.model.net.pad_id)
            try:
                for idx, s in enumerate(self.eval_sets, start=1):
                    name = s["name"]
                    self.plotter.update_status(
                        self.step, epoch=cur_epoch, status=f"eval ({idx}/{len(self.eval_sets)}): {name}..."
                    )
                    summary, _ = evaluate_model(self.model, s, batch_size=self.eval_batch_size, show_progress=False)
                    summaries[name] = summary
                    extra_dfs[name] = s["last_extra_df"]
            finally:
                self.model.net.graphs = None
                if was_training:
                    self.model.train()
                free_gpu()
        return summaries, extra_dfs

    def validate_and_save(self, epoch_idx, next_start, n_epoch_groups, epoch_rng_state):
        summaries, extra_dfs = self.evaluate(cur_epoch=epoch_idx + 1)
        val_summary = summaries[self.val_name]
        val_ce = float(val_summary["ce"])
        val_kl = float(val_summary["dist_kl"])
        self.plotter.log_eval(self.step, summaries, extra_dfs, epoch=epoch_idx + 1)

        train_kl = float(self.plotter.history["window_kl"][-1])
        extra = {
            "init_from": self.init_from,
            "step": self.step,
            "epoch": epoch_idx + 1,
            "val_set": self.val_name,
            "train_kl": train_kl,
            "val_ce": val_ce,
            "val_dist_kl": val_kl,
            "eval_metrics": {name: {k: float(v) for k, v in s.items()} for name, s in summaries.items()},
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
                "gradient_checkpointing": self.gradient_checkpointing,
                "epochs": self.epochs,
                "samples_per_epoch": self.samples_per_epoch,
                "seed": self.seed,
            },
        }
        if val_kl < self.best_val_kl:
            self.best_val_kl = val_kl
            save_checkpoint(self.model, self.save_dir, "best", extra=extra)
        save_checkpoint(self.model, self.save_dir, "last", extra=extra)

        at_epoch_end = next_start >= n_epoch_groups
        torch.save(
            {
                "step": self.step,
                "epoch": epoch_idx + 1 if at_epoch_end else epoch_idx,
                "next_start": 0 if at_epoch_end else next_start,
                "best_val_kl": self.best_val_kl,
                "rng_state": self.rng.getstate(),
                "epoch_rng_state": None if at_epoch_end else epoch_rng_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                **self.plotter.state_dict(),
            },
            self.state_path,
        )
        self.plotter.update_status(self.step, epoch=epoch_idx + 1, status="training", lr=self.current_lr)

    def _prepare_epoch(
        self,
        epoch_idx,
        *,
        resume_rng_state=None,
        epoch_rng_state_saved=None,
        epoch_groups_saved=None,
    ):
        if epoch_idx == self.start_epoch and (epoch_rng_state_saved is not None or epoch_groups_saved is not None):
            if epoch_rng_state_saved is not None:
                epoch_rng_state = epoch_rng_state_saved
                self.rng.setstate(epoch_rng_state)
                sampled = (
                    self.rng.sample(self.train_records, self.n_per_epoch)
                    if self.samples_per_epoch
                    else self.train_records
                )
                epoch_groups = group_train_records(
                    sampled,
                    max_questions=self.max_questions_per_group,
                    max_chars=self.max_chars_per_group,
                    rng=self.rng,
                )
            else:
                epoch_rng_state = None
                epoch_groups = epoch_groups_saved
            self.rng.setstate(resume_rng_state)
            first_start = self.resume_start_idx
        else:
            epoch_rng_state = self.rng.getstate()
            sampled = (
                self.rng.sample(self.train_records, self.n_per_epoch)
                if self.samples_per_epoch
                else self.train_records
            )
            epoch_groups = group_train_records(
                sampled,
                max_questions=self.max_questions_per_group,
                max_chars=self.max_chars_per_group,
                rng=self.rng,
            )
            first_start = 0
        return epoch_groups, epoch_rng_state, first_start

    def cleanup(self, free_model=False):
        """Free optimizer, scheduler, gradients, CudaGraphs, and cached GPU blocks."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer = None
        self.scheduler = None
        if self.model is not None:
            for p in self.model.trainable_parameters():
                p.grad = None
            if self.model.net.graphs is not None:
                self.model.net.graphs = None
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

        if self.model.net.graphs is not None:
            self.model.net.graphs = None
        free_gpu()

        self.plotter = (
            self._custom_plotter
            if self._custom_plotter is not None
            else TrainingPlotter(
                self.total_steps,
                self.log_every,
                self.eval_sets,
                self.val_name,
                run_name=self.save_dir.name,
            )
        )
        self.plotter.init_display(force=True)

        try:
            self.rng = random.Random(self.seed)
            self.optimizer = torch.optim.AdamW(
                self.model.trainable_parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.lr, total_steps=self.total_steps, pct_start=self.pct_start
            )

            if self.resume:
                st = torch.load(self.state_path, map_location="cpu", weights_only=False)
                self.step = st["step"]
                self.start_epoch = st["epoch"]
                self.resume_start_idx = st["next_start"]
                self.best_val_kl = st["best_val_kl"]
                resume_rng_state = st["rng_state"]
                epoch_rng_state_saved = st["epoch_rng_state"] if "epoch_rng_state" in st else None
                epoch_groups_saved = st["epoch_records"] if "epoch_records" in st else None
                self.optimizer.load_state_dict(st["optimizer"])
                self.scheduler.load_state_dict(st["scheduler"])
                self.plotter.load_state_dict(st, step=self.step)
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
                resume_rng_state = None
                epoch_rng_state_saved = None
                epoch_groups_saved = None
                summaries_0, extra_dfs_0 = self.evaluate(cur_epoch=0)
                self.best_val_kl = float(summaries_0[self.val_name]["dist_kl"])
                self.plotter.log_eval(0, summaries_0, extra_dfs_0, epoch=0)
                self.plotter.update_status(0, epoch=1, status="starting epoch 1...")

            if self.gradient_checkpointing:
                self.model.enable_checkpointing()
            self.model.train()

            ran_steps = False
            n_epoch_groups = 0
            epoch_rng_state = None
            for e in range(self.start_epoch, self.epochs):
                epoch_groups, epoch_rng_state, first_start = self._prepare_epoch(
                    e,
                    resume_rng_state=resume_rng_state,
                    epoch_rng_state_saved=epoch_rng_state_saved,
                    epoch_groups_saved=epoch_groups_saved,
                )
                n_epoch_groups = len(epoch_groups)

                for start in range(first_start, n_epoch_groups, self.effective_groups_bs):
                    macro_batch = epoch_groups[start : start + self.effective_groups_bs]
                    step_kls = self.train_one_step(macro_batch)
                    self.step += 1
                    ran_steps = True
                    self.plotter.log_step(self.step, step_kls, epoch=e + 1, lr=self.current_lr)

                    if self.step % self.val_every == 0:
                        self.validate_and_save(
                            e, start + self.effective_groups_bs, n_epoch_groups, epoch_rng_state
                        )

            if ran_steps and self.step % self.val_every != 0:
                self.validate_and_save(
                    self.epochs - 1, n_epoch_groups, n_epoch_groups, epoch_rng_state
                )
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

        rows = []
        for mq, mc in grid:
            groups = group_train_records(
                self.train_records, max_questions=mq, max_chars=mc, rng=None
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

            worst_macro = [worst_g] * self.batch_size
            opt = torch.optim.AdamW(self.model.trainable_parameters(), lr=0.0, weight_decay=0.0)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
            if self.gradient_checkpointing:
                self.model.enable_checkpointing()
            self.model.train()
            if self.model.net.graphs is not None:
                self.model.net.graphs = None
            free_gpu()
            torch.cuda.reset_peak_memory_stats()

            try:
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
                "mean_q_per_group": round(len(self.train_records) / len(groups), 2),
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
