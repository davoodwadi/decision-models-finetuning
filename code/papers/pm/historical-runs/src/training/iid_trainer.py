import math
import random
import traceback

import pandas as pd
import torch

from data.base import _record_chars, make_view, release_cuda_graphs
from training.plotter import TrainingPlotter
from training.trainer import GroupedTrainer, free_gpu


def _iid_row_chars(r):
    """Estimated formatted character length of a full single-record `(persona, question)` prompt."""
    return 16 + len(r["persona"]) + _record_chars(r)


class IIDTrainer(GroupedTrainer):
    """Ungrouped i.i.d. trainer where each step draws `batch_size * grad_accum_steps` random `(persona, question)` records."""

    def __init__(
        self,
        *,
        train_records,
        eval_records,
        save_dir,
        eval_max_questions,
        eval_max_chars,
        model=None,
        base_model=None,
        init_from=None,
        lora_rank=64,
        device="cuda",
        weights_dtype="bf16",
        use_bf16=True,
        eval_bf16=None,
        gradient_checkpointing=True,
        epochs=2,
        lr=2e-4,
        lr_schedule="onecycle",
        min_lr=None,
        decay_pct=0.2,
        weight_decay=0.01,
        warmup_steps=1000,
        pct_start=None,
        batch_size=26,
        grad_accum_steps=12,
        sort_within_step=True,
        max_chars_per_microbatch=None,
        val_every=300,
        save_every=None,
        eval_at_start=True,
        samples_per_epoch=None,
        seed=0,
        resume=False,
        plotter=None,
        free_model_on_exit=False,
    ):
        if eval_max_questions is None or eval_max_questions < 1:
            raise ValueError(f"eval_max_questions is required and must be >= 1, got {eval_max_questions!r}")
        if eval_max_chars is None or eval_max_chars < 1:
            raise ValueError(f"eval_max_chars is required and must be >= 1, got {eval_max_chars!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps!r}")
        if max_chars_per_microbatch is not None and max_chars_per_microbatch < 1:
            raise ValueError(
                f"max_chars_per_microbatch must be >= 1 or None, got {max_chars_per_microbatch!r}"
            )

        self.sort_within_step = bool(sort_within_step)
        self.max_chars_per_microbatch = (
            int(max_chars_per_microbatch) if max_chars_per_microbatch is not None else None
        )

        super().__init__(
            train_records=train_records,
            eval_records=eval_records,
            save_dir=save_dir,
            model=model,
            base_model=base_model,
            init_from=init_from,
            lora_rank=lora_rank,
            device=device,
            weights_dtype=weights_dtype,
            use_bf16=use_bf16,
            eval_bf16=eval_bf16,
            gradient_checkpointing=gradient_checkpointing,
            epochs=epochs,
            lr=lr,
            lr_schedule=lr_schedule,
            min_lr=min_lr,
            decay_pct=decay_pct,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            pct_start=pct_start,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            max_questions_per_group=None,
            max_chars_per_group=None,
            eval_max_questions=eval_max_questions,
            eval_max_chars=eval_max_chars,
            val_every=val_every,
            save_every=save_every,
            eval_at_start=eval_at_start,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
            resume=resume,
            plotter=plotter,
            free_model_on_exit=free_model_on_exit,
        )

    def _epoch_groups_for(self, epoch_idx):
        """Return a flat list of i.i.d. shuffled records (optionally length-sorted within each step window)."""
        epoch_rng = random.Random(f"{self.seed}:epoch:{epoch_idx}")
        recs = (
            epoch_rng.sample(self.train_records, self.n_per_epoch)
            if self.samples_per_epoch is not None
            else list(self.train_records)
        )
        if len(recs) > 1:
            epoch_rng.shuffle(recs)
        step_size = self.batch_size * self.grad_accum_steps
        if self.sort_within_step and step_size > 1:
            for i in range(0, len(recs), step_size):
                recs[i : i + step_size] = sorted(recs[i : i + step_size], key=_iid_row_chars)
        return recs

    def _recompute_schedule(self):
        """Compute record counts, steps_per_epoch, total_steps, and warmup pct_start."""
        self.effective_groups_bs = self.batch_size * self.grad_accum_steps
        self.n_per_epoch = (
            min(self.samples_per_epoch, len(self.train_records))
            if self.samples_per_epoch is not None
            else len(self.train_records)
        )
        self.groups_per_epoch = self.n_per_epoch
        self.steps_per_epoch = math.ceil(self.n_per_epoch / self.effective_groups_bs)
        self.total_steps = self.epochs * self.steps_per_epoch
        self.mean_questions_per_group = 1.0

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

    def set_grouping(self, *args, **kwargs):
        raise NotImplementedError("set_grouping() is not applicable to IIDTrainer; configure batch_size and grad_accum_steps")

    def _split_microbatch(self, records):
        if self.max_chars_per_microbatch is None or len(records) <= 1:
            return [records]
        sub_batches = []
        cur = []
        cur_max = 0
        for r in records:
            r_chars = _iid_row_chars(r)
            nxt_max = max(cur_max, r_chars)
            if cur and (len(cur) + 1) * nxt_max > self.max_chars_per_microbatch:
                sub_batches.append(cur)
                cur = []
                cur_max = 0
                nxt_max = r_chars
            cur.append(r)
            cur_max = nxt_max
        if cur:
            sub_batches.append(cur)
        return sub_batches

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
        """Run one optimizer step over `macro_batch` of flat i.i.d. records using `shared_prefix=False`."""
        opt = optimizer if optimizer is not None else self.optimizer
        sched = scheduler if scheduler is not None else self.scheduler
        cur_step = step if step is not None else self.step
        max_steps = total_steps if total_steps is not None else self.total_steps
        step_rng = rng if rng is not None else self.rng

        total_questions = len(macro_batch)
        step_kls = []
        for sub_start in range(0, total_questions, self.batch_size):
            mb_records = macro_batch[sub_start : sub_start + self.batch_size]
            for sub_records in self._split_microbatch(mb_records):
                prepared = [self.model.prepare(make_view(r, rng=step_rng)) for r in sub_records]
                with self._autocast():
                    logits_batch = self.model.forward([enc for enc, _ in prepared], shared_prefix=False)

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

    def _run_signature(self):
        sig = {
            "trainer": "iid",
            "base": self._model_base,
            "lora_rank": self.lora_rank,
            "weights_dtype": self.weights_dtype,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "pct_start": round(self.pct_start, 8),
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "sort_within_step": self.sort_within_step,
            "max_chars_per_microbatch": self.max_chars_per_microbatch,
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

    def describe(self):
        return {
            "trainer": "iid",
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
            "questions_per_step": self.effective_groups_bs,
            "sort_within_step": self.sort_within_step,
            "max_chars_per_microbatch": self.max_chars_per_microbatch,
            "val_every": self.val_every,
            "save_every": self.save_every,
            "samples_per_epoch": self.samples_per_epoch,
            "weights_dtype": self.weights_dtype,
            "use_bf16": self.use_bf16,
            "eval_bf16": self.eval_bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "eval_max_questions": self.eval_max_questions,
            "eval_max_chars": self.eval_max_chars,
            "records_per_epoch": self.n_per_epoch,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "warmup_steps": int(round(self.pct_start * self.total_steps)),
            "pct_start": round(self.pct_start, 5),
        }

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
                "trainer": "iid",
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "pct_start": self.pct_start,
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "questions_per_step": self.effective_groups_bs,
                "sort_within_step": self.sort_within_step,
                "max_chars_per_microbatch": self.max_chars_per_microbatch,
                "weights_dtype": self.weights_dtype,
                "use_bf16": self.use_bf16,
                "eval_bf16": self.eval_bf16,
                "gradient_checkpointing": self.gradient_checkpointing,
                "epochs": self.epochs,
                "samples_per_epoch": self.samples_per_epoch,
                "seed": self.seed,
            },
        }

    def probe_vram(self, batch_sizes, *, stop_on_oom=True):
        """Sweep candidate `batch_size` values on the longest records to measure steady-state peak VRAM."""
        if self.model is None:
            raise ValueError("self.model is None")

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_vram_gb:.1f} GiB total)")

        probe_records = (
            random.Random(f"{self.seed}:epoch:0").sample(self.train_records, self.n_per_epoch)
            if self.samples_per_epoch is not None
            else self.train_records
        )
        max_bs = max(batch_sizes)
        longest_records = sorted(probe_records, key=_iid_row_chars, reverse=True)[:max_bs]
        longest_lens = [len(self.model.prepare(make_view(r))[0]["ids"]) for r in longest_records]

        orig_bs = self.batch_size
        rows = []
        try:
            for bs in batch_sizes:
                if bs < 1:
                    raise ValueError(f"batch_size in batch_sizes must be >= 1, got {bs!r}")
                self.batch_size = bs
                worst_mb = longest_records[:bs]
                worst_L = max(longest_lens[:bs])
                worst_padded_toks = bs * worst_L
                worst_macro = worst_mb * min(2, self.grad_accum_steps)

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

                questions_per_step = bs * self.grad_accum_steps
                steps_per_epoch = math.ceil(len(probe_records) / questions_per_step)
                row = {
                    "batch_size": bs,
                    "grad_accum_steps": self.grad_accum_steps,
                    "questions_per_step": questions_per_step,
                    "steps_per_epoch": steps_per_epoch,
                    f"total_steps_{self.epochs}ep": self.epochs * steps_per_epoch,
                    "worst_(B, L_max)": f"({bs}, {worst_L})",
                    "worst_padded_toks": worst_padded_toks,
                    "peak_alloc_GiB": round(peak_alloc_gb, 2),
                    "peak_reserved_GiB": round(peak_res_gb, 2),
                    "status": status,
                }
                rows.append(row)
                print(
                    f"bs={bs:3d} (q/step={questions_per_step:4d}) -> {row['status']:3s} | "
                    f"peak_alloc={row['peak_alloc_GiB']:5.2f} GiB (reserved={row['peak_reserved_GiB']:5.2f} GiB) | "
                    f"worst={row['worst_(B, L_max)']} ({row['worst_padded_toks']:,} toks) | "
                    f"{self.epochs}ep_steps={row[f'total_steps_{self.epochs}ep']:,}"
                )
                if status == "OOM" and stop_on_oom:
                    break
        finally:
            self.batch_size = orig_bs
            self.effective_groups_bs = self.batch_size * self.grad_accum_steps

        return pd.DataFrame(rows)
