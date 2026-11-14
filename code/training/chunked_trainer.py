"""Decoupled logical-persona grouped and marginal trainers with VRAM micro-chunking."""

from collections import defaultdict
import math
import random
import traceback

import pandas as pd
import torch

from data.base import ALL_PERSONA_NAMES, _record_chars, release_cuda_graphs
from data.marginals import (
    DEFAULT_K_SCHEDULE,
    PersonaMarginalLookup,
    augment_group_with_marginals,
    validate_k_schedule,
)
from training.plotter import TrainingPlotter
from training.trainer import GroupedTrainer, free_gpu


def _validate_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return value


def _build_question_spec(rec):
    qid = rec["qid"]
    q_type = rec["type"]
    option_texts = rec["option_texts"]
    option_descs = rec["option_descriptions"] if "option_descriptions" in rec else None
    human_dist = rec["human_dist"]

    if q_type == "score":
        if len(option_texts) != len(human_dist):
            raise ValueError(f"{qid}: option_texts ({len(option_texts)}) != human_dist ({len(human_dist)})")
        criteria = option_texts
        target = {str(i): float(p) for i, p in enumerate(human_dist)}
    elif q_type == "choice":
        if option_descs is not None:
            criteria = dict(zip(option_texts, option_descs, strict=True))
        else:
            criteria = {text: None for text in option_texts}
        if len(criteria) != len(option_texts):
            raise ValueError(f"{qid}: duplicate option_texts in choice question: {option_texts!r}")
        target = {text: float(p) for text, p in zip(option_texts, human_dist, strict=True)}
    else:
        raise ValueError(f"{qid}: unexpected question type {q_type!r}")

    return {
        "type": q_type,
        "instructions": rec["question"],
        "criteria": criteria,
        "target": target,
    }


def sample_persona_state(personas, *, rng=None, persona_names=ALL_PERSONA_NAMES):
    """Format `personas` into a Kev `state` string, sampling attribute order at most once via `rng`."""
    if not isinstance(personas, dict) or not personas:
        raise ValueError(f"personas must be a non-empty dict, got {personas!r}")
    items = list(personas.items())
    if rng is not None and len(items) > 1:
        rng.shuffle(items)
    return "\n".join(f"{persona_names[c]}: {v}" for c, v in items)


def make_state_chunk_view(state, chunk_records):
    """Build a multi-question Kev view from a pre-sampled `state` string and one micro-chunk of records."""
    if not isinstance(state, str) or not state:
        raise ValueError(f"state must be a non-empty string, got {state!r}")
    if not chunk_records:
        raise ValueError("chunk_records must not be empty")

    first_personas = chunk_records[0]["personas"]
    if not first_personas:
        raise ValueError(f"Expected non-empty personas dict for qid={chunk_records[0]['qid']!r}")

    questions = {}
    for idx, rec in enumerate(chunk_records):
        if rec["personas"] != first_personas:
            raise ValueError("All records in chunk_records must share the exact same personas dict")
        questions[f"{idx}:{rec['qid']}"] = _build_question_spec(rec)

    return {"state": state, "questions": questions}


def group_persona_logical(records, *, questions_per_group, rng=None):
    """Partition `records` by exact full persona into battery-mixed logical groups of up to `questions_per_group`."""
    _validate_positive_int(questions_per_group, "questions_per_group")
    if not records:
        raise ValueError("records must not be empty")

    by_state = defaultdict(list)
    for r in records:
        personas = r["personas"]
        if not personas:
            raise ValueError(f"Expected non-empty personas dict for qid={r['qid']!r}")
        key = tuple(sorted(personas.items()))
        by_state[key].append(r)

    groups = []
    for state_recs in by_state.values():
        state_recs = list(state_recs)
        if rng is not None and len(state_recs) > 1:
            rng.shuffle(state_recs)

        n_recs = len(state_recs)
        n_groups = math.ceil(n_recs / questions_per_group)
        base, rem = divmod(n_recs, n_groups)
        state_groups = []
        offset = 0
        for g_idx in range(n_groups):
            size = base + (1 if g_idx < rem else 0)
            state_groups.append(state_recs[offset : offset + size])
            offset += size

        groups.extend(state_groups)

    if rng is not None and len(groups) > 1:
        rng.shuffle(groups)
    return groups


def build_logical_marginal_groups(
    train_records,
    lookup,
    *,
    questions_per_group,
    rng,
    k_schedule=DEFAULT_K_SCHEDULE,
):
    """Build 1 epoch of battery-mixed logical persona groups augmented with stateless `k_schedule` marginals."""
    if rng is None:
        raise ValueError("rng is required for build_logical_marginal_groups")
    if not isinstance(lookup, PersonaMarginalLookup):
        raise TypeError(
            f"lookup must be a PersonaMarginalLookup instance, got {type(lookup).__name__}"
        )
    schedule = validate_k_schedule(k_schedule)
    epoch_base_groups = group_persona_logical(
        train_records,
        questions_per_group=questions_per_group,
        rng=rng,
    )

    marginal_groups = []
    for base_group in epoch_base_groups:
        full_persona = base_group[0]["personas"]
        if not full_persona:
            raise ValueError(f"Expected non-empty personas dict in base_group[0] for qid={base_group[0]['qid']!r}")
        k_choice = rng.choice(schedule)
        k_traits = len(full_persona) if k_choice == "full" else min(k_choice, len(full_persona))
        chosen_traits = set(rng.sample(list(full_persona.keys()), k_traits))
        subpersona = {col: val for col, val in full_persona.items() if col in chosen_traits}
        marginal_groups.append(augment_group_with_marginals(base_group, subpersona, lookup))
    return marginal_groups


def split_logical_group_into_microchunks(logical_group, *, micro_max_questions, micro_max_chars):
    """Deterministically pack `logical_group` into length-sorted shared-prefix micro-chunks without using any RNG."""
    if not logical_group:
        raise ValueError("logical_group must not be empty")
    _validate_positive_int(micro_max_questions, "micro_max_questions")
    _validate_positive_int(micro_max_chars, "micro_max_chars")

    first_personas = logical_group[0]["personas"]
    if not first_personas:
        raise ValueError(f"Expected non-empty personas dict for qid={logical_group[0]['qid']!r}")

    sized = []
    for idx, rec in enumerate(logical_group):
        if rec["personas"] != first_personas:
            raise ValueError("All records in logical_group must share the exact same personas dict")
        sized.append((idx, _record_chars(rec), rec))
    sized.sort(key=lambda item: (item[1], item[0]))

    chunks = []
    cur = []
    for _, q_chars, rec in sized:
        if cur and (len(cur) >= micro_max_questions or (len(cur) + 1) * q_chars > micro_max_chars):
            chunks.append(cur)
            cur = []
        cur.append(rec)
    if cur:
        chunks.append(cur)
    return chunks


class ChunkedGroupedTrainer(GroupedTrainer):
    """Persona-grouped trainer decoupling logical `(questions_per_group, personas_per_step)` from VRAM micro-chunks."""

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
        epochs=1,
        lr=5e-5,
        min_lr=None,
        lr_schedule="onecycle",
        decay_pct=0.20,
        weight_decay=0.01,
        warmup_steps=2000,
        pct_start=None,
        questions_per_group=80,
        personas_per_step=12,
        micro_max_questions=32,
        micro_max_chars=20_000,
        micro_batch_size=1,
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
        if not train_records:
            raise ValueError("train_records must not be empty")
        self.questions_per_group = _validate_positive_int(questions_per_group, "questions_per_group")
        self.personas_per_step = _validate_positive_int(personas_per_step, "personas_per_step")
        self.micro_max_questions = _validate_positive_int(micro_max_questions, "micro_max_questions")
        self.micro_max_chars = _validate_positive_int(micro_max_chars, "micro_max_chars")
        self.micro_batch_size = _validate_positive_int(micro_batch_size, "micro_batch_size")
        self._chunked_explicit_eval_max_questions = (
            _validate_positive_int(eval_max_questions, "eval_max_questions")
            if eval_max_questions is not None
            else None
        )
        self._chunked_explicit_eval_max_chars = (
            _validate_positive_int(eval_max_chars, "eval_max_chars")
            if eval_max_chars is not None
            else None
        )
        resolved_eval_max_questions = (
            self._chunked_explicit_eval_max_questions
            if self._chunked_explicit_eval_max_questions is not None
            else self.micro_max_questions * 2
        )
        resolved_eval_max_chars = (
            self._chunked_explicit_eval_max_chars
            if self._chunked_explicit_eval_max_chars is not None
            else self.micro_max_chars * 2
        )

        super().__init__(
            train_records=train_records,
            eval_records=eval_records,
            save_dir=save_dir,
            model=model,
            base_model=base_model,
            lora_rank=lora_rank,
            init_from=init_from,
            device=device,
            weights_dtype=weights_dtype,
            use_bf16=use_bf16,
            eval_bf16=eval_bf16,
            gradient_checkpointing=gradient_checkpointing,
            epochs=epochs,
            lr=lr,
            min_lr=min_lr,
            lr_schedule=lr_schedule,
            decay_pct=decay_pct,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            pct_start=pct_start,
            batch_size=1,
            grad_accum_steps=self.personas_per_step,
            max_questions_per_group=None,
            max_chars_per_group=None,
            eval_max_questions=resolved_eval_max_questions,
            eval_max_chars=resolved_eval_max_chars,
            val_every=val_every,
            save_every=save_every,
            eval_at_start=eval_at_start,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
            resume=resume,
            plotter=plotter,
            free_model_on_exit=free_model_on_exit,
        )

    def _logical_base_groups_for(self, epoch_idx):
        epoch_rng = random.Random(f"{self.seed}:epoch:{epoch_idx}")
        sampled = (
            epoch_rng.sample(self.train_records, self.n_per_epoch)
            if self.samples_per_epoch is not None
            else self.train_records
        )
        return group_persona_logical(
            sampled,
            questions_per_group=self.questions_per_group,
            rng=epoch_rng,
        )

    def _epoch_groups_for(self, epoch_idx):
        return self._logical_base_groups_for(epoch_idx)

    def _recompute_microchunk_stats(self, epoch0_groups):
        self.epoch0_micro_chunks = sum(
            len(
                split_logical_group_into_microchunks(
                    g,
                    micro_max_questions=self.micro_max_questions,
                    micro_max_chars=self.micro_max_chars,
                )
            )
            for g in epoch0_groups
        )
        self.mean_micro_chunks_per_group = self.epoch0_micro_chunks / max(1, len(epoch0_groups))

    def _recompute_schedule(self):
        """Compute logical group counts, steps_per_epoch, total_steps, warmup pct_start, and micro-chunk stats."""
        self.effective_groups_bs = self.personas_per_step
        self.n_per_epoch = (
            min(self.samples_per_epoch, len(self.train_records))
            if self.samples_per_epoch is not None
            else len(self.train_records)
        )
        epoch0_groups = self._logical_base_groups_for(0)
        self.groups_per_epoch = len(epoch0_groups)
        self.steps_per_epoch = math.ceil(self.groups_per_epoch / self.effective_groups_bs)
        if self.samples_per_epoch is None:
            self.total_steps = self.epochs * self.steps_per_epoch
        else:
            epoch_group_counts = [len(epoch0_groups)] + [
                len(self._logical_base_groups_for(e)) for e in range(1, self.epochs)
            ]
            self.total_steps = sum(
                math.ceil(c / self.effective_groups_bs) for c in epoch_group_counts
            )
        self.mean_questions_per_group = self.n_per_epoch / max(1, self.groups_per_epoch)
        self._recompute_microchunk_stats(epoch0_groups)
        del epoch0_groups

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
        raise NotImplementedError(
            "set_grouping() is not applicable to ChunkedGroupedTrainer; use set_microbatching(...) or set_logical_grouping(...)"
        )

    def set_logical_grouping(self, questions_per_group, personas_per_step=None):
        """Update logical grouping parameters before training and recompute `total_steps`."""
        if self.resume:
            raise ValueError(
                "Cannot call set_logical_grouping() when resume=True; logical grouping must match the saved run"
            )
        self.questions_per_group = _validate_positive_int(questions_per_group, "questions_per_group")
        if personas_per_step is not None:
            self.personas_per_step = _validate_positive_int(personas_per_step, "personas_per_step")
            self.grad_accum_steps = self.personas_per_step
        self._recompute_schedule()
        return self.describe()

    def set_microbatching(
        self,
        micro_max_questions,
        micro_max_chars,
        *,
        micro_batch_size=None,
        gradient_checkpointing=None,
        eval_max_questions=None,
        eval_max_chars=None,
    ):
        """Update VRAM micro-chunk execution bounds without changing `total_steps` or logical batch composition."""
        self.micro_max_questions = _validate_positive_int(micro_max_questions, "micro_max_questions")
        self.micro_max_chars = _validate_positive_int(micro_max_chars, "micro_max_chars")
        if micro_batch_size is not None:
            self.micro_batch_size = _validate_positive_int(micro_batch_size, "micro_batch_size")
        if gradient_checkpointing is not None:
            self.gradient_checkpointing = bool(gradient_checkpointing)
            if self.model is not None:
                if self.gradient_checkpointing:
                    self.model.enable_checkpointing()
                else:
                    self.model.net.lm.gradient_checkpointing_disable()
        if eval_max_questions is not None:
            self._chunked_explicit_eval_max_questions = _validate_positive_int(
                eval_max_questions, "eval_max_questions"
            )
        if eval_max_chars is not None:
            self._chunked_explicit_eval_max_chars = _validate_positive_int(
                eval_max_chars, "eval_max_chars"
            )
        self.eval_max_questions = (
            self._chunked_explicit_eval_max_questions
            if self._chunked_explicit_eval_max_questions is not None
            else self.micro_max_questions * 2
        )
        self.eval_max_chars = (
            self._chunked_explicit_eval_max_chars
            if self._chunked_explicit_eval_max_chars is not None
            else self.micro_max_chars * 2
        )
        epoch0_groups = self._logical_base_groups_for(0)
        self._recompute_microchunk_stats(epoch0_groups)
        del epoch0_groups
        return self.describe()

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
        """Run one optimizer step over `macro_batch` of logical persona groups via micro-chunk accumulation."""
        if not macro_batch:
            raise ValueError("macro_batch must not be empty")
        opt = optimizer if optimizer is not None else self.optimizer
        sched = scheduler if scheduler is not None else self.scheduler
        cur_step = step if step is not None else self.step
        max_steps = total_steps if total_steps is not None else self.total_steps
        step_rng = rng if rng is not None else self.rng

        total_questions = sum(len(g) for g in macro_batch)
        if total_questions < 1:
            raise ValueError("macro_batch must contain at least 1 question")

        chunk_views = []
        for logical_group in macro_batch:
            if not logical_group:
                raise ValueError("Each logical_group in macro_batch must be non-empty")
            # Sample persona attribute order strictly once per logical group (never per micro-chunk)
            state = sample_persona_state(logical_group[0]["personas"], rng=step_rng)
            chunks = split_logical_group_into_microchunks(
                logical_group,
                micro_max_questions=self.micro_max_questions,
                micro_max_chars=self.micro_max_chars,
            )
            for chunk in chunks:
                max_c_chars = max(_record_chars(r) for r in chunk)
                chunk_views.append((max_c_chars, len(chunk_views), make_state_chunk_view(state, chunk)))

        if self.micro_batch_size > 1 and len(chunk_views) > 1:
            chunk_views.sort(key=lambda item: (item[0], item[1]))

        step_kls = []
        for sub_start in range(0, len(chunk_views), self.micro_batch_size):
            batch_views = [item[2] for item in chunk_views[sub_start : sub_start + self.micro_batch_size]]
            prepared = [self.model.prepare(v) for v in batch_views]
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

        if len(step_kls) != total_questions:
            raise AssertionError(f"Expected {total_questions} question KLs, got {len(step_kls)}")

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
            "trainer": "chunked_grouped",
            "base": self._model_base,
            "lora_rank": self.lora_rank,
            "weights_dtype": self.weights_dtype,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "pct_start": round(self.pct_start, 8),
            "questions_per_group": self.questions_per_group,
            "personas_per_step": self.personas_per_step,
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
            "trainer": "ChunkedGroupedTrainer",
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
            "questions_per_group": self.questions_per_group,
            "personas_per_step": self.personas_per_step,
            "micro_max_questions": self.micro_max_questions,
            "micro_max_chars": self.micro_max_chars,
            "micro_batch_size": self.micro_batch_size,
            "val_every": self.val_every,
            "save_every": self.save_every,
            "samples_per_epoch": self.samples_per_epoch,
            "weights_dtype": self.weights_dtype,
            "use_bf16": self.use_bf16,
            "eval_bf16": self.eval_bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "eval_max_questions": self.eval_max_questions,
            "eval_max_chars": self.eval_max_chars,
            "groups_per_epoch": self.groups_per_epoch,
            "epoch0_micro_chunks": self.epoch0_micro_chunks,
            "mean_micro_chunks_per_group": round(self.mean_micro_chunks_per_group, 2),
            "mean_questions_per_group": round(self.mean_questions_per_group, 2),
            "mean_questions_per_step": round(self.personas_per_step * self.mean_questions_per_group, 1),
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
                "trainer": "chunked_grouped",
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "pct_start": self.pct_start,
                "questions_per_group": self.questions_per_group,
                "personas_per_step": self.personas_per_step,
                "micro_max_questions": self.micro_max_questions,
                "micro_max_chars": self.micro_max_chars,
                "micro_batch_size": self.micro_batch_size,
                "weights_dtype": self.weights_dtype,
                "use_bf16": self.use_bf16,
                "eval_bf16": self.eval_bf16,
                "gradient_checkpointing": self.gradient_checkpointing,
                "epochs": self.epochs,
                "samples_per_epoch": self.samples_per_epoch,
                "seed": self.seed,
            },
        }

    def probe_vram(self, micro_grid, *, stop_on_oom=True):
        """Sweep `(micro_max_questions, micro_max_chars[, micro_batch_size])` while holding `total_steps` fixed."""
        if self.model is None:
            raise ValueError("self.model is None")
        if not micro_grid:
            raise ValueError("micro_grid must not be empty")

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_vram_gb:.1f} GiB total)")

        logical_groups = self._logical_base_groups_for(0)
        orig_mq = self.micro_max_questions
        orig_mc = self.micro_max_chars
        orig_mb = self.micro_batch_size

        rows = []
        try:
            for entry in micro_grid:
                if len(entry) == 2:
                    mq, mc = entry
                    mb = orig_mb
                elif len(entry) == 3:
                    mq, mc, mb = entry
                else:
                    raise ValueError(
                        f"Each micro_grid entry must be (mq, mc) or (mq, mc, mb), got {entry!r}"
                    )
                _validate_positive_int(mq, "micro_max_questions")
                _validate_positive_int(mc, "micro_max_chars")
                _validate_positive_int(mb, "micro_batch_size")

                all_chunks = []
                for g in logical_groups:
                    all_chunks.extend(
                        split_logical_group_into_microchunks(
                            g, micro_max_questions=mq, micro_max_chars=mc
                        )
                    )

                by_padded_chars = sorted(
                    all_chunks,
                    key=lambda c: len(c) * max(_record_chars(r) for r in c),
                    reverse=True,
                )[:15]
                by_len = sorted(
                    all_chunks,
                    key=lambda c: (len(c), max(_record_chars(r) for r in c)),
                    reverse=True,
                )[:15]
                candidates = by_padded_chars + by_len

                worst_c, worst_tokens, worst_shape = None, -1, None
                for c in candidates:
                    state = sample_persona_state(c[0]["personas"], rng=None)
                    enc, _ = self.model.prepare(make_state_chunk_view(state, c))
                    Ls = enc["seg"].count(0)
                    q = max(enc["seg"])
                    Lb = max(enc["seg"].count(s) for s in range(1, q + 1))
                    padded_tokens = Ls + q * Lb
                    if padded_tokens > worst_tokens:
                        worst_tokens = padded_tokens
                        worst_c = c
                        worst_shape = (q, Ls, Lb)

                self.micro_max_questions = mq
                self.micro_max_chars = mc
                self.micro_batch_size = mb

                worst_macro = [worst_c] * (mb * 2)
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
                row = {
                    "questions_per_group": self.questions_per_group,
                    "personas_per_step": self.personas_per_step,
                    "micro_max_questions": mq,
                    "micro_max_chars": mc,
                    "micro_batch_size": mb,
                    "logical_groups": len(logical_groups),
                    "micro_chunks": len(all_chunks),
                    "chunks_per_group": round(len(all_chunks) / len(logical_groups), 2),
                    "steps_per_epoch": self.steps_per_epoch,
                    f"total_steps_{self.epochs}ep": self.total_steps,
                    "worst_(Q, Ls, Lb)": f"({q}, {Ls}, {Lb})",
                    "worst_padded_toks": worst_tokens,
                    "peak_alloc_GiB": round(peak_alloc_gb, 2),
                    "peak_reserved_GiB": round(peak_res_gb, 2),
                    "status": status,
                }
                rows.append(row)
                print(
                    f"micro=({mq:2d}, {mc:6d}, mb={mb}) -> {row['status']:3s} | "
                    f"peak_alloc={row['peak_alloc_GiB']:5.2f} GiB (reserved={row['peak_reserved_GiB']:5.2f} GiB) | "
                    f"worst={row['worst_(Q, Ls, Lb)']} ({row['worst_padded_toks']:,} toks) | "
                    f"chunks={len(all_chunks):,} ({row['chunks_per_group']:.2f}/grp) | "
                    f"{self.epochs}ep_steps={self.total_steps:,}"
                )
                if status == "OOM" and stop_on_oom:
                    break
        finally:
            self.micro_max_questions = orig_mq
            self.micro_max_chars = orig_mc
            self.micro_batch_size = orig_mb

        return pd.DataFrame(rows)


class ChunkedMarginalTrainer(ChunkedGroupedTrainer):
    """Decoupled logical-persona trainer supervising exact empirical `k`-trait marginals via VRAM micro-chunks."""

    def __init__(
        self,
        *,
        train_records,
        eval_records,
        save_dir,
        marginal_lookup,
        k_schedule=DEFAULT_K_SCHEDULE,
        **kwargs,
    ):
        if not isinstance(marginal_lookup, PersonaMarginalLookup):
            raise TypeError(
                f"marginal_lookup must be a PersonaMarginalLookup instance, got {type(marginal_lookup).__name__}"
            )
        self.marginal_lookup = marginal_lookup
        self.k_schedule = validate_k_schedule(k_schedule)

        super().__init__(
            train_records=train_records,
            eval_records=eval_records,
            save_dir=save_dir,
            **kwargs,
        )

    def _epoch_groups_for(self, epoch_idx):
        epoch_rng = random.Random(f"{self.seed}:epoch:{epoch_idx}")
        sampled = (
            epoch_rng.sample(self.train_records, self.n_per_epoch)
            if self.samples_per_epoch is not None
            else self.train_records
        )
        return build_logical_marginal_groups(
            sampled,
            self.marginal_lookup,
            questions_per_group=self.questions_per_group,
            rng=epoch_rng,
            k_schedule=self.k_schedule,
        )

    def _run_signature(self):
        sig = super()._run_signature()
        sig["trainer"] = "chunked_marginal"
        sig["k_schedule"] = list(self.k_schedule)
        sig["include_country"] = self.marginal_lookup.include_country
        sig["num_unique_personas"] = self.marginal_lookup.num_unique_personas
        return sig

    def describe(self):
        info = super().describe()
        return {
            **info,
            "trainer": "ChunkedMarginalTrainer",
            "k_schedule": self.k_schedule,
            "unique_personas": self.marginal_lookup.num_unique_personas,
            "trait_value_masks": len(self.marginal_lookup.trait_masks),
        }

    def _build_extra(self, epoch_idx, train_kl, val_ce=None, val_kl=None, summaries=None):
        extra = super()._build_extra(
            epoch_idx, train_kl, val_ce=val_ce, val_kl=val_kl, summaries=summaries
        )
        extra["settings"]["trainer"] = "chunked_marginal"
        extra["settings"]["k_schedule"] = list(self.k_schedule)
        return extra
