"""Modular base trainer for KevPointerModel with pluggable epoch batching subclasses."""

from abc import ABC, abstractmethod
import contextlib
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm.auto import tqdm

from data.persona_spec import PersonaAttribute
from data.records import prompt_map_from_spec
from models.kev_pointer import KevPointerModel


def validate_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return value


def free_gpu() -> None:
    """Clear Python exception frames, run GC, and release cached CUDA blocks."""
    if hasattr(sys, "last_traceback") and sys.last_traceback is not None:
        traceback.clear_frames(sys.last_traceback)
        sys.last_traceback = None
    for attr in ("last_value", "last_type", "last_exc"):
        if hasattr(sys, attr):
            setattr(sys, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_cuda_graphs(model: KevPointerModel) -> None:
    """Break closure reference cycles inside `CudaGraphs` and release `model.net.graphs`."""
    graphs = model.net.graphs
    if graphs is not None:
        graphs.pending.clear()
        graphs.graphs.clear()
        graphs.failed.clear()
        graphs.eager_runs.clear()
        model.net.graphs = None


def save_checkpoint(model: KevPointerModel, save_dir: str | Path, name: str, extra: dict) -> Path:
    """Atomically write a Kev checkpoint and `metrics.json` to `save_dir / name`."""
    root = Path(save_dir)
    folder, tmp, old = root / name, root / f"{name}-tmp", root / f"{name}-old"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    model.save(tmp, extra=extra)
    (tmp / "metrics.json").write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")
    if folder.exists():
        folder.rename(old)
    tmp.rename(folder)
    shutil.rmtree(old, ignore_errors=True)
    return folder


def load_checkpoint(
    ckpt_dir: str | Path,
    device: str | torch.device = "cuda",
    *,
    trainable: bool = False,
    lora_rank: int | None = None,
    weights_dtype: str = "bf16",
    gradient_checkpointing: bool = False,
) -> KevPointerModel:
    """Load a saved Kev checkpoint for inference or continued training."""
    path = Path(ckpt_dir)
    if not (path / "head.pt").exists():
        raise FileNotFoundError(f"Checkpoint not found: {path / 'head.pt'}")
    free_gpu()
    if trainable:
        model = KevPointerModel.build(
            str(path), device=device, lora_rank=lora_rank, weights_dtype=weights_dtype
        )
        model.net.head.temperature = 1.0
        if gradient_checkpointing:
            model.enable_checkpointing()
        return model
    dtype = torch.bfloat16 if weights_dtype == "bf16" else torch.float32
    model = KevPointerModel.load(str(path), device=device, dtype=dtype)
    model.net.head.temperature = 1.0
    return model


@dataclass(frozen=True)
class MicroBatch:
    """One forward/backward call's Kev views and whether they share a persona prefix."""

    views: list[dict]
    shared_prefix: bool


@dataclass(frozen=True)
class StepBatch:
    """One optimizer step composed of one or more gradient-accumulated `MicroBatch`es."""

    micro_batches: list[MicroBatch]
    total_questions: int


class BaseTrainer(ABC):
    """Shared optimization, loss accumulation, LR scheduling, evaluation, and checkpointing loop.

    Subclasses define only how each epoch's records are turned into logical optimizer steps
    (`build_epoch_macro_batches`) and how one logical step is packed into VRAM micro-batches
    (`materialize_step`).
    """

    trainer_name: str = "base"

    def __init__(
        self,
        *,
        train_records: list[dict],
        spec: dict[str, PersonaAttribute],
        model: KevPointerModel | None = None,
        base_model: str | None = None,
        init_from: str | Path | None = None,
        lora_rank: int | None = 64,
        device: str | torch.device = "cuda",
        weights_dtype: str = "bf16",
        use_bf16: bool = True,
        gradient_checkpointing: bool = True,
        epochs: int = 1,
        lr: float = 1e-4,
        lr_schedule: str = "onecycle",
        min_lr: float | None = None,
        decay_pct: float = 0.2,
        weight_decay: float = 0.01,
        warmup_steps: int | None = None,
        pct_start: float | None = None,
        max_grad_norm: float = 1.0,
        seed: int = 0,
        save_dir: str | Path | None = None,
        save_every: int | None = None,
        eval_fn=None,
        eval_every: int | None = None,
        eval_at_start: bool = True,
        best_metric: str | None = None,
        show_progress: bool = True,
        free_model_on_exit: bool = False,
    ):
        if not train_records:
            raise ValueError("train_records must not be empty")
        if not spec:
            raise ValueError("spec must not be empty")
        if weights_dtype not in ("fp32", "bf16"):
            raise ValueError(f"weights_dtype must be 'fp32' or 'bf16', got {weights_dtype!r}")
        validate_positive_int(epochs, "epochs")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr!r}")
        if lr_schedule not in ("onecycle", "wsd", "linear", "constant"):
            raise ValueError(
                f"lr_schedule must be one of ('onecycle', 'wsd', 'linear', 'constant'), got {lr_schedule!r}"
            )
        if min_lr is not None:
            min_lr = float(min_lr)
            if min_lr < 0 or min_lr > lr:
                raise ValueError(f"min_lr must satisfy 0 <= min_lr <= lr ({lr}), got {min_lr!r}")
            if lr_schedule == "onecycle" and min_lr == 0.0:
                raise ValueError("min_lr must be > 0 when lr_schedule='onecycle' (or pass min_lr=None)")
        if lr_schedule == "wsd" and not (0.0 < decay_pct < 1.0):
            raise ValueError(f"decay_pct must satisfy 0 < decay_pct < 1 when lr_schedule='wsd', got {decay_pct!r}")
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm!r}")
        if save_every is not None:
            validate_positive_int(save_every, "save_every")
        if eval_every is not None:
            validate_positive_int(eval_every, "eval_every")
        if best_metric is not None and eval_fn is None:
            raise ValueError("best_metric requires eval_fn")

        self.train_records = train_records
        self.spec = spec
        self.prompt_map = prompt_map_from_spec(spec)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.weights_dtype = weights_dtype
        self.use_bf16 = bool(use_bf16)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.epochs = epochs
        self.lr = float(lr)
        self.lr_schedule = lr_schedule
        self.min_lr = min_lr
        self.decay_pct = float(decay_pct)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = warmup_steps
        self._explicit_pct_start = pct_start
        self.max_grad_norm = float(max_grad_norm)
        self.seed = seed
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.save_every = save_every
        self.eval_fn = eval_fn
        self.eval_every = eval_every
        self.eval_at_start = bool(eval_at_start)
        self.best_metric = best_metric
        self.show_progress = bool(show_progress)
        self.free_model_on_exit = bool(free_model_on_exit)

        if model is None:
            free_gpu()
            torch.manual_seed(self.seed)
            if init_from is not None:
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
                    str(base_model),
                    self.device,
                    lora_rank=lora_rank,
                    weights_dtype=self.weights_dtype,
                )
                self.model.net.head.temperature = 1.0
                if self.gradient_checkpointing:
                    self.model.enable_checkpointing()
            else:
                raise ValueError("Provide either `model`, `base_model`, or `init_from`")
        else:
            if lora_rank is not None and model.meta.lora != lora_rank:
                raise ValueError(
                    f"Provided model has lora_rank={model.meta.lora}, expected {lora_rank}"
                )
            self.model = model
            self.init_from = str(init_from) if init_from is not None else (
                str(base_model) if base_model is not None else str(model.meta.base)
            )
            self.model.net.head.temperature = 1.0
            if self.gradient_checkpointing:
                self.model.enable_checkpointing()

        self.lora_rank = self.model.meta.lora
        self.optimizer = None
        self.scheduler = None
        self.step = 0
        self.best_metric_val = float("inf")
        self._compute_schedule()

    @abstractmethod
    def build_epoch_macro_batches(self, epoch_idx: int, rng: random.Random) -> list:
        """Return the ordered list of logical macro-batches (one per optimizer step) for `epoch_idx`."""

    @abstractmethod
    def materialize_step(self, macro_batch, rng: random.Random) -> StepBatch:
        """Pack one logical macro-batch into one or more VRAM `MicroBatch`es."""

    @abstractmethod
    def _probe_param_name(self) -> str:
        """Name of the VRAM micro-batch parameter tuned by `find_max_micro_batch_size`."""

    @abstractmethod
    def _max_probe_size(self) -> int:
        """Upper bound on the micro-batch size meaningful for this trainer's logical batch."""

    @abstractmethod
    def _prepare_probe_context(self) -> dict:
        """Find the exact worst-case (longest tokenized) sequence/chunk in `train_records` once before probing."""

    @abstractmethod
    def _build_probe_micro_batch(self, candidate_size: int, ctx: dict) -> MicroBatch:
        """Build a worst-case `MicroBatch` of size `candidate_size` from the precomputed probe context."""

    @abstractmethod
    def _set_probed_micro_batch_size(self, size: int) -> None:
        """Apply the discovered safe micro-batch size onto this trainer."""

    def _probe_once(self, micro_batch: MicroBatch, *, vram_limit_bytes: int) -> tuple[bool, float | None, str]:
        """Run one forward+backward probe without stepping weights; return `(ok, peak_gb, status)`."""
        prepared = logits_batch = losses = loss = None
        free_gpu()
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            prepared = [self.model.prepare(v) for v in micro_batch.views]
            with self._autocast():
                logits_batch = self.model.forward(
                    [enc for enc, _ in prepared],
                    shared_prefix=micro_batch.shared_prefix,
                )
            losses = []
            for (_, targets), record_logits in zip(prepared, logits_batch, strict=True):
                for z, q in zip(record_logits, targets, strict=True):
                    t = q["target"].to(self.device)
                    logp = torch.log_softmax(z.float(), dim=-1)
                    losses.append(-(t * logp).sum())
            loss = torch.stack(losses).sum() / len(losses)
            loss.backward()
            torch.cuda.synchronize(self.device)
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_gb = peak_bytes / 1024**3
            if peak_bytes <= vram_limit_bytes:
                return True, peak_gb, "ok"
            return False, peak_gb, "exceeds_headroom"
        except torch.OutOfMemoryError:
            return False, None, "oom"
        finally:
            del prepared, logits_batch, losses, loss
            for p in self.model.trainable_parameters():
                p.grad = None
            free_gpu()

    def find_max_micro_batch_size(
        self,
        *,
        max_size: int | None = None,
        headroom_gb: float = 2.0,
        apply: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Binary-search the largest worst-case micro-batch that fits in VRAM with `headroom_gb` spare.

        Probes use the longest sequences in `train_records`, reserve fp32 AdamW state overhead plus
        `headroom_gb`, and never update model weights.
        """
        if self.model is None:
            raise ValueError("self.model is None")
        if self.device.type != "cuda":
            raise RuntimeError("find_max_micro_batch_size requires a CUDA device")
        if headroom_gb < 0:
            raise ValueError(f"headroom_gb must be >= 0, got {headroom_gb!r}")

        upper = self._max_probe_size()
        if max_size is not None:
            upper = min(upper, validate_positive_int(max_size, "max_size"))
        if upper < 1:
            raise ValueError(f"Probe upper bound must be >= 1, got {upper}")

        release_cuda_graphs(self.model)
        free_gpu()

        was_training = self.model.net.training
        if self.gradient_checkpointing:
            self.model.enable_checkpointing()
        self.model.train()

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        base_alloc = torch.cuda.memory_allocated(self.device)
        adam_bytes = 2 * sum(p.numel() * 4 for p in self.model.trainable_parameters())
        headroom_bytes = int(headroom_gb * 1024**3)
        vram_limit_bytes = base_alloc + free_bytes - adam_bytes - headroom_bytes
        if vram_limit_bytes <= base_alloc:
            raise RuntimeError(
                f"Insufficient free VRAM ({free_bytes / 1024**3:.2f} GiB) for AdamW "
                f"({adam_bytes / 1024**3:.2f} GiB) + headroom ({headroom_gb:.2f} GiB)"
            )

        param_name = self._probe_param_name()
        vram_limit_gb = vram_limit_bytes / 1024**3
        adam_gb = adam_bytes / 1024**3
        probe_ctx = self._prepare_probe_context()
        probes: list[dict] = []
        best_size = 0
        best_peak_gb: float | None = None

        if verbose:
            print(
                f"Probing max {param_name} (1..{upper}) | {probe_ctx['summary']} | "
                f"free={free_bytes / 1024**3:.2f} GiB, adam={adam_gb:.2f} GiB, "
                f"headroom={headroom_gb:.2f} GiB -> limit={vram_limit_gb:.2f} GiB "
                f"(gc={self.gradient_checkpointing})"
            )

        def run_candidate(sz: int) -> bool:
            nonlocal best_size, best_peak_gb
            mb = self._build_probe_micro_batch(sz, probe_ctx)
            ok, peak_gb, status = self._probe_once(mb, vram_limit_bytes=vram_limit_bytes)
            probes.append({"size": sz, "ok": ok, "status": status, "peak_vram_gb": peak_gb})
            if verbose:
                peak_str = f"{peak_gb:.2f} GiB" if peak_gb is not None else "OOM"
                print(f"  {param_name}={sz:3d} -> {status:<16s} (peak={peak_str})")
            if ok and sz > best_size:
                best_size = sz
                best_peak_gb = peak_gb
            return ok

        try:
            # 1. Exponential bracketing: 1, 2, 4, 8, ...
            lo = 0
            hi = None
            cur = 1
            while True:
                if run_candidate(cur):
                    lo = cur
                    if cur >= upper:
                        break
                    cur = min(upper, cur * 2)
                else:
                    hi = cur
                    break

            # 2. Binary search between (lo, hi)
            if hi is not None and lo > 0:
                left, right = lo + 1, hi - 1
                while left <= right:
                    mid = (left + right) // 2
                    if run_candidate(mid):
                        lo = mid
                        left = mid + 1
                    else:
                        right = mid - 1
        finally:
            self.model.train(was_training)
            for p in self.model.trainable_parameters():
                p.grad = None
            free_gpu()

        if best_size < 1:
            raise RuntimeError(
                f"Even {param_name}=1 exceeded VRAM budget ({vram_limit_gb:.2f} GiB). "
                "Try gradient_checkpointing=True or a smaller headroom_gb."
            )

        if apply:
            self._set_probed_micro_batch_size(best_size)

        if verbose:
            action = f"applied {param_name}={best_size}" if apply else f"recommended {param_name}={best_size}"
            print(f"=> {action} (worst-case peak={best_peak_gb:.2f} GiB <= {vram_limit_gb:.2f} GiB)")

        return {
            "param": param_name,
            "max_size": best_size,
            "peak_vram_gb": best_peak_gb,
            "vram_limit_gb": vram_limit_gb,
            "total_vram_gb": total_bytes / 1024**3,
            "adam_overhead_gb": adam_gb,
            "headroom_gb": headroom_gb,
            "worst_case": probe_ctx["summary"],
            "probes": probes,
        }

    def _epoch_rng(self, epoch_idx: int) -> random.Random:
        return random.Random(f"{self.seed}:epoch:{epoch_idx}")

    def _compute_schedule(self) -> None:
        self.steps_per_epoch = [
            len(self.build_epoch_macro_batches(e, self._epoch_rng(e))) for e in range(self.epochs)
        ]
        self.total_steps = sum(self.steps_per_epoch)
        if self.total_steps < 1:
            raise ValueError("Training schedule produced 0 total steps")

        if self._explicit_pct_start is not None:
            self.pct_start = float(self._explicit_pct_start)
        elif self.warmup_steps is not None:
            self.pct_start = min(0.1, self.warmup_steps / max(1, self.total_steps))
        else:
            self.pct_start = 0.1

        if self.lr_schedule == "wsd" and self.pct_start + self.decay_pct > 1.0:
            raise ValueError(
                f"pct_start ({self.pct_start:.4f}) + decay_pct ({self.decay_pct:.4f}) > 1.0 for wsd"
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

    def _autocast(self):
        if self.use_bf16 and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.lr_schedule == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.total_steps,
                pct_start=self.pct_start,
                div_factor=25.0,
                final_div_factor=self.final_div_factor,
            )

        warmup = int(round(self.pct_start * self.total_steps))
        last_step = max(1, self.total_steps - 1)
        init_ratio = 1.0 / 25.0
        min_ratio = self.min_lr_ratio

        if self.lr_schedule == "constant":
            def lr_lambda(step):
                if warmup > 0 and step < warmup:
                    return init_ratio + (1.0 - init_ratio) * (step / warmup)
                return 1.0
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if self.lr_schedule == "linear":
            def lr_lambda(step):
                if warmup > 0 and step < warmup:
                    return init_ratio + (1.0 - init_ratio) * (step / warmup)
                pct = (step - warmup) / max(1, last_step - warmup)
                return max(min_ratio, 1.0 - (1.0 - min_ratio) * pct)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if self.lr_schedule == "wsd":
            stable_end = int(round((1.0 - self.decay_pct) * self.total_steps))
            def lr_lambda(step):
                if warmup > 0 and step < warmup:
                    pct = step / warmup
                    return init_ratio + (1.0 - init_ratio) * (1.0 - math.cos(math.pi * pct)) / 2.0
                if step < stable_end:
                    return 1.0
                pct = min(1.0, (step - stable_end) / max(1, last_step - stable_end))
                return min_ratio + (1.0 - min_ratio) * (1.0 + math.cos(math.pi * pct)) / 2.0
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        raise ValueError(f"Unsupported lr_schedule: {self.lr_schedule!r}")

    def describe(self) -> dict:
        return {
            "trainer": self.trainer_name,
            "init_from": self.init_from,
            "lora_rank": self.lora_rank,
            "n_train_records": len(self.train_records),
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "lr": self.lr,
            "lr_schedule": self.lr_schedule,
            "min_lr": self.effective_min_lr,
            "weight_decay": self.weight_decay,
            "pct_start": round(self.pct_start, 5),
            "weights_dtype": self.weights_dtype,
            "use_bf16": self.use_bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
        }

    def train_step(self, step_batch: StepBatch) -> tuple[float, float, float, bool]:
        """Run one gradient-accumulated optimizer step over `step_batch`."""
        if not step_batch.micro_batches:
            raise ValueError("step_batch.micro_batches must not be empty")
        if step_batch.total_questions < 1:
            raise ValueError("step_batch.total_questions must be >= 1")

        total_q = step_batch.total_questions
        step_ces: list[float] = []
        step_kls: list[float] = []

        for mb in step_batch.micro_batches:
            prepared = [self.model.prepare(v) for v in mb.views]
            with self._autocast():
                logits_batch = self.model.forward(
                    [enc for enc, _ in prepared],
                    shared_prefix=mb.shared_prefix,
                )

            losses = []
            for (_, targets), record_logits in zip(prepared, logits_batch, strict=True):
                for z, q in zip(record_logits, targets, strict=True):
                    t = q["target"].to(self.device)
                    logp = torch.log_softmax(z.float(), dim=-1)
                    ce = -(t * logp).sum()
                    losses.append(ce)
                    step_ces.append(ce.item())
                    nz = t > 0
                    kl_bits = (t[nz] * (t[nz].log() - logp[nz])).sum() / math.log(2)
                    step_kls.append(kl_bits.item())

            loss = torch.stack(losses).sum() / total_q
            loss.backward()
            del prepared, logits_batch, losses, loss

        if len(step_kls) != total_q:
            raise AssertionError(f"Expected {total_q} question losses, got {len(step_kls)}")

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.trainable_parameters(), self.max_grad_norm
        )
        grad_norm_val = float(grad_norm.item())
        skipped = not math.isfinite(grad_norm_val)
        if not skipped:
            self.optimizer.step()
            if self.step + 1 < self.total_steps:
                self.scheduler.step()
        else:
            print(f"Skipping step {self.step + 1}: non-finite grad_norm={grad_norm_val}")
        self.optimizer.zero_grad(set_to_none=True)

        mean_ce = sum(step_ces) / total_q
        mean_kl = sum(step_kls) / total_q
        return mean_ce, mean_kl, grad_norm_val, skipped

    def _run_eval(self, step: int, epoch: int) -> dict | None:
        if self.eval_fn is None:
            return None
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self._autocast():
                result = self.eval_fn(self.model, step, epoch)
        finally:
            if was_training:
                self.model.train()
        return result

    def cleanup(self, *, free_model: bool = False) -> None:
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
        free_gpu()

    def train(self) -> dict:
        """Execute full training across all epochs and return structured step/eval history."""
        if self.model is None:
            raise ValueError("self.model is None")

        release_cuda_graphs(self.model)
        free_gpu()

        step_rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = self._build_scheduler(self.optimizer)
        self.step = 0
        self.best_metric_val = float("inf")

        history = {
            "config": self.describe(),
            "steps": [],
            "train_ce": [],
            "train_kl": [],
            "lr": [],
            "grad_norm": [],
            "skipped": [],
            "evals": [],
        }

        try:
            if self.eval_fn is not None and self.eval_at_start:
                ev0 = self._run_eval(step=0, epoch=0)
                history["evals"].append({"step": 0, "epoch": 0, "metrics": ev0})
                if self.best_metric is not None and ev0 is not None:
                    self.best_metric_val = float(ev0[self.best_metric])

            if self.gradient_checkpointing:
                self.model.enable_checkpointing()
            self.model.train()

            pbar = tqdm(total=self.total_steps, desc=self.trainer_name, disable=not self.show_progress)
            try:
                for epoch_idx in range(self.epochs):
                    macro_batches = self.build_epoch_macro_batches(epoch_idx, self._epoch_rng(epoch_idx))
                    for mb_idx, macro_batch in enumerate(macro_batches):
                        lr_used = float(self.scheduler.get_last_lr()[0])
                        step_batch = self.materialize_step(macro_batch, step_rng)
                        mean_ce, mean_kl, grad_norm, skipped = self.train_step(step_batch)
                        self.step += 1

                        history["steps"].append(self.step)
                        history["train_ce"].append(mean_ce)
                        history["train_kl"].append(mean_kl)
                        history["lr"].append(lr_used)
                        history["grad_norm"].append(grad_norm)
                        history["skipped"].append(skipped)

                        pbar.update(1)
                        pbar.set_postfix(
                            ep=f"{epoch_idx + 1}/{self.epochs}",
                            kl=f"{mean_kl:.4f}",
                            ce=f"{mean_ce:.3f}",
                            lr=f"{lr_used:.1e}",
                        )

                        is_final = (epoch_idx == self.epochs - 1) and (mb_idx == len(macro_batches) - 1)
                        should_eval = self.eval_fn is not None and (
                            is_final or (self.eval_every is not None and self.step % self.eval_every == 0)
                        )
                        eval_metrics = None
                        if should_eval:
                            eval_metrics = self._run_eval(step=self.step, epoch=epoch_idx + 1)
                            history["evals"].append(
                                {"step": self.step, "epoch": epoch_idx + 1, "metrics": eval_metrics}
                            )

                        if self.save_dir is not None:
                            extra = {
                                "step": self.step,
                                "epoch": epoch_idx + 1,
                                "train_ce": mean_ce,
                                "train_kl": mean_kl,
                                "eval_metrics": eval_metrics,
                                "config": history["config"],
                            }
                            if (
                                self.best_metric is not None
                                and eval_metrics is not None
                                and float(eval_metrics[self.best_metric]) < self.best_metric_val
                            ):
                                self.best_metric_val = float(eval_metrics[self.best_metric])
                                save_checkpoint(self.model, self.save_dir, "best", extra=extra)
                            if is_final or (self.save_every is not None and self.step % self.save_every == 0):
                                save_checkpoint(self.model, self.save_dir, "last", extra=extra)
            finally:
                pbar.close()
        finally:
            self.cleanup(free_model=self.free_model_on_exit)

        return history
