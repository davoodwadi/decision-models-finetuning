"""Fine-tune a model to predict how people with a given persona answer survey questions.

Three layers: the data (the recipe given with --data: training_data.py, constructions.py), the model (the model flags:
models/), and this trainer, which uses only their interfaces.

Steps (see data/README.md and data/issp2020/DECISIONS.md):
  1. Training data: the recipe given with --data (configs/data/*.json; training_data.py, PLAN-record-construction.md):
     a source (the ISSP years, as respondent-level data: one row per respondent with every attribute and answer) and a
     construction (how a training record is built from a (year, country, question) slot: group answer shares, a single
     respondent, a fixed subset, ...). Records are drawn on the fly. Calibration set: the source's held-out slots
     (calibration_frac of each year's slots, never trained on, the same in every run), per slot the whole country and
     every one-attribute group (e.g. Sex: Female) with the answer shares of all matching respondents
     (IsspSource.calibration_records), whatever the construction. Test set: SubPOP's evaluation split (TEST_DATASET),
     fixed whatever the training data, without questions that the ISSP data also asks (data/subpop/README.md).
  2. The model (models/, PLAN-model-layer.md). With --kev, Kev's pointer model: by default the released Kev (LoRA
     adapter + pointer head, both trained); with --lora_rank R a new untrained rank-R adapter on the plain base and the
     released head; with --full_finetune no adapter, the plain base backbone and the released head, every weight
     trained. Without --kev, an LLM (--base, default Qwen/Qwen3.5-0.8B) read through its own next-token head: a system
     prompt (--prompt) asks for the letter of one option, and the answer distribution is the softmax over the option
     letters' logits; every weight trained (models/llm_letters.py, PLAN-llm-letters.md).
  3. Train: an epoch is one pass through the construction's slots in random order, each drawn as a record and shown in
     the view --augment gives (issp_augment.view; by default every variation off: 'Key: value' persona lines in stored
     order, scales low -> high, options in questionnaire order; `--augment full` switches on every persona
     presentation, reversed scales and shuffled options, as runs before 2026-10-14 trained); the loss is soft
     cross-entropy between the model's answer distribution and the observed answer shares.
  4. Step 0 (the model before any training) is the baseline: score the calibration and test sets. Then validate on the
     calibration set every --validate_every steps; keep the weights with the lowest KL divergence.
  5. Evaluate the kept weights on the test set, next to the step-0 baseline.
  6. Saving: --out is a base path (runs/issp-0.8b); each run gets the next free version, <out>-v1, <out>-v2, ...
     Before training, <run>/run_info.json records what the run is (command, settings, data recipe, model, set sizes,
     GPU, what the drawn records look like); its end time and best step are added when the run finishes. Whenever a
     validation finds new best weights, they are written at once to <run>/best/ in the model's checkpoint format
     (models.load reads any of them); every validation also writes the current weights to <run>/last/ in the
     same format, so an interrupted run keeps its latest validated model. <run>/training_log.json is rewritten after every validation;
     <run>/metrics.json holds the final test results.

No temperature anywhere: we compare the model's raw distribution (softmax of its scores) with the observed shares,
so rescaling it would only hide how well it fits. Calibration and test are never augmented: always the canonical
view (every variation off), the same inputs every scoring.

Run:  python train.py --kev --data configs/data/issp-all-group-min10.json --out runs/issp-0.8b   (Kev)
      python train.py --data configs/data/issp-all-group-min10.json --lr 1e-5 --out runs/issp-llm   (the LLM)
"""
import argparse
import dataclasses
import datetime
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# Triton compiles small C launchers with $CC; use gcc with warnings off (tools/cc_quiet) to hide a harmless header warning
os.environ.setdefault("CC", str(Path(__file__).resolve().parent / "tools" / "cc_quiet"))
# single-threaded tokenizer: run issp-all-0.8b-r64-v2 aborted with heap corruption inside its multithreaded batch encoding
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F

import models                                            # noqa: E402  the model layer: flags -> model
from issp_augment import canonical, parse_augmentation, view   # noqa: E402  how a record is shown
from training_data import (                              # noqa: E402  the dataset layer: recipe -> training data
    GROUP_SIZES, TEST_DATASET, TrainingData, group_size, load_recipe, read_records)
from mlflow_sync import sync_run                         # noqa: E402  sync run logs/metrics to local MLflow SQLite

LOG_EVERY = 10           # optimizer steps between training-loss printouts
RULE = "-" * 80         # frames validation results so they stand out from the training progress lines
MAX_GRAD_NORM = 1.0     # gradients are clipped to this norm, as in Kev's own trainer


# ---------------------------------------------------------------- run folder and run info

def next_run_folder(base):
    """Create and return the next free version of a base path: runs/issp-0.8b -> runs/issp-0.8b-v1, -v2, ..."""
    base = Path(base)
    taken = [int(path.name.removeprefix(base.name + "-v")) for path in base.parent.glob(base.name + "-v*")
             if path.name.removeprefix(base.name + "-v").isdigit()]
    folder = base.parent / f"{base.name}-v{max(taken, default=0) + 1}"
    folder.mkdir(parents=True)             # fails rather than overwrite, e.g. if two runs start at the same moment
    return folder


def run_info(run, args, data, model, sets, drawn, augmentation):
    """What this run is: written to <run>/run_info.json before training starts. settings.recipe is the full recipe, so
    the run's training data can be rebuilt (training_data.recipe_from_settings); model is the model's describe();
    train_view is every variation the training view switches on or off (issp_augment.Augmentation)."""
    return {"run": run.name,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "settings": {**vars(args), "recipe": data.recipe},
            "model": model.describe(),
            "train_view": dataclasses.asdict(augmentation),
            "train_datasets": data.source.names,
            "calibration_slots": len(data.source.held_out),
            "test_dataset": TEST_DATASET,
            "datasets": data.file_hashes(),
            "sets": {"train": {"respondents": data.source.respondents, "country_years": len(data.source.units),
                               "slots": len(data.slots),
                               "drawn_records": drawn},
                     **{name: {"records": len(records), "targets": sum(len(r["questions"]) for r in records)}
                        for name, records in sets.items()}},
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__}


# ---------------------------------------------------------------- loss and best weights

def soft_cross_entropy(logits, target):
    """Cross-entropy of the model's distribution (softmax of logits) against the observed answer shares."""
    return -(target * F.log_softmax(logits, dim=-1)).sum()


def trainable_state(model):
    """A CPU copy of the trained weights (whatever the model trains: an adapter + head, or every weight), to restore the
    best epoch later."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore(model, state):
    params = dict(model.named_parameters())
    for name, value in state.items():
        params[name].data.copy_(value)


# ---------------------------------------------------------------- training

def entropy(target):
    """Entropy of the observed shares in nats: the lowest cross-entropy any prediction can reach on this question."""
    return -(target * target.clamp_min(1e-12).log()).sum()


def train_epoch(model, data, augmentation, optimizer, scheduler, rng, args, autocast, device, step_log, epoch,
                total_steps, validate, write_training_log):
    """One epoch: every slot of `data` once, in random order, each drawn as a record and shown as `augmentation` allows,
    `args.batch` records per step (forward and backward --micro_batch records at a time; one optimizer step).

    Calls validate(step) every `args.validate_every` steps and after the last step of training.

    Every LOG_EVERY steps it prints averages since the last printout: training KL divergence (loss minus the
    targets' own entropy, in bits, comparable to the calibration numbers), also per group size (GROUP_SIZES: how many
    respondents a target's shares come from; one-respondent targets are mostly noise, large groups show whether the
    model fits the real signal), training loss, gradient norm before
    clipping, learning rate, and GPU memory: the peak since the last printout (it depends on the longest batch in that
    window) and what is allocated right after the step (a leak would make this grow)."""
    model.train()
    window = defaultdict(list)
    slots = rng.sample(data.slots, len(data.slots))
    for start in range(0, len(slots), args.batch):
        drawn = [data.draw(rng, slot) for slot in slots[start:start + args.batch]]
        inputs = [model.prepare(view(r, augmentation, rng)) for r in drawn]
        questions_in_step = sum(len(questions) for _, questions in inputs)
        losses, entropies, sizes = [], [], []
        # forward + backward --micro_batch records at a time: the gradients add up to those of the whole batch's mean
        # loss (each question's loss / all the step's questions), so the update is the same as one pass, in less memory
        for part in range(0, len(inputs), args.micro_batch):
            with autocast:
                logits = model.forward([model_input for model_input, _ in inputs[part:part + args.micro_batch]])
            part_losses = []
            for r, (_, questions), record_logits in zip(drawn[part:], inputs[part:part + args.micro_batch], logits):
                for z, q in zip(record_logits, questions):
                    target = q["target"].to(device)
                    part_losses.append(soft_cross_entropy(z.float(), target))
                    entropies.append(entropy(target))
                    sizes.append(group_size(r.get("n", {}).get(q["v"])))
            (torch.stack(part_losses).sum() / questions_in_step).backward()
            losses += [l.detach() for l in part_losses]
        loss = torch.stack(losses).mean()                # every question target counts equally
        grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step = len(step_log) + 1
        entry = {"step": step, "epoch": epoch, "loss": loss.item(),
                 "kl_bits": (loss.item() - torch.stack(entropies).mean().item()) / math.log(2),
                 "grad_norm": grad_norm.item(), "lr": scheduler.get_last_lr()[0]}
        step_log.append(entry)
        for key in ("loss", "kl_bits", "grad_norm"):
            window[key].append(entry[key])
        target_kl = ((torch.stack(losses) - torch.stack(entropies)) / math.log(2)).tolist()
        for size, kl in zip(sizes, target_kl):
            if size is not None:
                window["kl_bits " + size].append(kl)      # per target, so each group size is averaged over its targets

        if step % LOG_EVERY == 0:
            mean = lambda key: sum(window[key]) / len(window[key])
            by_n = " ".join(f"{s}:{mean('kl_bits ' + s):.2f}" for s in GROUP_SIZES if window["kl_bits " + s])
            peak, held = torch.cuda.max_memory_allocated() / 1e9, torch.cuda.memory_allocated() / 1e9
            diagnostics = model.diagnostics()           # e.g. the LLM's probability on the valid letters
            entry.update(diagnostics)
            print(f"  ep {epoch} {step}/{total_steps} ({100 * step / total_steps:.0f}%) | "
                  f"KL {mean('kl_bits'):.3f} [{by_n}] | loss {mean('loss'):.3f} | "
                  f"grad {mean('grad_norm'):.2f} | lr {entry['lr']:.1e} | gpu {peak:.1f}/{held:.1f}G"
                  + "".join(f" | {key} {value:.3f}" for key, value in diagnostics.items()), flush=True)
            window = defaultdict(list)
            torch.cuda.reset_peak_memory_stats()
            if not (step % args.validate_every == 0 or step == total_steps):
                write_training_log()

        if step % args.validate_every == 0 or step == total_steps:
            validate(step)
            model.train()                                # validate() switched the model to eval mode


@torch.no_grad()
def predict(model, records, batch, autocast):
    """Logits for every question of every record, canonical view: [{v, type, logits, target}] on the CPU."""
    model.eval()
    rows = []
    for start in range(0, len(records), batch):
        inputs = [model.prepare(canonical(r)) for r in records[start:start + batch]]
        with autocast:
            logits = model.forward([model_input for model_input, _ in inputs])
        for (_, questions), record_logits in zip(inputs, logits):
            for z, q in zip(record_logits, questions):
                rows.append({**q, "logits": z.detach().float().cpu()})
    return rows


# ---------------------------------------------------------------- evaluation

def kl_bits(a, b):
    """KL(a || b) in bits; terms where a is 0 contribute nothing."""
    return float((a * (torch.log2(a.clamp_min(1e-12)) - torch.log2(b.clamp_min(1e-12)))).sum())


def kl_divergence(observed, predicted):
    """How far the prediction is from the observed shares, in bits: 0 = identical, however spread out the answers are.
    (Cross-entropy minus the observed shares' own entropy, so questions where people disagree are not penalised.)"""
    return kl_bits(observed, predicted)


def js_divergence(observed, predicted):
    """Jensen-Shannon divergence in bits: 0 = identical distributions, 1 = no overlap. Not logged in training; the
    SubPOP notebooks use it."""
    middle = (observed + predicted) / 2
    return (kl_bits(observed, middle) + kl_bits(predicted, middle)) / 2


def wasserstein(observed, predicted):
    """Earth mover's distance between two distributions over ordered levels, in levels: how far probability mass must
    move to turn one into the other (0 = identical; 1 = on average one level off). For `score` questions only."""
    return float((observed.cumsum(0) - predicted.cumsum(0)).abs().sum())


def metrics(rows):
    """Average error measures over rows that carry `probs` (prediction) and `target` (observed shares), per type."""
    by_type = {"all": rows, **{qtype: [r for r in rows if r["type"] == qtype] for qtype in ("choice", "score", "noul")}}
    result = {}
    for qtype, group in by_type.items():
        if not group:
            continue
        average = lambda measure: sum(measure(r["target"], r["probs"]) for r in group) / len(group)
        result[qtype] = {"questions": len(group), "kl_divergence": average(kl_divergence)}
        if qtype == "score":
            result[qtype]["wasserstein"] = average(wasserstein)
    return result


def with_probs(rows):
    """Add the model's raw answer distribution (softmax of its scores, no temperature) to each row."""
    return [{**r, "probs": F.softmax(r["logits"], dim=-1)} for r in rows]


def describe(by_type):
    """Metrics per question type on one line: KL divergence (bits), Wasserstein distance (levels, score only)."""
    return " | ".join(f"{t}: KL {m['kl_divergence']:.3f}"
                      + (f" W {m['wasserstein']:.2f}" if "wasserstein" in m else "")
                      for t, m in by_type.items())


def print_table(results):
    for set_name, by_model in results.items():
        print(f"\n{set_name}")
        for model_name, by_type in by_model.items():
            print(f"  {model_name:26} {describe(by_type)}")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    models.add_arguments(parser)
    parser.add_argument("--data", required=True,
                        help="recipe of the training data (configs/data/*.json): its source (which ISSP years, the "
                             "calibration share) and construction (how a record is built: group_shares, "
                             "single_respondent, fixed_subset); see training_data.py")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4, help="peak learning rate")
    parser.add_argument("--lr_schedule", choices=["onecycle", "constant"], default="onecycle",
                        help="onecycle: warm up over the first 5%% of steps from lr/25 to lr, then anneal to 0 by the "
                             "last step; constant: the same warm-up, then lr until the end")
    parser.add_argument("--batch", type=int, default=64, help="records per optimizer step")
    parser.add_argument("--micro_batch", type=int, default=None,
                        help="records per forward + backward pass (default: --batch). Smaller saves GPU memory; the "
                             "gradients of a step's passes add up to the same update as one pass over --batch records")
    parser.add_argument("--eval_batch", type=int, default=256,
                        help="records per forward pass when scoring the calibration and test sets (no gradients, so it "
                             "can be much larger; Kev splits each pass into chunks of at most 16,384 tokens)")
    parser.add_argument("--augment", nargs="*", default=[],
                        help="how training records are shown (calibration and test never augmented): none (default: "
                             "every variation off), full (all on), or switches such as layouts=lines,json key_names "
                             "value_wording punctuation shuffle_order p_reverse=0.5 shuffle_options "
                             "(issp_augment.parse_augmentation)")
    parser.add_argument("--validate_every", type=int, default=100, help="optimizer steps between calibration-set validations")
    parser.add_argument("--checkpointing", action="store_true",
                        help="gradient checkpointing: recompute activations in the backward pass instead of storing them; "
                             "much less GPU memory, ~20-30%% slower, same results (needed for larger models)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_eval", action="store_true",
                        help="quick runs: training loss only; no calibration or test scoring and no saved weights")
    parser.add_argument("--out", required=True,
                        help="base path of the run folder, e.g. runs/issp-0.8b; the run goes to the next free version, "
                             "runs/issp-0.8b-v1, -v2, ...")
    args = parser.parse_args()
    models.check_arguments(parser, args)
    args.micro_batch = min(args.micro_batch or args.batch, args.batch)
    try:
        augmentation = parse_augmentation(args.augment)
    except ValueError as error:
        parser.error(str(error))

    out = next_run_folder(args.out)
    device = "cuda"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)   # bf16 compute, fp32 weights

    data = TrainingData(load_recipe(args.data))
    sets = {} if args.no_eval else {"calibration": data.calibration_records(), "test": read_records(TEST_DATASET)}
    drawn = data.statistics(args.seed + 1)   # its own generator: training draws are unaffected
    by_n = " ".join(f"{k}:{v:.2f}" for k, v in drawn["targets_by_respondents"].items() if v)
    by_k = " ".join(f"{k}:{v:.2f}" for k, v in drawn["attributes_shown"].items() if v)
    print(f"run: {out} | data: {args.data} | train: {data.source.respondents} resp, {len(data.source.units)} country-yrs, "
          f"{len(data.slots)} slots/ep | "
          + ("no eval" if args.no_eval else
             f"cal: {len(data.source.held_out)} slots ({sum(len(r['questions']) for r in sets['calibration'])} targets) | "
             f"test: {sum(len(r['questions']) for r in sets['test'])} targets"))
    print(f"draws (n={drawn['records_drawn']}): resp [{by_n}] | attrs [{by_k}]")

    model = models.build(args, device)
    if args.train_last_layers is not None:
        model.train_last_layers(args.train_last_layers)   # before the optimizer: it takes only trainable weights
    if args.checkpointing:
        model.enable_checkpointing()
    print(f"model: {model.kind} {args.base} | trainable params: {sum(p.numel() for p in model.trainable_parameters()) / 1e6:.1f}M")
    info = run_info(out, args, data, model, sets, drawn, augmentation)
    (out / "run_info.json").write_text(json.dumps(info, indent=1))

    groups = model.parameter_groups(args.lr, args.lr_decay)   # first group: the full lr, the one the log records
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    total_steps = args.epochs * math.ceil(len(data.slots) / args.batch)
    if args.lr_schedule == "onecycle":   # each group's own peak; the same shape for all
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups],
                                                        total_steps=total_steps, pct_start=0.05)
    else:   # OneCycleLR's warm-up (cosine from lr/25 to lr over 5% of the steps), then constant
        warmup = 0.05 * total_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1 / 25 + (1 - 1 / 25) * (1 - math.cos(math.pi * min(step / warmup, 1))) / 2)

    step_log, validation_log = [], []
    best = {"kl": math.inf, "step": None, "state": None}

    def write_training_log():
        (out / "training_log.json").write_text(json.dumps({"steps": step_log, "validations": validation_log,
                                                            "best_step": best["step"], "args": vars(args)}, indent=1))
        sync_run(out)

    def validate(step):
        """Score the calibration set; when the KL divergence is the lowest so far, keep these weights in memory and
        save them to <out>/best/; every validation after step 0 also saves the current weights to <out>/last/, so an
        interrupted run keeps its latest validated model. Step 0 is the model before training: the baseline. With
        --no_eval: only the log."""
        if args.no_eval:
            write_training_log()
            return
        started = time.time()
        calibration = metrics(with_probs(predict(model, sets["calibration"], args.eval_batch, autocast)))
        diagnostics = model.diagnostics()               # on the calibration set
        is_best = calibration["all"]["kl_divergence"] < best["kl"]
        if is_best:
            best.update(kl=calibration["all"]["kl_divergence"], step=step, state=trainable_state(model))
            model.save(out / "best", extra={"init_from": args.base, "train_args": vars(args), "step": step,
                                            "calibration_kl_divergence": best["kl"]})
        if step > 0:   # written beside last/, then swapped in: an interruption mid-save leaves the previous last/
            model.save(out / "last-tmp", extra={"init_from": args.base, "train_args": vars(args), "step": step,
                                                "calibration_kl_divergence": calibration["all"]["kl_divergence"]})
            shutil.rmtree(out / "last", ignore_errors=True)
            (out / "last-tmp").rename(out / "last")
        validation_log.append({"step": step, "calibration": calibration, **diagnostics, "seconds": time.time() - started})
        write_training_log()
        verdict = "*BEST*" if is_best else f"best={best['step']}"
        print(f"{RULE}\nval step {step} ({time.time() - started:.0f}s) [{verdict}] cal: {describe(calibration)}"
              + "".join(f" | {key} {value:.3f}" for key, value in diagnostics.items()), flush=True)
        if step > 0:
            print(RULE, flush=True)              # step 0 adds its test-set lines before closing the frame

    # step 0, the baseline: the model before any training, on the calibration and test sets
    validate(0)
    test_sets = [name for name in sets if name.startswith("test")]
    released = {}
    for name in test_sets:
        released[name] = metrics(with_probs(predict(model, sets[name], args.eval_batch, autocast)))
        print(f"  {name}: {describe(released[name])}"
              + "".join(f" | {key} {value:.3f}" for key, value in model.diagnostics().items()), flush=True)
    if not args.no_eval:
        print(RULE, flush=True)

    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs} ({total_steps // args.epochs} steps)", flush=True)
        epoch_start = time.time()
        train_epoch(model, data, augmentation, optimizer, scheduler, rng, args, autocast, device, step_log, epoch,
                    total_steps, validate, write_training_log)
        print(f"epoch {epoch} done ({(time.time() - epoch_start) / 60:.1f}m)", flush=True)

    if args.no_eval:
        info.update(finished=datetime.datetime.now().isoformat(timespec="seconds"), total_steps=total_steps)
        (out / "run_info.json").write_text(json.dumps(info, indent=1))
        write_training_log()
        print(f"saved: {out / 'training_log.json'}")
        return
    restore(model, best["state"])
    results = {}
    for name in test_sets:
        after = predict(model, sets[name], args.eval_batch, autocast)
        results[name] = {"step 0": released[name],
                         f"best (step {best['step']})": metrics(with_probs(after))}
    print_table(results)

    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    info.update(finished=datetime.datetime.now().isoformat(timespec="seconds"), total_steps=total_steps,
                best_step=best["step"], best_calibration_kl_divergence=best["kl"])
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    write_training_log()
    print(f"saved: {out / 'best'} (step {best['step']}) | {out / 'last'} (step {validation_log[-1]['step']}) | {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
