"""Fine-tune Kev to predict how people with a given persona answer survey questions.

Steps (see data/README.md and data/issp2020/DECISIONS.md):
  1. Training data: the ISSP years named with --train (default: all), as respondent-level data (one row per respondent
     with every attribute and answer, data/issp<year>/issp<year>_respondents.parquet and _questions.json). Records are
     drawn on the fly by issp_sampler.IsspSampler: a random country and question (one question per record), a random
     subset of attributes (any number, from none to all), a random value combination, and as target the answer shares
     of the respondents who match (PLAN-respondent-sampling.md). Calibration set: --calibration_frac of each training
     year's (country, question) slots, never trained on (the same slots in every run); per held-out slot, the whole
     country and every one-attribute group (e.g. Sex: Female), with the answer shares of all matching respondents
     (IsspSampler.calibration_records). Test set: SubPOP's evaluation split (TEST_DATASET), fixed whatever the training
     data, without questions that the ISSP data also asks (data/subpop/README.md).
  2. Load the released Kev model (backbone + LoRA adapter + pointer head); only the adapter and head are trained.
  3. Train: an epoch is one pass through every (year, country, question) slot in random order, each drawn as a record
     with a fresh persona and shown as an augmented view (issp_augment.augment:
     persona presentation, reversed scales, shuffled options); the loss is soft cross-entropy between the model's
     answer distribution and the observed answer shares.
  4. Step 0 (the released model before any training) is the baseline: score the calibration and test sets. Then
     validate on the calibration set every --validate_every steps; keep the weights with the lowest KL divergence.
  5. Evaluate the kept weights on the test set, next to the step-0 baseline.
  6. Saving: --out is a base path (runs/issp-0.8b); each run gets the next free version, <out>-v1, <out>-v2, ...
     Before training, <run>/run_info.json records what the run is (command, settings, training data, set sizes,
     model, GPU, what the drawn records look like); its end time and best step are added when the run finishes. Whenever a validation finds new best
     weights, they are written at once to <run>/best/ in Kev's checkpoint format (adapter + head.pt + tokenizer), and
     <run>/training_log.json is rewritten after every validation; <run>/metrics.json holds the final test results.

No temperature anywhere: we compare the model's raw distribution (softmax of its scores) with the observed shares,
so rescaling it would only hide how well it fits. Calibration and evaluation use the canonical (non-augmented) view.

Run:  python train_issp.py --out runs/issp-0.8b      (all ISSP years; -> runs/issp-0.8b-v1, -v2, ...)
"""
import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Triton compiles small C launchers with $CC; use gcc with warnings off (tools/cc_quiet) to hide a harmless header warning
os.environ.setdefault("CC", str(Path(__file__).resolve().parent / "tools" / "cc_quiet"))

import torch
import torch.nn.functional as F

sys.path.insert(0, "kev")
from kev.api import SystemOneRequest, to_record          # noqa: E402  request -> Kev's record + option keys
from kev.checkpoint import Checkpoint, write_meta        # noqa: E402  load the released model, save ours
from kev.model import DecisionModel, load_tokenizer      # noqa: E402  the model itself

from issp_augment import augment, canonical              # noqa: E402  our training views
from issp_sampler import IsspSampler, dataset_files      # noqa: E402  training records drawn from respondents
from mlflow_sync import sync_run                         # noqa: E402  sync run logs/metrics to local MLflow SQLite

# Training data: data/issp<year>/issp<year>_respondents.parquet (+ _questions.json); issp<year> is what --train takes.
# The calibration set is held out from it (--calibration_frac); the test set is a fixed dataset of its own,
# data/subpop/<name>.jsonl, never trained on.
DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_DATASET = "subpop_eval"           # SubPOP evaluation split (US General Social Survey)
ISSP_DATASETS = sorted(path.name.removesuffix("_respondents.parquet")
                       for path in DATA_DIR.glob("*/*_respondents.parquet"))
LOG_EVERY = 10           # optimizer steps between training-loss printouts
RULE = "-" * 80         # frames validation results so they stand out from the training progress lines
MAX_GRAD_NORM = 1.0     # gradients are clipped to this norm, as in Kev's own trainer


# ---------------------------------------------------------------- data

def read_records(name):
    """All records of the dataset data/*/<name>.jsonl."""
    (path,) = DATA_DIR.glob(f"*/{name}.jsonl")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def load_sets(sampler):
    """{set name: [records]}: "calibration" = the fixed records of the sampler's held-out ISSP slots, "test" = the
    fixed TEST_DATASET. Note: calibration stays group-level even with --single_respondent; we may later add an option
    to calibrate on single responses with full personas."""
    return {"calibration": sampler.calibration_records(), "test": read_records(TEST_DATASET)}


def to_model_input(view):
    """A view as Kev's record, plus per question: id, type, and the target as probabilities in Kev's option order."""
    request = SystemOneRequest.model_validate({"state": view["state"], "questions": {
        v: {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}
        for v, q in view["questions"].items()}})
    record, option_keys = to_record(request)
    questions = []
    for (v, q), meta in zip(view["questions"].items(), option_keys):
        target = torch.tensor([q["target"].get(key, 0.0) for key in meta["keys"]])
        questions.append({"v": v, "src": q["src"], "type": q["type"],
                          "target": target / target.sum()})   # shares were rounded: renormalise
    return record, questions


# ---------------------------------------------------------------- run folder and run info

def next_run_folder(base):
    """Create and return the next free version of a base path: runs/issp-0.8b -> runs/issp-0.8b-v1, -v2, ..."""
    base = Path(base)
    taken = [int(path.name.removeprefix(base.name + "-v")) for path in base.parent.glob(base.name + "-v*")
             if path.name.removeprefix(base.name + "-v").isdigit()]
    folder = base.parent / f"{base.name}-v{max(taken, default=0) + 1}"
    folder.mkdir(parents=True)             # fails rather than overwrite, e.g. if two runs start at the same moment
    return folder


def sha256(path):
    """Fingerprint of a file's contents: tells later whether a dataset was rebuilt after this run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_info(run, args, sets, sampler, drawn):
    """What this run is: written to <run>/run_info.json before training starts."""
    datasets = {}
    for name in sorted(args.train):
        for path in dataset_files(name):
            datasets[path.name] = {"file": str(path.relative_to(DATA_DIR.parent)), "sha256": sha256(path)}
    (path,) = DATA_DIR.glob(f"*/{TEST_DATASET}.jsonl")
    datasets[TEST_DATASET] = {"file": str(path.relative_to(DATA_DIR.parent)), "sha256": sha256(path)}
    return {"run": run.name,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "settings": vars(args),
            "train_datasets": args.train,
            "calibration_slots": len(sampler.held_out),
            "test_dataset": TEST_DATASET,
            "datasets": datasets,
            "sets": {"train": {"respondents": sampler.respondents, "country_years": len(sampler.units),
                               "slots": len(sampler.slots),
                               "drawn_records": drawn},
                     **{name: {"records": len(records), "targets": sum(len(r["questions"]) for r in records)}
                        for name, records in sets.items()}},
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__}


# ---------------------------------------------------------------- model

def load_model(base, device):
    """The released Kev model ready to train: base backbone with its LoRA adapter and pointer head loaded."""
    checkpoint = Checkpoint(base)
    meta = checkpoint.meta
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    model = DecisionModel(meta.base, tok, device, lora=meta.lora, revision=meta.base_revision, head_dim=meta.head_dim,
                          option_isolation=meta.option_isolation, special_embeddings=meta.special_embeddings,
                          lora_targets=meta.extra.get("args", {}).get("lora_targets", "all"))
    model.lm.config.use_cache = False                    # no generation: the cache only costs memory
    checkpoint.warm_start(model, meta)                   # loads adapter + head; refuses a mismatched architecture
    return tok, model, meta


def soft_cross_entropy(logits, target):
    """Cross-entropy of the model's distribution (softmax of logits) against the observed answer shares."""
    return -(target * F.log_softmax(logits, dim=-1)).sum()


def trainable_state(model):
    """A CPU copy of the trained weights (adapter + head), to restore the best epoch later."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(model, tok, released_meta, folder, extra):
    """Write the current weights in Kev's checkpoint format (adapter files + head.pt + tokenizer), loadable by
    kev.checkpoint / kev.serve. Temperature 1.0: Kev's server divides scores by it, so it serves the raw distribution."""
    folder.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(folder)
    head = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
    write_meta(folder, dataclasses.replace(released_meta, head=head, temperature=1.0, extra=extra))
    tok.save_pretrained(folder)


def restore(model, state):
    params = dict(model.named_parameters())
    for name, value in state.items():
        params[name].data.copy_(value)


# ---------------------------------------------------------------- training

GROUP_SIZES = ("1", "2-9", "10-99", "100+")   # respondents behind a target, as in IsspSampler.statistics


def group_size(n):
    """The GROUP_SIZES bucket of a target from n respondents."""
    return "1" if n == 1 else "2-9" if n < 10 else "10-99" if n < 100 else "100+"


def entropy(target):
    """Entropy of the observed shares in nats: the lowest cross-entropy any prediction can reach on this question."""
    return -(target * target.clamp_min(1e-12).log()).sum()


def train_epoch(model, tok, sampler, optimizer, scheduler, rng, args, autocast, device, step_log, epoch, total_steps,
                validate, write_training_log):
    """One epoch: every slot of `sampler` once, in random order, each drawn as a record with a fresh persona and shown
    as an augmented view, `args.batch` records per step.

    Calls validate(step) every `args.validate_every` steps and after the last step of training.

    Every LOG_EVERY steps it prints averages since the last printout: training KL divergence (loss minus the
    targets' own entropy, in bits, comparable to the calibration numbers), also per group size (GROUP_SIZES: how many
    respondents a target's shares come from; one-respondent targets are mostly noise, large groups show whether the
    model fits the real signal), training loss, gradient norm before
    clipping, learning rate, and GPU memory: the peak since the last printout (it depends on the longest batch in that
    window) and what is allocated right after the step (a leak would make this grow)."""
    model.train()
    window = defaultdict(list)
    slots = rng.sample(sampler.slots, len(sampler.slots))
    for start in range(0, len(slots), args.batch):
        drawn = [sampler.record(rng, slot)[0] for slot in slots[start:start + args.batch]]
        inputs = [to_model_input(augment(r, rng, args.p_reverse)) for r in drawn]
        with autocast:
            logits = model.forward_batch([model.encode(tok, record) for record, _ in inputs])
        losses, entropies, sizes = [], [], []
        for r, (_, questions), record_logits in zip(drawn, inputs, logits):
            for z, q in zip(record_logits, questions):
                target = q["target"].to(device)
                losses.append(soft_cross_entropy(z.float(), target))
                entropies.append(entropy(target))
                sizes.append(group_size(r["n"][q["v"]]))
        loss = torch.stack(losses).mean()                # every question target counts equally
        loss.backward()
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
            window["kl_bits " + size].append(kl)          # per target, so each group size is averaged over its targets

        if step % LOG_EVERY == 0:
            mean = lambda key: sum(window[key]) / len(window[key])
            by_n = " ".join(f"{s}:{mean('kl_bits ' + s):.2f}" for s in GROUP_SIZES if window["kl_bits " + s])
            peak, held = torch.cuda.max_memory_allocated() / 1e9, torch.cuda.memory_allocated() / 1e9
            print(f"  ep {epoch} {step}/{total_steps} ({100 * step / total_steps:.0f}%) | "
                  f"KL {mean('kl_bits'):.3f} [{by_n}] | loss {mean('loss'):.3f} | "
                  f"grad {mean('grad_norm'):.2f} | lr {entry['lr']:.1e} | gpu {peak:.1f}/{held:.1f}G", flush=True)
            window = defaultdict(list)
            torch.cuda.reset_peak_memory_stats()
            if not (step % args.validate_every == 0 or step == total_steps):
                write_training_log()

        if step % args.validate_every == 0 or step == total_steps:
            validate(step)
            model.train()                                # validate() switched the model to eval mode


@torch.no_grad()
def predict(model, tok, records, batch, autocast):
    """Logits for every question of every record, canonical view: [{v, type, logits, target}] on the CPU."""
    model.eval()
    rows = []
    for start in range(0, len(records), batch):
        inputs = [to_model_input(canonical(r)) for r in records[start:start + batch]]
        with autocast:
            logits = model.forward_batch([model.encode(tok, record) for record, _ in inputs])
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
    parser.add_argument("--base", default="jaredpalmer/kev-0.8b", help="released Kev model to start from")
    parser.add_argument("--train", nargs="+", choices=ISSP_DATASETS, default=ISSP_DATASETS,
                        help="ISSP years to draw training records from (default: all); the calibration set is held "
                             "out of these years (--calibration_frac); the test set is always the fixed SubPOP set")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4, help="peak learning rate")
    parser.add_argument("--batch", type=int, default=64, help="records per optimizer step")
    parser.add_argument("--calibration_frac", type=float, default=0.01,
                        help="share of each training year's (country, question) slots held out as the calibration set "
                             "(never trained on; the same slots in every run)")
    parser.add_argument("--single_respondent", action="store_true",
                        help="train on single respondents with their full non-missing persona (one-hot target per "
                             "record) instead of sampling an attribute subset and averaging matching respondents; "
                             "calibration stays unchanged")
    parser.add_argument("--eval_batch", type=int, default=256,
                        help="records per forward pass when scoring the calibration and test sets (no gradients, so it "
                             "can be much larger; Kev splits each pass into chunks of at most 16,384 tokens)")
    parser.add_argument("--p_reverse", type=float, default=0.5, help="chance a score scale is shown high -> low")
    parser.add_argument("--validate_every", type=int, default=100, help="optimizer steps between calibration-set validations")
    parser.add_argument("--checkpointing", action="store_true",
                        help="gradient checkpointing: recompute activations in the backward pass instead of storing them; "
                             "much less GPU memory, ~20-30%% slower, same results (needed for larger models)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True,
                        help="base path of the run folder, e.g. runs/issp-0.8b; the run goes to the next free version, "
                             "runs/issp-0.8b-v1, -v2, ...")
    args = parser.parse_args()

    out = next_run_folder(args.out)
    device = "cuda"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)   # bf16 compute, fp32 weights

    sampler = IsspSampler(args.train, held_out=args.calibration_frac, single_respondent=args.single_respondent)
    sets = load_sets(sampler)
    drawn = sampler.statistics(random.Random(args.seed + 1))   # its own generator: training draws are unaffected
    by_n = " ".join(f"{k}:{v:.2f}" for k, v in drawn["targets_by_respondents"].items() if v)
    by_k = " ".join(f"{k}:{v:.2f}" for k, v in drawn["attributes_shown"].items() if v)
    print(f"run: {out} | train: {sampler.respondents} resp, {len(sampler.units)} country-yrs, {len(sampler.slots)} slots/ep | "
          f"cal: {len(sampler.held_out)} slots ({sum(len(r['questions']) for r in sets['calibration'])} targets) | "
          f"test: {sum(len(r['questions']) for r in sets['test'])} targets")
    print(f"draws (n={drawn['records_drawn']}): resp [{by_n}] | attrs [{by_k}]")
    info = run_info(out, args, sets, sampler, drawn)
    (out / "run_info.json").write_text(json.dumps(info, indent=1))

    tok, model, released_meta = load_model(args.base, device)
    if args.checkpointing:
        # the same setting as Kev's own trainer (kev/kev/train.py --checkpointing)
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"trainable params: {sum(p.numel() for p in model.trainable_parameters()) / 1e6:.1f}M")


    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * math.ceil(len(sampler.slots) / args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    step_log, validation_log = [], []
    best = {"kl": math.inf, "step": None, "state": None}

    def write_training_log():
        (out / "training_log.json").write_text(json.dumps({"steps": step_log, "validations": validation_log,
                                                            "best_step": best["step"], "args": vars(args)}, indent=1))
        sync_run(out)

    def validate(step):
        """Score the calibration set; when the KL divergence is the lowest so far, keep these weights in memory and
        save them to <out>/best/. Step 0 is the released model before training: the baseline."""
        started = time.time()
        calibration = metrics(with_probs(predict(model, tok, sets["calibration"], args.eval_batch, autocast)))
        is_best = calibration["all"]["kl_divergence"] < best["kl"]
        if is_best:
            best.update(kl=calibration["all"]["kl_divergence"], step=step, state=trainable_state(model))
            save_checkpoint(model, tok, released_meta, out / "best",
                            extra={"init_from": args.base, "issp_args": vars(args), "step": step,
                                   "calibration_kl_divergence": best["kl"]})
        validation_log.append({"step": step, "calibration": calibration, "seconds": time.time() - started})
        write_training_log()
        verdict = "*BEST*" if is_best else f"best={best['step']}"
        print(f"{RULE}\nval step {step} ({time.time() - started:.0f}s) [{verdict}] cal: {describe(calibration)}", flush=True)
        if step > 0:
            print(RULE, flush=True)              # step 0 adds its test-set lines before closing the frame

    # step 0, the baseline: the released model before any training, on the calibration and test sets
    validate(0)
    test_sets = [name for name in sets if name.startswith("test")]
    released = {name: metrics(with_probs(predict(model, tok, sets[name], args.eval_batch, autocast))) for name in test_sets}
    for name, result in released.items():
        print(f"  {name}: {describe(result)}", flush=True)
    print(RULE, flush=True)

    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs} ({total_steps // args.epochs} steps)", flush=True)
        epoch_start = time.time()
        train_epoch(model, tok, sampler, optimizer, scheduler, rng, args, autocast, device, step_log, epoch,
                    total_steps, validate, write_training_log)
        print(f"epoch {epoch} done ({(time.time() - epoch_start) / 60:.1f}m)", flush=True)

    restore(model, best["state"])
    results = {}
    for name in test_sets:
        after = predict(model, tok, sets[name], args.eval_batch, autocast)
        results[name] = {"released (step 0)": released[name],
                         f"best (step {best['step']})": metrics(with_probs(after))}
    print_table(results)

    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    info.update(finished=datetime.datetime.now().isoformat(timespec="seconds"), total_steps=total_steps,
                best_step=best["step"], best_calibration_kl_divergence=best["kl"])
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    write_training_log()
    print(f"saved: {out / 'best'} (step {best['step']}) | {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
