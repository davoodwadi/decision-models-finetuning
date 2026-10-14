"""Print random training samples as the model gets them: x (the exact text Kev reads) and y (the target).

ISSP (default): each call draws fresh samples for one question with the training sampler (issp_sampler.IsspSampler,
from the year's respondent data): a random country where the question was asked; without --group a random number of
random attributes (uniform from 0 to all), with --group the variables given; a random value combination among those
that occur; as target the answer shares of the respondents who match (1 respondent gives a one-hot target).

SubPOP (--dataset subpop_eval or subpop100): one question of that set (--question, or a random one), for every
subpopulation that has it, as stored, side by side per attribute (Male then Female, Democrat then Republican, ...),
each attribute closed by a table comparing the answer shares of its values. --attribute shows only the attributes
named (e.g. SEX RACE); --samples does not apply.

Every sample then goes through the training view (issp_augment.augment, exactly as train_issp.train_epoch: persona
presentation, reversed scales, shuffled options), or, with --canonical, the fixed view calibration and test use; then
Kev's input format (train_issp.to_model_input) and encoder (kev.model.encode, with the model's tokenizer).
  x = the encoded token ids decoded back to text: special tokens included, the state cut at Kev's MAX_STATE tokens
      as in training. For reading, each special token except <|box_end|> starts a new line (display only: Kev reads
      no such line breaks).
  y = the target the training loss fits (soft cross-entropy): the answer shares, in the order the options are
      shown. Kev's own `label` field is not used in training (it is always 0 here) and is not shown.
Special tokens: <|fim_prefix|> starts the state, <|fim_middle|> a question, <|box_start|> / <|box_end|> enclose each
option, <|fim_suffix|> is where the model decides.

Needs the tokenizer in the Hugging Face cache (no model weights are loaded). Run from the project root:
  kev/.venv/bin/python show_train_samples.py --year 2022 --question v45                   (random attributes)
  kev/.venv/bin/python show_train_samples.py --year 2022 --question v45 --group SEX
  kev/.venv/bin/python show_train_samples.py --year 2023 --question v31 --group RELIGGRP PRTY
  kev/.venv/bin/python show_train_samples.py --dataset subpop_eval --question natrace
  kev/.venv/bin/python show_train_samples.py --dataset subpop_eval --canonical             (the view the test uses)
  kev/.venv/bin/python show_train_samples.py --dataset subpop_eval --canonical --attribute SEX POLPARTY
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data"))
from issp_common import PERSONA_KEY                    # noqa: E402  ISSP variable -> persona line
from issp_sampler import IsspSampler                   # noqa: E402  the training sampler
import train_issp                                      # noqa: E402  Kev's input format and the datasets, as in training
from issp_augment import augment, canonical            # noqa: E402  the training view / the calibration-and-test view
from kev.checkpoint import Checkpoint                  # noqa: E402
from kev.model import MAX_STATE, encode, load_tokenizer  # noqa: E402

RULE = "-" * 100                                       # separates samples
# SubPOP attributes in the order shown, each with its values in comparison order (values not listed follow, sorted)
SUBPOP_ORDER = {"SEX": ["Male", "Female"], "POLPARTY": ["Democrat", "Republican"],
                "POLIDEOLOGY": ["Liberal", "Moderate", "Conservative"], "RACE": ["White", "Black", "Hispanic", "Asian"],
                "RELIG": ["Protestant", "Jewish", "Muslim", "Hindu", "Atheist"],
                "EDUCATION": ["Less than high school", "College graduate/some postgrad"],
                "INCOME": ["Less than $30,000", "$100,000 or more"], "CREGION": ["Northeast", "South"]}

def issp_samples(args, rng):
    """(title, record) per ISSP sample, drawn by issp_sampler.IsspSampler as in training, for one question."""
    sampler = IsspSampler([f"issp{args.year}"], single_respondent=args.single_respondent)
    question = args.question or rng.choice(sorted({v for u in sampler.units for v in u.questions}))
    keys = [PERSONA_KEY[name] for name in args.group] if args.group else None
    slots = []
    for i, unit in enumerate(sampler.units):
        if question not in unit.questions:
            continue
        pool = sampler.pool(unit, [question])
        columns = [unit.attribute_names.index(k) for k in keys] if keys else []
        if not keys or (unit.attrs[np.ix_(pool, columns)] >= 0).all(axis=1).any():
            slots.append((i, question))
    if not slots:
        sys.exit(f"no country has answers to {question} in ISSP {args.year}"
                 + (f" together with {', '.join(args.group)}" if args.group else ""))
    qtype = sampler.units[slots[0][0]].questions[question]["type"]
    print(f"ISSP {args.year} {question} ({qtype}), grouped by "
          f"{', '.join(args.group) if args.group else 'a random subset of attributes per sample'}: {len(slots)} countries\n")
    for _ in range(args.samples):
        record, info = sampler.record(rng, rng.choice(slots), keys)
        shown = list(record["persona"].values())[1:1 + info["attributes"]]   # after Country, before filter lines
        yield (f"{info['country']}, {info['attributes']} attributes: {', '.join(shown) or '(country only)'}  "
               f"(n = {info['n'][question]} respondents; {info['combinations']} value combinations)", record)


def subpop_samples(args, rng):
    """[(attribute, [(title, record)])]: per attribute (SUBPOP_ORDER), its subpopulations that have the question in
    comparison order, records as stored with only that question."""
    records = train_issp.read_records(args.dataset)
    questions = sorted({v for r in records for v in r["questions"]})
    question = args.question or rng.choice(questions)
    having = [r for r in records if question in r["questions"]]
    if not having:
        sys.exit(f"{question} is not in {args.dataset}")
    print(f"{args.dataset} {question} ({having[0]['questions'][question]['type']}): {len(having)} subpopulations\n")
    by_attribute = {}
    for r in having:
        _, attribute, value = r["group"].split("/", 2)
        by_attribute.setdefault(attribute, {})[value] = {**r, "questions": {question: r["questions"][question]}}
    order = list(SUBPOP_ORDER) + sorted(set(by_attribute) - set(SUBPOP_ORDER))
    groups = []
    for attribute in order:
        if attribute not in by_attribute or (args.attribute and attribute not in args.attribute):
            continue
        values = by_attribute[attribute]
        known = SUBPOP_ORDER.get(attribute, [])
        ordered = [v for v in known if v in values] + sorted(set(values) - set(known))
        groups.append((attribute, [(f"{attribute} = {v}", values[v]) for v in ordered]))
    return groups


def compare(attribute, samples):
    """A table of the answer shares of each value of one attribute, options in canonical order."""
    rows = []
    for title, record in samples:
        kev_record, questions = train_issp.to_model_input(canonical(record))
        rows.append((title.split(" = ", 1)[1], kev_record["questions"][0]["options"], questions[0]["target"].tolist()))
    options = rows[0][1]
    width = max(len(value) for value, _, _ in rows)
    print(f"=== comparison by {attribute}")
    print(f"  {'':{width}}  " + " | ".join(options))
    for value, _, shares in rows:
        print(f"  {value:{width}}  " + " | ".join(f"{share:>{len(option)}.1%}" for option, share in zip(options, shares)))
    print()


def show(i, total, title, record, rng, tok, option_isolation):
    """Print x and y for one record."""
    view = canonical(record) if rng is None else augment(record, rng)   # the same call as train_issp.train_epoch
    kev_record, questions = train_issp.to_model_input(view)
    enc = encode(tok, kev_record, option_isolation=option_isolation)
    question = kev_record["questions"][0]
    print(f"{RULE}\n=== sample {i}/{total}: {title}")
    cut = f", CUT to {MAX_STATE}" if enc["state_truncated"] else ""
    print(f"x ({len(enc['ids'])} tokens; state {enc['state_tokens']} of at most {MAX_STATE}{cut}):")
    text = tok.decode(enc["ids"])
    for token in ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"):   # state, question, decide: own lines
        text = text.replace(token, f"\n{token}\n")
    text = text.replace("<|box_start|>", "\n<|box_start|>")                 # one option per line
    print("  " + text.strip("\n").replace("\n\n", "\n").replace("\n", "\n  "))
    shares = questions[0]["target"].tolist()
    print(f"y ({questions[0]['type']}; answer shares, * = most common):")
    for option, share in zip(question["options"], shares):
        print(f"  {share:6.1%} {'*' if share == max(shares) else ' '} {option}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="issp", choices=["issp", "subpop_eval", "subpop100"],
                        help="issp: sampled from respondents (--year); subpop_eval / subpop100: the SubPOP sets")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--question", default=None, help="item id (default: v45 for ISSP 2022, otherwise random)")
    parser.add_argument("--group", nargs="+", default=None,
                        help="ISSP only. Fixed grouping variables (default: a random subset per sample, as in "
                             f"training): {', '.join(PERSONA_KEY)}")
    parser.add_argument("--single_respondent", action="store_true",
                        help="ISSP only: draw single respondents with their full persona (one-hot target)")
    parser.add_argument("--canonical", action="store_true",
                        help="show the fixed view calibration and test use, not the training view")
    parser.add_argument("--base", default="jaredpalmer/kev-0.8b", help="the model whose tokenizer encodes x")
    parser.add_argument("--samples", type=int, default=10, help="ISSP only: number of samples")
    parser.add_argument("--attribute", nargs="+", default=None,
                        help=f"SubPOP only: attributes to compare (default: all): {', '.join(SUBPOP_ORDER)}")
    parser.add_argument("--seed", type=int, default=None, help="fix to repeat the same draw")
    args = parser.parse_args()
    if args.dataset == "issp" and args.question is None and args.year == 2022:
        args.question = "v45"
    rng = random.Random(args.seed)

    meta = Checkpoint(args.base).meta                    # the tokenizer training uses (train_issp.load_model)
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    view_rng = None if args.canonical else rng
    if args.dataset == "issp":
        samples = list(issp_samples(args, rng))
        for i, (title, record) in enumerate(samples, start=1):
            show(i, len(samples), title, record, view_rng, tok, meta.option_isolation)
        return
    groups = subpop_samples(args, rng)
    total, i = sum(len(samples) for _, samples in groups), 0
    for attribute, samples in groups:
        for title, record in samples:
            i += 1
            show(i, total, title, record, view_rng, tok, meta.option_isolation)
        compare(attribute, samples)


if __name__ == "__main__":
    main()
