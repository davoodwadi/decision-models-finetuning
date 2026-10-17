"""Print random training samples as the model gets them: x (the exact text the model reads) and y (the target).

ISSP (default): each call draws fresh samples for one question as training does (the year's respondent data,
issp_sampler.IsspSource, and a construction from constructions.py: group_shares, or single_respondent): a random country where the question was asked; without --group a random number of
random attributes (uniform from 0 to all), with --group the variables given; a random value combination among those
that occur among at least --min_n respondents; as target the answer shares of the respondents who match.

--data configs/data/<recipe>.json: samples drawn as the training of that recipe draws them (its source and
construction, training_data.TrainingData): a random training slot (any year, question and country the recipe has),
then the construction's record; --year, --question, --group, --min_n and --single_respondent do not apply.

SubPOP (--dataset subpop_eval or subpop100): one question of that set (--question, or a random one), for every
subpopulation that has it, as stored, side by side per attribute (Male then Female, Democrat then Republican, ...),
each attribute closed by a table comparing the answer shares of its values. --attribute shows only the attributes
named (e.g. SEX RACE); --samples does not apply.

Every sample then goes through the view --augment gives (issp_augment.view, exactly as train.py's flag of the same
name: by default every variation off, the view calibration and test use too; `--augment full` or switches such as
`layouts=json,prose key_names` for the variations to see); then the model's own input format (models.encoder: the same model flags as train.py).
  x = what the model reads. With --kev: Kev's encoded token ids decoded back to text, special tokens included, the
      state cut at Kev's MAX_STATE tokens as in training; for reading, each special token except <|box_end|> starts a
      new line (display only: Kev reads no such line breaks). <|fim_prefix|> starts the state, <|fim_middle|> a
      question, <|box_start|> / <|box_end|> enclose each option, <|fim_suffix|> is where the model decides.
      Without --kev: the LLM's prompt (chat template applied: system prompt, persona, question, lettered options),
      up to where it answers with a letter.
  y = the target the training loss fits (soft cross-entropy): the answer shares, in the order the options are
      shown (for the LLM: A, B, C, ...).

Needs the tokenizer in the Hugging Face cache (no model weights are loaded). Run from the project root:
  kev/.venv/bin/python show_train_samples.py --kev --year 2022 --question v45             (random attributes)
  kev/.venv/bin/python show_train_samples.py --year 2022 --question v45                   (the LLM's prompt)
  kev/.venv/bin/python show_train_samples.py --kev --year 2022 --question v45 --group SEX
  kev/.venv/bin/python show_train_samples.py --kev --year 2023 --question v31 --group RELIGGRP PRTY
  kev/.venv/bin/python show_train_samples.py --kev --data configs/data/issp-us-group-min100-attr2.json
  kev/.venv/bin/python show_train_samples.py --kev --dataset subpop_eval --question natrace
  kev/.venv/bin/python show_train_samples.py --kev --dataset subpop_eval --attribute SEX POLPARTY
  kev/.venv/bin/python show_train_samples.py --kev --year 2022 --question v45 --augment layouts=json,prose key_names
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data"))
from issp_common import PERSONA_KEY                    # noqa: E402  ISSP variable -> persona line
from issp_sampler import IsspSource                    # noqa: E402  the ISSP respondent data
from constructions import GroupShares, SingleRespondent  # noqa: E402  how training builds a record
import models                                          # noqa: E402  the model layer: each model's input format
from models.base import options_and_targets            # noqa: E402  options as text + targets, as every model reads
from training_data import TrainingData, load_recipe, read_records   # noqa: E402  recipes; the SubPOP sets
from issp_augment import canonical, parse_augmentation, view   # noqa: E402  how a record is shown

RULE = "-" * 100                                       # separates samples
# SubPOP attributes in the order shown, each with its values in comparison order (values not listed follow, sorted)
SUBPOP_ORDER = {"SEX": ["Male", "Female"], "POLPARTY": ["Democrat", "Republican"],
                "POLIDEOLOGY": ["Liberal", "Moderate", "Conservative"], "RACE": ["White", "Black", "Hispanic", "Asian"],
                "RELIG": ["Protestant", "Jewish", "Muslim", "Hindu", "Atheist"],
                "EDUCATION": ["Less than high school", "College graduate/some postgrad"],
                "INCOME": ["Less than $30,000", "$100,000 or more"], "CREGION": ["Northeast", "South"]}

def issp_samples(args, rng):
    """(title, record) per ISSP sample, drawn by a construction as in training, for one question."""
    sampler = IsspSource([f"issp{args.year}"])
    question = args.question or rng.choice(sorted({v for u in sampler.units for v in u.questions}))
    keys = [PERSONA_KEY[name] for name in args.group] if args.group else None
    slots = []
    for i, unit in enumerate(sampler.units):
        if question not in unit.questions:
            continue
        pool = sampler.pool(unit, [question])
        columns = [unit.attribute_names.index(k) for k in keys] if keys else []
        values = unit.attrs[np.ix_(pool, columns)]
        values = values[(values >= 0).all(axis=1)]
        if not keys and len(pool) >= args.min_n or keys and len(values) and \
                np.unique(values, axis=0, return_counts=True)[1].max() >= args.min_n:
            slots.append((i, question))
    if not slots:
        sys.exit(f"no country has {args.min_n} respondents answering {question} in ISSP {args.year}"
                 + (f" together with {', '.join(args.group)}" if args.group else ""))
    qtype = sampler.units[slots[0][0]].questions[question]["type"]
    print(f"ISSP {args.year} {question} ({qtype}), grouped by "
          f"{', '.join(args.group) if args.group else 'a random subset of attributes per sample'}: {len(slots)} countries\n")
    for _ in range(args.samples):
        slot = rng.choice(slots)
        if args.single_respondent and not keys:
            record, info = SingleRespondent().draw(sampler, rng, slot)
        else:
            record, info = GroupShares(args.min_n).draw(sampler, rng, slot, keys)
        shown = list(record["persona"].values())[1:1 + info["attributes"]]   # after Country, before filter lines
        yield (f"{info['country']}, {info['attributes']} attributes: {', '.join(shown) or '(country only)'}  "
               f"(n = {info['n'][question]} respondents; {info['combinations']} value combinations)", record)


def recipe_samples(args, rng):
    """(title, record) per sample of a recipe, drawn as train.train_epoch draws them: a random training slot, then the
    construction's record for it."""
    data = TrainingData(load_recipe(args.data))
    print(f"recipe {args.data}: {len(data.slots)} training slots\n")
    for _ in range(args.samples):
        slot = rng.choice(data.slots)
        record, info = data.construction.draw(data.source, rng, slot)
        unit, question = data.source.units[slot[0]], slot[1]
        shown = list(record["persona"].values())[1:1 + info["attributes"]]   # after Country, before filter lines
        yield (f"{unit.name} {question}, {info['country']}, {info['attributes']} attributes: "
               f"{', '.join(shown) or '(country only)'}  (n = {info['n'][question]} respondents)", record)


def subpop_samples(args, rng):
    """[(attribute, [(title, record)])]: per attribute (SUBPOP_ORDER), its subpopulations that have the question in
    comparison order, records as stored with only that question."""
    records = read_records(args.dataset)
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
        shown, questions = options_and_targets(canonical(record))
        rows.append((title.split(" = ", 1)[1], shown["questions"][0]["options"], questions[0]["target"].tolist()))
    options = rows[0][1]
    width = max(len(value) for value, _, _ in rows)
    print(f"=== comparison by {attribute}")
    print(f"  {'':{width}}  " + " | ".join(options))
    for value, _, shares in rows:
        print(f"  {value:{width}}  " + " | ".join(f"{share:>{len(option)}.1%}" for option, share in zip(options, shares)))
    print()


def show(i, total, title, record, augmentation, rng, encoder):
    """Print x and y for one record."""
    request = view(record, augmentation, rng)            # the same call as train.train_epoch
    shown, questions = options_and_targets(request)
    question = shown["questions"][0]
    print(f"{RULE}\n=== sample {i}/{total}: {title}")
    header, text = encoder.preview(request)
    print(f"x ({header}):")
    print("  " + text.replace("\n", "\n  "))
    shares = questions[0]["target"].tolist()
    print(f"y ({questions[0]['type']}; answer shares, * = most common):")
    for option, share in zip(question["options"], shares):
        print(f"  {share:6.1%} {'*' if share == max(shares) else ' '} {option}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="issp", choices=["issp", "subpop_eval", "subpop100"],
                        help="issp: sampled from respondents (--year); subpop_eval / subpop100: the SubPOP sets")
    parser.add_argument("--data", default=None,
                        help="a recipe (configs/data/*.json): samples drawn as that recipe's training draws them")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--question", default=None, help="item id (default: v45 for ISSP 2022, otherwise random)")
    parser.add_argument("--group", nargs="+", default=None,
                        help="ISSP only. Fixed grouping variables (default: a random subset per sample, as in "
                             f"training): {', '.join(PERSONA_KEY)}")
    parser.add_argument("--single_respondent", action="store_true",
                        help="ISSP only: draw single respondents with their full persona (one-hot target)")
    parser.add_argument("--min_n", type=int, default=10,
                        help="ISSP only: fewest respondents a drawn persona may match, as min_n in a group_shares recipe")
    parser.add_argument("--augment", nargs="*", default=[],
                        help="how records are shown, as train.py's --augment: none (default: every variation off), "
                             "full, or switches such as layouts=lines,json key_names p_reverse=0.5")
    parser.add_argument("--samples", type=int, default=10, help="ISSP only: number of samples")
    parser.add_argument("--attribute", nargs="+", default=None,
                        help=f"SubPOP only: attributes to compare (default: all): {', '.join(SUBPOP_ORDER)}")
    parser.add_argument("--seed", type=int, default=None, help="fix to repeat the same draw")
    models.add_arguments(parser)                         # --kev, --base, --prompt: the model whose input is shown
    args = parser.parse_args()
    models.check_arguments(parser, args)
    if args.dataset == "issp" and args.question is None and args.year == 2022:
        args.question = "v45"
    rng = random.Random(args.seed)

    encoder = models.encoder(args)                       # the model's input format, tokenizer only
    try:
        augmentation = parse_augmentation(args.augment)
    except ValueError as error:
        parser.error(str(error))
    if args.data or args.dataset == "issp":
        samples = list(recipe_samples(args, rng) if args.data else issp_samples(args, rng))
        for i, (title, record) in enumerate(samples, start=1):
            show(i, len(samples), title, record, augmentation, rng, encoder)
        return
    groups = subpop_samples(args, rng)
    total, i = sum(len(samples) for _, samples in groups), 0
    for attribute, samples in groups:
        for title, record in samples:
            i += 1
            show(i, total, title, record, augmentation, rng, encoder)
        compare(attribute, samples)


if __name__ == "__main__":
    main()
