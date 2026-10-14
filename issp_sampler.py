"""Draw ISSP training records from respondent-level data (PLAN-respondent-sampling.md, step 3).

The data of a year (data/issp<year>/, written by its build_dataset.py) is one row per respondent with every attribute
and answer as data codes (issp<year>_respondents.parquet), the questions per country (issp<year>_questions.json) and
each attribute code's text per country (issp<year>_attributes.json). Nothing is aggregated in advance: each training
record is made when it is drawn.

  1. Slot: a (year, country, question) drawn uniformly among those with answers. A record holds this one question.
  2. Default mode (single_respondent=False):
     - k, the number of attributes shown besides the country: uniform from 0 to all attributes the slot's respondents
       have (no cap).
     - Which attributes: k of them, at random.
     - Their values: uniformly among the value combinations that occur among the slot's respondents (those who answered
       the question). Every persona counts the same, however many people it matches.
     - Target: the answer shares of the matching respondents (no minimum: a persona matching one person gives a one-hot
       target).
  3. Single-respondent mode (single_respondent=True):
     - Pick one respondent uniformly among those who answered the slot's question.
     - Show that respondent's full persona (all non-missing attributes) and use their single response (one-hot) as the
       target, without any group averaging.

Calibration: a share of each year's slots (`held_out`, chosen by a hash of year, country and question, so the same in
every run) is never drawn for training. Each held-out slot gives fixed records instead: the whole country, and one
record per value of each single attribute (e.g. Sex: Female), with the answer shares of all matching respondents.
(Note: calibration currently always uses these group-level records regardless of `single_respondent`; a future option
is to also calibrate on single-respondent full-persona records using cross-entropy / log-loss.)

A record has the stored format issp_augment expects: persona = Country, the k attributes, and the asked_of lines of
the question's block (who was asked); its question with type, criteria, instructions, label, target, src.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def dataset_files(name):
    """(respondent table, question file, attribute file) of a dataset such as 'issp2020'."""
    (respondents,) = DATA_DIR.glob(f"*/{name}_respondents.parquet")
    return (respondents, respondents.with_name(f"{name}_questions.json"),
            respondents.with_name(f"{name}_attributes.json"))


def label_of(qtype, key):
    """The stored `label` of an option key: level index (score), True/False (noul), option text (choice)."""
    return {"score": int, "noul": lambda k: k == "true"}.get(qtype, str)(key)


class Unit:
    """One country in one year: its respondents' attributes and answers as integer arrays (-1 = missing)."""

    def __init__(self, name, src, code, country, table, attribute_names, attribute_values, questions, asked_of):
        self.name, self.src, self.code, self.country = name, src, code, country
        self.attribute_names = attribute_names
        self.values, columns = [], []                    # per attribute: the value texts; their index per respondent
        for key in attribute_names:                      # data codes -> this country's text for them
            text = {int(c): t for c, t in attribute_values[key]["values"].get(code, {}).items()}
            codes, uniques = pd.factorize(table[key].map(text, na_action="ignore"), use_na_sentinel=True)
            self.values.append([str(u) for u in uniques])
            columns.append(codes)
        self.attrs = np.stack(columns, axis=1).astype(np.int32) if columns else np.zeros((len(table), 0), np.int32)
        self.questions, self.keys, self.answers, self.block_of = {}, {}, {}, {}
        for v, q in questions.items():
            spec = q["countries"].get(code)
            if spec is None:
                continue
            keys = list(dict.fromkeys(spec["codes"].values()))   # option keys in the order the options are listed
            index = {int(c): keys.index(k) for c, k in spec["codes"].items()}
            raw = table[v].to_numpy(dtype="float64", na_value=np.nan)
            answers = np.full(len(table), -1, np.int16)
            for c, i in index.items():
                answers[raw == c] = i
            if (answers >= 0).any():
                self.questions[v], self.keys[v], self.answers[v] = spec["question"], keys, answers
                self.block_of[v] = q["block"]
        self.asked_of = {block: lines.get(code, {}) for block, lines in asked_of.items()}


class IsspSampler:
    """Training records drawn from the respondent data of the given datasets (e.g. ['issp2015', ..., 'issp2024']).

    held_out: the share of each dataset's slots kept out of training for calibration (`self.held_out`); training draws
    only from `self.slots`.
    single_respondent: if True, each training draw picks a single respondent and uses their full non-missing persona
    and one-hot answer instead of sampling an attribute subset and averaging over matching respondents."""

    def __init__(self, names, held_out=0.0, single_respondent=False):
        self.units, self.slots, self.held_out = [], [], []
        self.single_respondent = single_respondent
        for name in names:
            slots = []
            respondents, question_file, attribute_file = dataset_files(name)
            table = pd.read_parquet(respondents)
            meta = json.loads(question_file.read_text(encoding="utf-8"))
            attribute_values = json.loads(attribute_file.read_text(encoding="utf-8"))
            attribute_names = [c for c in table.columns if c not in ("respondent", "country") and c not in meta["questions"]]
            for code, rows in table.groupby("country", sort=True):
                unit = Unit(name, meta["src"], code, meta["countries"][code], rows.reset_index(drop=True),
                            attribute_names, attribute_values, meta["questions"], meta["asked_of"])
                self.units.append(unit)
                slots += [(len(self.units) - 1, v) for v in unit.questions]
            # held out: the round(held_out * slots) slots of this dataset with the lowest hash of year, country, question
            by_hash = sorted(slots, key=lambda slot: hashlib.sha256(
                f"{name}/{self.units[slot[0]].code}/{slot[1]}".encode()).hexdigest())
            held = by_hash[:round(held_out * len(slots))]
            self.held_out += held
            self.slots += [slot for slot in slots if slot not in held]
        self.respondents = sum(len(u.attrs) for u in self.units)

    def record(self, rng, slot=None, keys=None):
        """(record, info): one training record drawn as described above.

        slot: (unit index, question id) instead of a random slot. keys: persona lines to group by instead of a random
        subset (all must be present). info: n per question, the attributes shown, how many value combinations there
        were."""
        while True:
            unit_index, v = slot if slot is not None else rng.choice(self.slots)
            unit = self.units[unit_index]
            questions = [v]
            pool = self.pool(unit, questions)
            if self.single_respondent and keys is None:
                r = int(pool[rng.randrange(len(pool))])
                chosen = [j for j, code in enumerate(unit.attrs[r]) if code >= 0]
                combo = unit.attrs[r, chosen]
                match = np.array([r])
                record = self.make(unit, questions, chosen, combo, match)
                info = {"dataset": unit.name, "country": unit.code, "question": v, "attributes": len(chosen),
                        "combinations": len(pool), "n": record["n"]}
                return record, info
            if keys is None:
                present = [j for j in range(len(unit.attribute_names)) if (unit.attrs[pool, j] >= 0).any()]
                chosen = rng.sample(present, rng.randint(0, len(present)))        # k uniform from 0 to all
            else:
                chosen = [unit.attribute_names.index(key) for key in keys]
            values = unit.attrs[np.ix_(pool, chosen)]
            complete = (values >= 0).all(axis=1)         # respondents who have every chosen attribute
            if complete.any():
                break
            if keys is not None and slot is not None:
                raise ValueError(f"no respondent in {unit.name} {unit.code} {v} has all of {keys}")
            # nobody has the drawn attributes together: draw again
        combos = np.unique(values[complete], axis=0)
        combo = combos[rng.randrange(len(combos))]       # uniform among the value combinations that occur
        match = pool[complete & (values == combo).all(axis=1)]
        record = self.make(unit, questions, chosen, combo, match)
        info = {"dataset": unit.name, "country": unit.code, "question": v, "attributes": len(chosen),
                "combinations": len(combos), "n": record["n"]}
        return record, info

    @staticmethod
    def pool(unit, questions):
        """Rows of the unit's respondents who answered at least one of the questions."""
        return np.flatnonzero(np.any([unit.answers[v] >= 0 for v in questions], axis=0))

    @staticmethod
    def make(unit, questions, chosen, combo, match):
        """The stored-format record of the respondents `match`, described by attributes `chosen` = values `combo`;
        `questions` are of one block (they share its asked_of lines)."""
        block = unit.block_of[questions[0]]
        persona = {"Country": unit.country}
        persona |= {unit.attribute_names[j]: unit.values[j][int(i)] for j, i in zip(chosen, combo)}
        persona |= unit.asked_of.get(block, {})
        stored, n = {}, {}
        for v in questions:
            answers = unit.answers[v][match]
            answers = answers[answers >= 0]
            if not len(answers):
                continue
            keys, spec = unit.keys[v], unit.questions[v]
            counts = np.bincount(answers, minlength=len(keys))
            stored[v] = {**spec, "label": label_of(spec["type"], keys[int(counts.argmax())]),
                         "target": {keys[i]: float(c / len(answers)) for i, c in enumerate(counts) if c},
                         "src": unit.src}
            n[v] = int(len(answers))
        group = "/".join([unit.code, *(unit.values[j][int(i)] for j, i in zip(chosen, combo))])
        return {"group": group, "country_code": unit.code, "block": block, "persona": persona, "n": n,
                "questions": stored}

    def statistics(self, rng, records=10_000):
        """What the drawn records look like: share of targets by number of matching respondents, and how many
        attributes personas show (from `records` draws)."""
        sizes, lengths = Counter(), Counter()
        for _ in range(records):
            _, info = self.record(rng)
            lengths[info["attributes"]] += 1
            for n in info["n"].values():
                sizes["1" if n == 1 else "2-9" if n < 10 else "10-99" if n < 100 else "100+"] += 1
        total = sum(sizes.values())
        return {"records_drawn": records,
                "targets_by_respondents": {k: round(sizes[k] / total, 3) for k in ("1", "2-9", "10-99", "100+")},
                "attributes_shown": {k: round(lengths[k] / records, 3) for k in sorted(lengths)}}

    def calibration_records(self):
        """The fixed records of the held-out slots: per slot, the whole country, then one record per value of each
        attribute that occurs among the slot's respondents.
        Note: kept as group-level distributions even when `single_respondent=True`; a future option is to calibrate on
        single-respondent full-persona records (one-hot targets scored by cross-entropy / log-loss)."""
        records = []
        for unit_index, v in self.held_out:
            unit = self.units[unit_index]
            pool = self.pool(unit, [v])
            records.append(self.make(unit, [v], [], [], pool))
            for j in range(len(unit.attribute_names)):
                codes = unit.attrs[pool, j]
                for code in np.unique(codes[codes >= 0]):
                    records.append(self.make(unit, [v], [j], [code], pool[codes == code]))
        return records

    def group(self, name, code, persona_values, questions):
        """The fixed record of one persona (no sampling): `persona_values` = {persona line: value}, e.g.
        {"Sex": "Female"} ({} = the whole country); `questions` = question ids of one block."""
        unit = next(u for u in self.units if u.name == name and u.code == code)
        questions = [v for v in questions if v in unit.answers]
        chosen = [unit.attribute_names.index(key) for key in persona_values]
        combo = [unit.values[j].index(persona_values[unit.attribute_names[j]]) for j in chosen]
        pool = self.pool(unit, questions)
        match = pool[(unit.attrs[np.ix_(pool, chosen)] == combo).all(axis=1)]
        return self.make(unit, questions, chosen, combo, match)
