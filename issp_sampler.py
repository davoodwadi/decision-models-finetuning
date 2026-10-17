"""The ISSP source of training data: respondent-level data loaded as it is, never aggregated in advance
(PLAN-respondent-sampling.md, PLAN-record-construction.md).

The data of a year (data/issp<year>/, written by its build_dataset.py) is one row per respondent with every attribute
and answer as data codes (issp<year>_respondents.parquet), the questions per country (issp<year>_questions.json) and
each attribute code's text per country (issp<year>_attributes.json).

IsspSource loads them and offers what every way of building records needs: the (year, country, question) slots, the
respondents who answered a slot (`pool`), and the stored record of any set of respondents (`make`). How a training
record is drawn from a slot (group shares, single respondent, ...) is the job of a construction (constructions.py).

Calibration: a share of each year's slots (`held_out`, chosen by a hash of year, country and question, so the same in
every run) is never drawn for training. With `countries`, only those countries are loaded (e.g. ["US"]): training and
calibration both come from them, the share is taken of their slots, and since slots are held out in a fixed hash order,
a larger share keeps every slot a smaller one held out and adds more. Each held-out slot gives fixed records instead: the whole country, and one
record per value of each single attribute (e.g. Sex: Female), with the answer shares of all matching respondents (with
calibration_min_n, only groups of at least that many respondents). They
are the same whatever construction a run trains with, so runs stay comparable.

A record has the stored format issp_augment expects: persona = Country, the attributes shown, and the asked_of lines of
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


class IsspSource:
    """The respondent data of the given datasets (e.g. ['issp2015', ..., 'issp2024']).

    calibration_frac: the share of each dataset's slots kept out of training for calibration (`self.held_out`);
    constructions draw training records only from `self.slots`. countries: country codes to load (e.g. ["US"]); None =
    every country. calibration_min_n: fewest respondents a calibration record's group may have (1 = every group)."""

    def __init__(self, names, calibration_frac=0.0, countries=None, calibration_min_n=1):
        self.names = list(names)
        self.calibration_min_n = calibration_min_n
        self.countries = None if countries is None else list(countries)
        self.units, self.slots, self.held_out = [], [], []
        for name in self.names:
            slots = []
            respondents, question_file, attribute_file = dataset_files(name)
            table = pd.read_parquet(respondents)
            meta = json.loads(question_file.read_text(encoding="utf-8"))
            attribute_values = json.loads(attribute_file.read_text(encoding="utf-8"))
            attribute_names = [c for c in table.columns if c not in ("respondent", "country") and c not in meta["questions"]]
            for code, rows in table.groupby("country", sort=True):
                if self.countries is not None and code not in self.countries:
                    continue
                unit = Unit(name, meta["src"], code, meta["countries"][code], rows.reset_index(drop=True),
                            attribute_names, attribute_values, meta["questions"], meta["asked_of"])
                self.units.append(unit)
                slots += [(len(self.units) - 1, v) for v in unit.questions]
            # held out: the round(calibration_frac * slots) slots of this dataset with the lowest hash of year, country, question
            by_hash = sorted(slots, key=lambda slot: hashlib.sha256(
                f"{name}/{self.units[slot[0]].code}/{slot[1]}".encode()).hexdigest())
            held = by_hash[:round(calibration_frac * len(slots))]
            self.held_out += held
            self.slots += [slot for slot in slots if slot not in held]
        self.respondents = sum(len(u.attrs) for u in self.units)

    def files(self):
        """The files the data was loaded from, to fingerprint in run_info.json."""
        return [path for name in self.names for path in dataset_files(name)]

    def summary(self):
        """What was loaded, for run_info.json and the run's first printout."""
        return {"respondents": self.respondents, "country_years": len(self.units), "calibration_slots": len(self.held_out)}

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

    def calibration_records(self):
        """The fixed records of the held-out slots: per slot, the whole country, then one record per value of each
        attribute that occurs among the slot's respondents, each from at least calibration_min_n respondents.
        Group-level whatever the construction."""
        records = []
        for unit_index, v in self.held_out:
            unit = self.units[unit_index]
            pool = self.pool(unit, [v])
            if len(pool) >= self.calibration_min_n:
                records.append(self.make(unit, [v], [], [], pool))
            for j in range(len(unit.attribute_names)):
                codes = unit.attrs[pool, j]
                for code in np.unique(codes[codes >= 0]):
                    if (codes == code).sum() >= self.calibration_min_n:
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
