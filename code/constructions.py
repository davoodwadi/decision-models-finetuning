"""Ways of building training records from a source's respondents (PLAN-record-construction.md).

A construction decides what one epoch passes over (`slots(source)`: an epoch is one pass through them) and how one
record is drawn for a slot (`draw(source, rng, slot)` -> (record, info)). Every construction gives records in the
stored format (IsspSource.make), so the trainer and its loss do not depend on which one a run uses. Adding one: write
a class with these methods, add it to CONSTRUCTIONS, and name it in a recipe (configs/data/*.json).

  group_shares       A random subset of the slot's attributes (k uniform from min_attributes, default 0, to
                     max_attributes, default all), a random value combination shared by at least min_n respondents, and
                     as target the answer shares of the respondents who match. min_attributes 1: never the country
                     alone; min_attributes = max_attributes = 2: always exactly two attributes besides the country.
  single_respondent  One respondent who answered the slot's question, with their full persona (every attribute they
                     have) and their own answer as target (one-hot).
  fixed_subset       n slots of another construction (`of`), each drawn once; the same records every epoch and in every
                     run (their own generator), for quick runs.
"""
import itertools
import random
from collections import Counter

import numpy as np

STATISTICS_DRAWS = 10_000   # draws describing a construction's records in run_info.json


def describe(infos):
    """What drawn records look like, from the `info` that draw() returns with each record: share of targets by number
    of matching respondents, and how many attributes personas show."""
    records = len(infos)
    sizes, lengths = Counter(), Counter()
    for info in infos:
        lengths[info["attributes"]] += 1
        for n in info["n"].values():
            sizes["1" if n == 1 else "2-9" if n < 10 else "10-99" if n < 100 else "100+"] += 1
    total = sum(sizes.values())
    return {"records_drawn": records,
            "targets_by_respondents": {k: round(sizes[k] / total, 3) for k in ("1", "2-9", "10-99", "100+")},
            "attributes_shown": {k: round(lengths[k] / records, 3) for k in sorted(lengths)}}


class Construction:
    """Shared by the constructions: a description of their records from draws with their own generator."""

    def statistics(self, source, rng):
        """describe() of STATISTICS_DRAWS records drawn from random slots (`rng` is not the training generator)."""
        slots = self.slots(source)
        return describe([self.draw(source, rng, rng.choice(slots))[1] for _ in range(STATISTICS_DRAWS)])


class GroupShares(Construction):
    """Random attribute subset and value combination; target = the answer shares of everyone who matches."""

    def __init__(self, min_n=1, min_attributes=0, max_attributes=None):
        if max_attributes is not None and max_attributes < min_attributes:
            raise ValueError(f"max_attributes {max_attributes} < min_attributes {min_attributes}")
        self.min_n, self.min_attributes, self.max_attributes = min_n, min_attributes, max_attributes

    def slots(self, source):
        """The source's slots that can give min_n respondents, in a persona of at least min_attributes attributes."""
        return [(u, v) for u, v in source.slots if self.qualifies(source.units[u], source.pool(source.units[u], [v]))]

    def qualifies(self, unit, pool):
        """Whether some min_attributes attributes have a value combination shared by min_n of `pool`. A combination
        shared by min_n is shared by min_n in each of its sub-combinations, so for 1 attribute one column suffices."""
        if len(pool) < self.min_n or self.min_attributes == 0:
            return len(pool) >= self.min_n                # the country alone: everyone who answered
        for chosen in itertools.combinations(range(len(unit.attribute_names)), self.min_attributes):
            values = unit.attrs[np.ix_(pool, chosen)]
            values = values[(values >= 0).all(axis=1)]
            if len(values) and np.unique(values, axis=0, return_counts=True)[1].max() >= self.min_n:
                return True
        return False

    def draw(self, source, rng, slot, keys=None):
        """(record, info) for `slot`. keys: persona lines to group by instead of a random subset (all must be present).
        If no value combination of the drawn attributes has min_n respondents, the attributes are drawn again."""
        unit_index, v = slot
        unit = source.units[unit_index]
        questions = [v]
        pool = source.pool(unit, questions)
        while True:
            if keys is None:
                present = [j for j in range(len(unit.attribute_names)) if (unit.attrs[pool, j] >= 0).any()]
                most = len(present) if self.max_attributes is None else min(self.max_attributes, len(present))
                chosen = rng.sample(present, rng.randint(self.min_attributes, most))   # k uniform: min_ to max_attributes
            else:
                chosen = [unit.attribute_names.index(key) for key in keys]
            values = unit.attrs[np.ix_(pool, chosen)]
            complete = (values >= 0).all(axis=1)         # respondents who have every chosen attribute
            combos, counts = np.unique(values[complete], axis=0, return_counts=True)
            combos = combos[counts >= self.min_n]        # value combinations shared by at least min_n respondents
            if len(combos):
                break
            if keys is not None:
                raise ValueError(f"no {self.min_n} respondents in {unit.name} {unit.code} {v} share values of {keys}")
        combo = combos[rng.randrange(len(combos))]       # uniform among the value combinations that qualify
        match = pool[complete & (values == combo).all(axis=1)]
        record = source.make(unit, questions, chosen, combo, match)
        info = {"dataset": unit.name, "country": unit.code, "question": v, "attributes": len(chosen),
                "combinations": len(combos), "n": record["n"]}
        return record, info


class SingleRespondent(Construction):
    """One respondent with their full persona; target = their own answer."""

    def slots(self, source):
        return list(source.slots)

    def draw(self, source, rng, slot):
        unit_index, v = slot
        unit = source.units[unit_index]
        pool = source.pool(unit, [v])
        r = int(pool[rng.randrange(len(pool))])
        chosen = [j for j, code in enumerate(unit.attrs[r]) if code >= 0]
        record = source.make(unit, [v], chosen, unit.attrs[r, chosen], np.array([r]))
        info = {"dataset": unit.name, "country": unit.code, "question": v, "attributes": len(chosen),
                "combinations": len(pool), "n": record["n"]}
        return record, info


class FixedSubset(Construction):
    """n slots of construction `of`, each drawn once with their own generator: the same records in every run, whatever
    the run's seed, and every epoch."""

    def __init__(self, n, of):
        self.n, self.of, self.fixed = n, of, None

    def slots(self, source):
        if self.fixed is None:
            rng = random.Random("train_slots")
            chosen = rng.sample(self.of.slots(source), self.n)
            self.fixed = {slot: self.of.draw(source, rng, slot) for slot in chosen}   # (record, info), in this order
        return list(self.fixed)

    def draw(self, source, rng, slot):
        return self.fixed[slot]

    def statistics(self, source, rng):
        self.slots(source)
        return describe([info for _, info in self.fixed.values()])


CONSTRUCTIONS = {"group_shares": GroupShares, "single_respondent": SingleRespondent, "fixed_subset": FixedSubset}


def build_construction(spec):
    """A construction from its recipe entry, {name: settings}, e.g. {"group_shares": {"min_n": 10}} or
    {"fixed_subset": {"n": 128, "of": {"group_shares": {"min_n": 10}}}}."""
    ((name, settings),) = spec.items()
    if name not in CONSTRUCTIONS:
        raise ValueError(f"unknown construction {name!r}; known: {', '.join(CONSTRUCTIONS)}")
    settings = dict(settings)
    if "of" in settings:
        settings["of"] = build_construction(settings["of"])
    return CONSTRUCTIONS[name](**settings)
