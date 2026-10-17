"""The dataset layer: what the trainer (train.py) trains, calibrates and tests on (PLAN-record-construction.md).

A recipe (configs/data/*.json) names a source and a construction:

    {"source": {"issp": {"datasets": "all", "calibration_frac": 0.01}},
     "construction": {"group_shares": {"min_n": 10}}}

  source        Where the respondents come from and which slots are held out for calibration (issp_sampler.IsspSource:
                the ISSP years, "all" or a list such as ["issp2020"]; optionally "countries": ["US"] to load only those
                countries, for training and calibration alike; optionally "calibration_min_n": 100 to keep only
                calibration groups of at least that many respondents, default 1 = every group).
  construction  How a training record is built from a slot (constructions.py: group_shares, single_respondent,
                fixed_subset).

build(recipe) gives the trainer a TrainingData: the slots of an epoch, one record per slot (`draw`), and the calibration
records. The calibration records come from the source and do not depend on the construction, so runs with different
constructions are compared on the same records. The test set (TEST_DATASET, SubPOP's evaluation split) is fixed and is
not part of any recipe. SubPOP is never training data: no source reads it.
"""
import hashlib
import json
import random
from pathlib import Path

from constructions import build_construction             # how records are built from a slot
from issp_sampler import IsspSource                      # ISSP respondent-level data

DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_DATASET = "subpop_eval"           # SubPOP evaluation split (US General Social Survey)
ISSP_DATASETS = sorted(path.name.removesuffix("_respondents.parquet")
                       for path in DATA_DIR.glob("*/*_respondents.parquet"))

GROUP_SIZES = ("1", "2-9", "10-99", "100+")   # respondents behind a target, as in constructions.describe


def group_size(n):
    """The GROUP_SIZES bucket of a target from n respondents (None when the record does not say)."""
    if n is None:
        return None
    return "1" if n == 1 else "2-9" if n < 10 else "10-99" if n < 100 else "100+"


def sha256(path):
    """Fingerprint of a file's contents: tells later whether a dataset was rebuilt after this run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_records(name):
    """All records of the dataset data/*/<name>.jsonl."""
    (path,) = DATA_DIR.glob(f"*/{name}.jsonl")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


# ---------------------------------------------------------------- recipes

def issp_source(datasets="all", calibration_frac=0.01, countries=None, calibration_min_n=1):
    """The ISSP source: `datasets` = "all" or a list of years such as ["issp2015", "issp2020"]; `countries` = None (every
    country) or a list of country codes such as ["US"], for training and calibration alike; `calibration_min_n` = the
    fewest respondents a calibration group may have (1 = every group)."""
    names = ISSP_DATASETS if datasets == "all" else list(datasets)
    unknown = sorted(set(names) - set(ISSP_DATASETS))
    if unknown:
        raise ValueError(f"unknown ISSP datasets {unknown}; known: {', '.join(ISSP_DATASETS)}")
    source = IsspSource(names, calibration_frac=calibration_frac, countries=countries,
                         calibration_min_n=calibration_min_n)
    if countries is not None:
        missing = sorted(set(countries) - {unit.code for unit in source.units})
        if missing:
            raise ValueError(f"no respondents from {missing} in {', '.join(names)}")
    return source


SOURCES = {"issp": issp_source}


def load_recipe(path):
    """A recipe file as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recipe_from_settings(settings):
    """The recipe of a run, from its run_info.json settings: the recipe itself for runs that have one, translated from
    the dataset flags of earlier runs (--train, --calibration_frac, --min_n, --single_respondent, --train_slots)."""
    if "recipe" in settings:
        return settings["recipe"]
    construction = ({"single_respondent": {}} if settings.get("single_respondent")
                    else {"group_shares": {"min_n": settings.get("min_n", 1)}})
    if settings.get("train_slots"):
        construction = {"fixed_subset": {"n": settings["train_slots"], "of": construction}}
    return {"source": {"issp": {"datasets": settings["train"], "calibration_frac": settings["calibration_frac"]}},
            "construction": construction}


class TrainingData:
    """What the trainer gets from a recipe: `slots` (one epoch = one pass through them), `draw(rng, slot)` (one record
    in the stored format), the calibration records, and what the data is for run_info.json."""

    def __init__(self, recipe):
        self.recipe = recipe
        ((source_name, source_settings),) = recipe["source"].items()
        if source_name not in SOURCES:
            raise ValueError(f"unknown source {source_name!r}; known: {', '.join(SOURCES)}")
        self.source = SOURCES[source_name](**source_settings)
        self.construction = build_construction(recipe["construction"])
        self.slots = self.construction.slots(self.source)

    def draw(self, rng, slot):
        """One training record for `slot`."""
        return self.construction.draw(self.source, rng, slot)[0]

    def calibration_records(self):
        return self.source.calibration_records()

    def statistics(self, seed):
        """What the construction's records look like, from draws with their own generator (training is unaffected)."""
        return self.construction.statistics(self.source, random.Random(seed))

    def file_hashes(self):
        """Fingerprints of the training files and the test set, for run_info.json."""
        paths = self.source.files() + list(DATA_DIR.glob(f"*/{TEST_DATASET}.jsonl"))
        return {path.name: {"file": str(path.relative_to(DATA_DIR.parent)), "sha256": sha256(path)} for path in paths}
