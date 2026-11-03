"""ISSP (2015-2024) dataset loader, persona record builder, and state-grouped evaluation loader."""

import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data.grummon import add_one_hot, group_records_by_state, make_long_view
from data.persona import PERSONA_MAPS, PERSONA_NAMES, apply_persona_mapping

CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
SPLITS_PATH = CODE_DIR / "splits" / "split.json"
DEFAULT_CACHE_DIR = CODE_DIR / ".cache"

SINGLE_PERSONA_ATTRS = [c for c in PERSONA_NAMES.keys() if c != "c_alphan"]
PAIR_PERSONA_ATTRS = [list(c) for c in itertools.combinations(SINGLE_PERSONA_ATTRS, 2)]


def load_issp_splits(splits_path=SPLITS_PATH, data_dir=DATA_DIR):
    """Load ISSP (2015-2024) train, valid, and test DataFrames according to `splits/split.json`."""
    splits_by_year = json.loads(Path(splits_path).read_text())
    split_frames = {"train": [], "valid": [], "test": []}
    data_dir = Path(data_dir)

    for year, year_splits in splits_by_year.items():
        df = pd.read_parquet(data_dir / f"issp{year}" / f"issp{year}_US_long.parquet")
        df = apply_persona_mapping(df, dataset="issp")

        for col in ("option_codes", "option_texts"):
            df[col] = df[col].map(list)
        df["year"] = year
        df["qid"] = year + ":" + df["v"]
        add_one_hot(df)

        for split_name in ("train", "valid", "test"):
            sub = df[df["v"].isin(year_splits[split_name])].copy()
            split_frames[split_name].append(sub)

    return {
        "train": pd.concat(split_frames["train"], ignore_index=True),
        "valid": pd.concat(split_frames["valid"], ignore_index=True),
        "test": pd.concat(split_frames["test"], ignore_index=True),
    }


def make_record(sub_df, cols):
    """Build a single persona-question record dict from a respondent row or subgroup DataFrame."""
    if isinstance(sub_df, dict):
        row = sub_df
        human_dist = row["one_hot"]
        n = 1
    else:
        for c in cols:
            if sub_df[c].nunique(dropna=False) != 1:
                raise ValueError(f"sub_df must have a single value for {c!r}, got {sub_df[c].unique().tolist()}")
        row = sub_df.iloc[0]
        human_dist = np.stack(sub_df["one_hot"].values).mean(axis=0)
        n = len(sub_df)

    personas = {c: row[c] for c in cols if c != "c_alphan"}
    return {
        "qid": row["qid"],
        "year": row["year"],
        "v": row["v"],
        "type": row["type"],
        "question": row["question"],
        "option_texts": row["option_texts"],
        "country": row["c_alphan"],
        "personas": personas,
        "attribute": "+".join(personas.keys()) if personas else "none",
        "persona": ", ".join(f"{PERSONA_NAMES[c]}={v}" for c, v in personas.items()) if personas else "All",
        "n": n,
        "human_dist": human_dist,
    }


def build_persona_dataset(df, attr_sets=None, min_n=None, desc="build_persona_dataset", cache_dir=DEFAULT_CACHE_DIR):
    """Build (or load from pickle cache) persona records for given attribute combinations and minimum sample size."""
    cache_file = None
    if cache_dir is not None:
        key = repr((len(df), df["qid"].iloc[0], df["qid"].iloc[-1], attr_sets, min_n, PERSONA_MAPS))
        digest = hashlib.md5(key.encode()).hexdigest()[:8]
        cache_file = Path(cache_dir) / f"{desc}_min{min_n}_{digest}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                records = pickle.load(f)
            print(f"Loaded {len(records):,} records from {cache_file.name}")
            return records

    records = []
    q_groups = list(df.groupby("qid", sort=False))
    for _, q_df in tqdm(q_groups, desc=desc, unit="question"):
        active_cols = [c for c in PERSONA_NAMES if c != "c_alphan" and c in q_df.columns and q_df[c].notna().any()]
        active_set = set(active_cols)
        if attr_sets is None:
            for row in q_df.to_dict("records"):
                cols = [c for c in active_cols if pd.notna(row[c])]
                records.append(make_record(row, cols))
            continue

        sets = [[attr_sets]] if isinstance(attr_sets, str) else attr_sets
        for cols in sets:
            cols = [cols] if isinstance(cols, str) else list(cols)
            if not cols:
                if min_n is None or len(q_df) >= min_n:
                    records.append(make_record(q_df, cols))
                continue
            if not active_set.issuperset(cols):
                continue
            for _, sub_df in q_df.groupby(cols, sort=False, observed=True):
                if min_n is None or len(sub_df) >= min_n:
                    records.append(make_record(sub_df, cols))

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(records):,} records to {cache_file.name}")

    return records


make_view = make_long_view


def load_issp_eval(split="test", min_n=100, grouped=True, split_df=None, cache_dir=DEFAULT_CACHE_DIR):
    """Load ISSP `valid` or `test` subpopulation evaluation set (grouped by state when `grouped=True`)."""
    if split not in ("valid", "test"):
        raise ValueError(f"split must be 'valid' or 'test', got {split!r}")
    if split_df is None:
        split_df = load_issp_splits()[split]
    records = build_persona_dataset(
        split_df,
        attr_sets=SINGLE_PERSONA_ATTRS,
        min_n=min_n,
        desc=f"{split}_records",
        cache_dir=cache_dir,
    )
    if grouped:
        return group_records_by_state(records, view_fn=make_view, name=f"issp_{split}")
    return records
