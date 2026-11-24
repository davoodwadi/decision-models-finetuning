"""Tokenize data rows into training items: {dataset, caseid, qids, enc, num_options, y_flat, outcome_rows}.

One item = one data row (one answered task) of a choice_df: the respondent's persona state + that task, with the
task's beta part from its `beta_ols` module (modules code task by task, so a one-row slice is a valid input).
Kev's prepared targets must equal the beta part's one-hot choices; that guarantees the model's logits come back in
the same order the outcome rows index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

import data.base
import data.persona
import kev.api
import kev.model
import models.base
import models.kev_pointer
from data.base import build_long_records, make_grouped_view

import beta_ols.core
from beta_ols.core import iter_respondents

from beta_train.cache import dataframe_fingerprint, load_or_compute, source_fingerprint


# This file's own module, so its source is part of the cache keys it writes
THIS_MODULE = sys.modules[__name__]

# Code that turns a respondent's rows into Kev's encoding: records and persona text (data.base, data.persona),
# options and targets (models.base, kev.api), tokenization (kev.model, models.kev_pointer). Any change rebuilds items.
ENCODING_MODULES = (data.base, data.persona, models.base, kev.api, kev.model, models.kev_pointer)


def make_item(prepare_fn, dataset_name: str, dataset_module, caseid: str, r_df: pd.DataFrame) -> dict:
    # Rows of one respondent (here: a single row) -> one training item, labelled with the (pseudo) dataset it belongs to
    part = dataset_module.respondent_beta_part(r_df)
    r_records = build_long_records(
        r_df, persona_cols=dataset_module.PERSONA_COLS, include_country=False, year=dataset_module.NAME
    )
    r_enc, r_targets = prepare_fn(make_grouped_view(r_records))
    prepared_counts = [int(q["target"].shape[0]) for q in r_targets]
    prepared_targets = torch.cat([q["target"].to(dtype=torch.float64) for q in r_targets])
    if prepared_counts != part.num_options or not torch.equal(prepared_targets, part.y_flat):
        raise AssertionError(f"{caseid}: Kev prepared targets do not match the beta part's one-hot choices")
    return {
        "dataset": dataset_name,
        "caseid": caseid,
        "qids": [str(qid) for qid in r_df["qid"]],
        "enc": r_enc,
        "num_options": part.num_options,
        "y_flat": part.y_flat,
        "outcome_rows": part.outcome_rows,
    }


def build_items(prepare_fn, dataset_modules: dict, choice_dfs: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    # Every data row of every given (pseudo) dataset -> {dataset name: [items]}, respondent by respondent in the
    # module's order (the order the population evaluation's flat option vector uses). `dataset_modules` maps each name
    # (e.g. "berke2023/delivery_choice_v0", see beta_ols.core.split_by_task) to the module that codes its rows.
    items_by_dataset = {}
    for dataset_name, dataset_module in dataset_modules.items():
        choice_df = choice_dfs[dataset_name]
        dataset_items = []
        progress = tqdm(total=len(choice_df), desc=f"items: {dataset_name}")
        for caseid, r_df in iter_respondents(choice_df):
            for row_idx in range(len(r_df)):
                row_df = r_df.iloc[[row_idx]].reset_index(drop=True)
                dataset_items.append(make_item(prepare_fn, dataset_name, dataset_module, caseid, row_df))
                progress.update(1)
        progress.close()
        items_by_dataset[dataset_name] = dataset_items
    return items_by_dataset


def build_items_cached(model, dataset_modules: dict, choice_dfs: dict[str, pd.DataFrame], cache_dir: Path) -> tuple[dict, pd.DataFrame]:
    # Same as build_items(model.prepare, ...), but each dataset's items are cached on disk. The cache key covers the
    # dataset's rows, the tokenizer and Kev's encoding code, the dataset module's coding, and this file, so any change
    # rebuilds. Returns (items by dataset, one row per dataset saying whether it was loaded or built).
    items_by_dataset = {}
    report_rows = []
    for dataset_name, dataset_module in dataset_modules.items():
        key = {
            "kind": "training_items_per_row",
            "dataset": dataset_name,
            "tokenizer": str(model.tok.name_or_path),
            "option_isolation": bool(model.encoder.option_isolation),
            "persona_cols": list(dataset_module.PERSONA_COLS),
            "choice_rows": dataframe_fingerprint(choice_dfs[dataset_name]),
            "code": source_fingerprint(dataset_module, beta_ols.core, THIS_MODULE, *ENCODING_MODULES),
        }

        def build_one_dataset():
            return build_items(model.prepare, {dataset_name: dataset_module}, choice_dfs)[dataset_name]

        dataset_items, loaded = load_or_compute(cache_dir, f"items_{dataset_name}", key, build_one_dataset)
        items_by_dataset[dataset_name] = dataset_items
        report_rows.append({"dataset": dataset_name, "items (rows)": len(dataset_items), "source": "cache" if loaded else "built"})
    return items_by_dataset, pd.DataFrame(report_rows).set_index("dataset")


def items_summary(items_by_dataset: dict[str, list[dict]], specs_by_dataset: dict[str, dict]) -> pd.DataFrame:
    # Items (data rows), respondents, and regression groups per dataset
    summary_rows = []
    for dataset_name, dataset_items in items_by_dataset.items():
        caseids = set()
        for item in dataset_items:
            caseids.add(item["caseid"])
        summary_rows.append(
            {
                "dataset": dataset_name,
                "items (rows)": len(dataset_items),
                "respondents": len(caseids),
                "regression_groups": ", ".join(specs_by_dataset[dataset_name]),
            }
        )
    return pd.DataFrame(summary_rows).set_index("dataset")
