"""ISSP (2015-2024) dataset loader and evaluation set builder."""

from pathlib import Path
import pandas as pd

from data.base import (
    DATA_DIR,
    DEFAULT_CACHE_DIR,
    SINGLE_PERSONA_ATTRS,
    add_one_hot,
    build_eval_set,
    filter_by_qids,
)
from data.persona import apply_persona_mapping

ISSP_YEARS = ("2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024")
_YEAR_DF_CACHE = {}


def _get_year_df(year_str, data_dir):
    cache_key = (str(Path(data_dir).resolve()), year_str)
    if cache_key not in _YEAR_DF_CACHE:
        df = pd.read_parquet(Path(data_dir) / f"issp{year_str}" / f"issp{year_str}_US_long.parquet")
        df = apply_persona_mapping(df, dataset="issp")
        for col in ("option_codes", "option_texts"):
            df[col] = df[col].map(list)
        df["year"] = year_str
        df["qid"] = year_str + ":" + df["v"]
        add_one_hot(df)
        _YEAR_DF_CACHE[cache_key] = df
    return _YEAR_DF_CACHE[cache_key]


def load_issp_df(qids=None, years=ISSP_YEARS, data_dir=DATA_DIR):
    """Load ISSP waves into a standardized long DataFrame, optionally filtered to `qids`."""
    if qids is not None:
        req_years = {str(q).split(":", 1)[0] for q in qids}
        years = [y for y in years if str(y) in req_years]
        if not years:
            raise ValueError(f"No matching ISSP years found for requested qids (e.g. {list(qids)[:5]})")

    frames = [_get_year_df(str(year), data_dir) for year in years]
    df = pd.concat(frames, ignore_index=True)
    return filter_by_qids(df, qids)


def load_issp_eval(
    *,
    attr_sets,
    include_country,
    qids=None,
    min_n=None,
    grouped=True,
    name="issp",
    valid=False,
    data_dir=DATA_DIR,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build an ISSP evaluation set for the given `qids`, `attr_sets`, and `include_country`."""
    df = load_issp_df(qids=qids, data_dir=data_dir)
    return build_eval_set(
        df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        grouped=grouped,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )

