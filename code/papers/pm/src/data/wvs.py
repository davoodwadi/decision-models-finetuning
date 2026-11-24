"""WVS Wave 7 (US) dataset loader and evaluation set builder."""

from pathlib import Path
import pandas as pd

from data.base import (
    DATA_DIR,
    DEFAULT_CACHE_DIR,
    add_one_hot,
    build_eval_set,
    filter_by_qids,
)
from data.persona import apply_persona_mapping

WVS_PARQUET_PATH = DATA_DIR / "WVS-Wave7" / "wvs7_US_long.parquet"


def load_wvs_df(qids=None, parquet_path=WVS_PARQUET_PATH):
    """Load `wvs7_US_long.parquet`, map its WVS demographic columns to global personas, attach `year`/`qid`/`one_hot`,
    and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"WVS parquet not found: {path}")
    df = pd.read_parquet(path)
    wvs_cols = list(df.columns[1:df.columns.get_loc("v")])   # WVS demographic columns, between CASEID and v
    df = apply_persona_mapping(df, dataset="wvs").drop(columns=wvs_cols)   # keep only the global persona columns
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)
    df["year"] = "wvs7"
    df["qid"] = "wvs7:" + df["v"]
    add_one_hot(df)
    return filter_by_qids(df, qids)


def load_wvs_eval(
    *,
    attr_sets,
    include_country,
    qids=None,
    min_n=None,
    grouped=True,
    name="wvs",
    valid=False,
    parquet_path=WVS_PARQUET_PATH,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build a WVS evaluation set for the given `qids`, `attr_sets`, and `include_country`."""
    df = load_wvs_df(qids=qids, parquet_path=parquet_path)
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
