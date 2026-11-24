"""Twin-2K-500-Mega-Study dataset loader and evaluation set builder."""

from pathlib import Path
import pandas as pd

from data.base import (
    DATA_DIR,
    DEFAULT_CACHE_DIR,
    SINGLE_PERSONA_ATTRS,
    add_one_hot,
    build_eval_set,
    build_persona_dataset,
    filter_by_qids,
    group_records_by_state,
    make_view,
)

TWIN2K_MEGASTUDY_PARQUET_PATH = DATA_DIR / "Twin-2K-500-Mega-Study" / "twin2k_megastudy_US_long.parquet"


def load_twin2k_megastudy_df(qids=None, parquet_path=TWIN2K_MEGASTUDY_PARQUET_PATH):
    """Load `twin2k_megastudy_US_long.parquet`, attach `year`/`qid`/`one_hot`, and optionally filter to `qids`."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Twin-2K-500-Mega-Study parquet not found: {path}")
    mega_df = pd.read_parquet(path)
    for col in ("option_codes", "option_texts"):
        mega_df[col] = mega_df[col].map(list)
    mega_df["year"] = "twin2k-mega"
    mega_df["qid"] = "twin2k-mega:" + mega_df["v"]
    add_one_hot(mega_df)
    return filter_by_qids(mega_df, qids)


def load_twin2k_megastudy_eval(
    *,
    attr_sets,
    include_country,
    qids=None,
    min_n=None,
    grouped=True,
    name="twin2k_megastudy",
    valid=False,
    parquet_path=TWIN2K_MEGASTUDY_PARQUET_PATH,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build a Twin-2K-500-Mega-Study evaluation set for the given `qids`, `attr_sets`, and `include_country`."""
    mega_df = load_twin2k_megastudy_df(qids=qids, parquet_path=parquet_path)
    return build_eval_set(
        mega_df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=min_n,
        grouped=grouped,
        name=name,
        valid=valid,
        cache_dir=cache_dir,
    )


def load_twin2k_megastudy_records(
    *,
    attr_sets,
    include_country,
    qids=None,
    min_n=None,
    grouped=False,
    parquet_path=TWIN2K_MEGASTUDY_PARQUET_PATH,
    desc="twin2k_megastudy_records",
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build (or load from cache) Twin-2K-500-Mega-Study persona records, optionally grouped by shared state."""
    mega_df = load_twin2k_megastudy_df(qids=qids, parquet_path=parquet_path)
    records = build_persona_dataset(
        mega_df,
        attr_sets=attr_sets,
        include_country=include_country,
        min_n=None if attr_sets is None else min_n,
        desc=desc,
        cache_dir=cache_dir,
    )
    if grouped:
        return group_records_by_state(records, view_fn=make_view, name="twin2k_megastudy")
    return records
