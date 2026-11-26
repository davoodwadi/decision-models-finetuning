"""Exact boolean-mask empirical k-trait marginal lookup over `persona_spec.yaml` personas."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.persona_spec import PersonaAttribute

DEFAULT_K_SCHEDULE = (1, 2, 3, 4, 5, "full", "full")


def validate_k_schedule(k_schedule) -> tuple[int | str, ...]:
    """Validate and normalize `k_schedule` as a non-empty tuple of positive ints and/or `'full'`."""
    if isinstance(k_schedule, (str, bytes)):
        raise TypeError(f"k_schedule must be a sequence of ints/'full', got {k_schedule!r}")
    schedule = tuple(k_schedule)
    if not schedule:
        raise ValueError("k_schedule must not be empty")
    for item in schedule:
        if item == "full":
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"Each entry in k_schedule must be an int >= 1 or 'full', got {item!r}")
    return schedule


@dataclass(frozen=True)
class MarginalLookup:
    """Precomputed `(trait, value)` boolean masks and per-question unique-persona vote arrays."""

    persona_cols: tuple[str, ...]
    include_country: bool
    num_unique_personas: int
    trait_masks: dict[tuple[str, object], np.ndarray]
    qid_persona_ids: dict[str, np.ndarray]
    qid_vote_counts: dict[str, np.ndarray]


def build_marginal_lookup(
    df: pd.DataFrame,
    *,
    spec: dict[str, PersonaAttribute],
    include_country: bool = False,
) -> MarginalLookup:
    """Index unique full personas in `df` into boolean `(trait, value)` masks and per-qid vote arrays."""
    if df.empty:
        raise ValueError("df must not be empty")
    for req_col in ("qid", "one_hot"):
        if req_col not in df.columns:
            raise KeyError(f"Expected column {req_col!r} in df")
    if include_country and "country" not in df.columns:
        raise KeyError("Expected 'country' column when include_country=True")

    persona_cols = tuple(
        c for c in spec if c in df.columns and df[c].notna().any() and (include_country or c != "country")
    )
    if not persona_cols:
        raise KeyError("No non-empty persona columns from persona_spec.yaml found in df")

    work = df.reset_index(drop=True)
    persona_groups = work.groupby(list(persona_cols), dropna=False, sort=False)
    row_persona_id = persona_groups.ngroup().to_numpy()
    unique_personas_df = persona_groups.size().rename("obs_count").reset_index()
    num_unique_personas = len(unique_personas_df)

    trait_masks: dict[tuple[str, object], np.ndarray] = {}
    for p_id, p_row in enumerate(unique_personas_df[list(persona_cols)].to_dict("records")):
        for col, val in p_row.items():
            if pd.notna(val):
                key = (col, val)
                if key not in trait_masks:
                    trait_masks[key] = np.zeros(num_unique_personas, dtype=bool)
                trait_masks[key][p_id] = True

    qid_persona_ids: dict[str, np.ndarray] = {}
    qid_vote_counts: dict[str, np.ndarray] = {}
    for qid, q_sub in work.groupby("qid", sort=False):
        p_ids = row_persona_id[q_sub.index.to_numpy()]
        one_hots = np.stack(q_sub["one_hot"].to_numpy())
        if not np.allclose(one_hots.sum(axis=1), 1.0):
            raise ValueError(f"Expected all one_hot rows for qid={qid!r} to sum to 1.0")
        uniq_p_ids, inverse_idx = np.unique(p_ids, return_inverse=True)
        summed_votes = np.zeros((len(uniq_p_ids), one_hots.shape[1]), dtype=np.float64)
        np.add.at(summed_votes, inverse_idx, one_hots)
        qid_persona_ids[qid] = uniq_p_ids
        qid_vote_counts[qid] = summed_votes

    return MarginalLookup(
        persona_cols=persona_cols,
        include_country=bool(include_country),
        num_unique_personas=num_unique_personas,
        trait_masks=trait_masks,
        qid_persona_ids=qid_persona_ids,
        qid_vote_counts=qid_vote_counts,
    )


def augment_group_with_marginals(
    base_group: list[dict],
    subpersona: dict[str, str],
    lookup: MarginalLookup,
) -> list[dict]:
    """Replace `personas`, `n`, and `human_dist` in `base_group` with exact empirical marginals for `subpersona`."""
    if not base_group:
        raise ValueError("base_group must not be empty")
    if not subpersona:
        raise ValueError("subpersona must not be empty")

    matching_mask = np.ones(lookup.num_unique_personas, dtype=bool)
    for col, val in subpersona.items():
        matching_mask &= lookup.trait_masks[(col, val)]

    marginal_group: list[dict] = []
    for rec in base_group:
        qid = rec["qid"]
        matched = matching_mask[lookup.qid_persona_ids[qid]]
        vote_counts = lookup.qid_vote_counts[qid][matched].sum(axis=0)
        matching_n = int(round(float(vote_counts.sum())))
        if matching_n < 1:
            raise ValueError(
                f"Expected matching_n >= 1 for qid={qid!r} and subpersona={subpersona!r}, got {matching_n}"
            )
        marginal_group.append(
            {
                **rec,
                "personas": subpersona,
                "n": matching_n,
                "human_dist": vote_counts / matching_n,
            }
        )
    return marginal_group
