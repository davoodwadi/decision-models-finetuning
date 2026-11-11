"""Fast boolean-mask empirical k-marginal lookup and epoch group augmentation over survey personas."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.base import ALL_PERSONA_NAMES, group_train_records
from data.persona import PERSONA_NAMES

DEFAULT_K_SCHEDULE = (1, 2, 3, 4, 5, "full", "full")


def validate_k_schedule(k_schedule):
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
class PersonaMarginalLookup:
    """Precomputed `(trait, value)` boolean masks and per-question unique-persona vote arrays."""

    persona_cols: tuple[str, ...]
    include_country: bool
    num_unique_personas: int
    unique_personas_df: pd.DataFrame
    trait_masks: dict[tuple[str, object], np.ndarray]
    qid_persona_ids: dict[str, np.ndarray]
    qid_vote_counts: dict[str, np.ndarray]


def build_marginal_lookup(train_df, *, include_country):
    """Index unique full personas in `train_df` into boolean `(trait, value)` masks and per-qid vote arrays."""
    if len(train_df) == 0:
        raise ValueError("train_df must not be empty")
    for req_col in ("qid", "one_hot"):
        if req_col not in train_df.columns:
            raise KeyError(f"Expected column {req_col!r} in train_df")

    persona_cols = tuple(c for c in PERSONA_NAMES if include_country or c != "c_alphan")
    missing_cols = [c for c in persona_cols if c not in train_df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected persona columns in train_df: {missing_cols}")

    df = train_df.reset_index(drop=True)
    persona_groups = df.groupby(list(persona_cols), dropna=False, sort=False)
    train_persona_id = persona_groups.ngroup().to_numpy()
    unique_personas_df = persona_groups.size().rename("obs_count").reset_index()
    num_unique_personas = len(unique_personas_df)

    trait_masks = {}
    for p_id, p_row in enumerate(unique_personas_df[list(persona_cols)].to_dict("records")):
        for col, val in p_row.items():
            if pd.notna(val):
                if (col, val) not in trait_masks:
                    trait_masks[(col, val)] = np.zeros(num_unique_personas, dtype=bool)
                trait_masks[(col, val)][p_id] = True

    qid_persona_ids = {}
    qid_vote_counts = {}
    for qid, q_sub_df in df.groupby("qid", sort=False):
        p_ids = train_persona_id[q_sub_df.index.to_numpy()]
        one_hots = np.stack(q_sub_df["one_hot"].to_numpy())
        if not np.allclose(one_hots.sum(axis=1), 1.0):
            raise ValueError(f"Expected all one_hot rows for qid={qid!r} to sum to 1.0")
        uniq_p_ids, inverse_idx = np.unique(p_ids, return_inverse=True)
        summed_votes = np.zeros((len(uniq_p_ids), one_hots.shape[1]), dtype=one_hots.dtype)
        np.add.at(summed_votes, inverse_idx, one_hots)
        qid_persona_ids[qid] = uniq_p_ids
        qid_vote_counts[qid] = summed_votes

    return PersonaMarginalLookup(
        persona_cols=persona_cols,
        include_country=bool(include_country),
        num_unique_personas=num_unique_personas,
        unique_personas_df=unique_personas_df,
        trait_masks=trait_masks,
        qid_persona_ids=qid_persona_ids,
        qid_vote_counts=qid_vote_counts,
    )


def augment_group_with_marginals(base_group, subpersona, lookup, *, persona_names=ALL_PERSONA_NAMES):
    """Replace `personas`, `n`, and `human_dist` in `base_group` with the exact empirical marginal for `subpersona`."""
    if not base_group:
        raise ValueError("base_group must not be empty")
    if not subpersona:
        raise ValueError("subpersona must not be empty")

    matching_mask = np.ones(lookup.num_unique_personas, dtype=bool)
    for col, val in subpersona.items():
        matching_mask &= lookup.trait_masks[(col, val)]

    attr_str = "+".join(subpersona.keys())
    persona_str = ", ".join(f"{persona_names[c]}={v}" for c, v in subpersona.items())

    marginal_group = []
    for rec in base_group:
        qid = rec["qid"]
        matched_entries = matching_mask[lookup.qid_persona_ids[qid]]
        vote_counts = lookup.qid_vote_counts[qid][matched_entries].sum(axis=0)
        matching_n = int(round(float(vote_counts.sum())))
        if matching_n < 1:
            raise ValueError(
                f"Expected matching_n >= 1 for qid={qid!r} and subpersona={subpersona!r}, got {matching_n}"
            )
        marginal_group.append({
            **rec,
            "personas": subpersona,
            "attribute": attr_str,
            "persona": persona_str,
            "n": matching_n,
            "human_dist": vote_counts / matching_n,
        })
    return marginal_group


def build_marginal_epoch_groups(
    train_records,
    lookup,
    *,
    max_questions,
    max_chars,
    rng,
    k_schedule=DEFAULT_K_SCHEDULE,
):
    """Build 1 epoch of preserved unique-persona question groups augmented with stateless `k_schedule` marginals."""
    if rng is None:
        raise ValueError("rng is required for build_marginal_epoch_groups")
    schedule = validate_k_schedule(k_schedule)
    epoch_base_groups = group_train_records(
        train_records,
        max_questions=max_questions,
        max_chars=max_chars,
        rng=rng,
    )

    marginal_groups = []
    for base_group in epoch_base_groups:
        full_persona = base_group[0]["personas"]
        if not full_persona:
            raise ValueError(f"Expected non-empty personas dict in base_group[0] for qid={base_group[0]['qid']!r}")
        k_choice = rng.choice(schedule)
        k_traits = len(full_persona) if k_choice == "full" else min(k_choice, len(full_persona))
        chosen_traits = set(rng.sample(list(full_persona.keys()), k_traits))
        subpersona = {col: val for col, val in full_persona.items() if col in chosen_traits}
        marginal_groups.append(augment_group_with_marginals(base_group, subpersona, lookup))
    return marginal_groups
