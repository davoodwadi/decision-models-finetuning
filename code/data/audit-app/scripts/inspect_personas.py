#!/usr/bin/env python3
"""Inspect respondent personas and answered questions from any dataset's *_long.parquet for the Audit UI.

Usage:
  python inspect_personas.py --dataset-dir /path/to/code/data/kantorowicz2026 --mode summary
  python inspect_personas.py --dataset-dir /path/to/code/data/kantorowicz2026 --mode questions --caseid 1 --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_PATH = Path(__file__).resolve()
DATA_DIR = SCRIPT_PATH.parents[2]
CODE_DIR = DATA_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data.persona import (
    ALL_PERSONA_NAMES,
    GLOBAL_PERSONA_LEVELS,
    PERSONA_NAMES,
)
from data.persona_spec import PersonaAttribute, load_persona_spec

PERSONA_SPECS: dict[str, PersonaAttribute] = load_persona_spec()

EXTRA_ID_CANDIDATES = ("Response.ID", "ResponseId", "CaseID", "participantid", "respondent", "arm")
ISSP_DIR_RE = re.compile(r"^issp(\d{4})$")
WVS7_DIR_NAME = "WVS-Wave7"
HINTS_DIR_NAME = "HINTS"
SHED_DIR_NAME = "SHED"


def _find_long_parquet(dataset_dir: Path) -> Path:
    parquets = sorted(
        list(dataset_dir.glob("*_long.parquet")) + list(dataset_dir.glob("*_marginals.parquet"))
    )
    if not parquets:
        raise FileNotFoundError(
            f"No *_long.parquet or *_marginals.parquet found in {dataset_dir}. "
            "Click 'Parquet -> This dataset' to build it first."
        )
    if len(parquets) > 1:
        raise ValueError(
            f"Expected exactly 1 dataset parquet in {dataset_dir}, found {[p.name for p in parquets]}"
        )
    return parquets[0]


TWIN2K_QID_PREFIX: dict[str, str] = {
    "Twin-2K-500": "twin2k",
    "Twin-2K-500-Mega-Study": "twin2k-mega",
}
SUBPOP_QID_PREFIX: dict[str, str] = {
    "subpop_eval": "subpop_eval",
    "subpop_train": "subpop_train",
}
COUNTRY_COLS = frozenset({"c_alphan", "country"})


def _dataset_loader_spec(
    dataset_dir: Path, schema_cols: list[str]
) -> tuple[str, list[str], str | None, bool]:
    """Return `(loader_kind, canonical_persona_cols, qid_prefix_if_missing, uses_yaml_spec)`."""
    dir_name = dataset_dir.name
    issp_match = ISSP_DIR_RE.match(dir_name)
    spec_cols = [c for c in PERSONA_SPECS if c in schema_cols]

    if issp_match:
        if not spec_cols:
            raise ValueError(
                f"{dataset_dir.name}: parquet uses stale legacy persona columns. "
                "Click 'Parquet -> This dataset' to rebuild with data.issp_export."
            )
        return "persona_spec.yaml", spec_cols, issp_match.group(1), True

    if dir_name == WVS7_DIR_NAME:
        if not spec_cols:
            raise ValueError(
                f"{dataset_dir.name}: parquet uses stale legacy (WorldValuesBench) persona columns. "
                "Click 'Parquet -> This dataset' to rebuild with data.wvs_export."
            )
        return "persona_spec.yaml", spec_cols, None, True

    if dir_name == HINTS_DIR_NAME:
        if not spec_cols:
            raise ValueError(
                f"{dataset_dir.name}: parquet missing canonical persona_spec.yaml columns. "
                "Click 'Parquet -> This dataset' to rebuild with data.hints7_export."
            )
        return "persona_spec.yaml", spec_cols, None, True

    if dir_name == SHED_DIR_NAME:
        if not spec_cols:
            raise ValueError(
                f"{dataset_dir.name}: parquet missing canonical persona_spec.yaml columns. "
                "Click 'Parquet -> This dataset' to rebuild with data.shed2025_export."
            )
        return "persona_spec.yaml", spec_cols, None, True

    if dir_name in TWIN2K_QID_PREFIX:
        if not spec_cols:
            raise ValueError(
                f"{dataset_dir.name}: parquet uses stale legacy persona columns. "
                "Click 'Parquet -> This dataset' to rebuild with data.twin2k_export."
            )
        return "persona_spec.yaml", spec_cols, TWIN2K_QID_PREFIX[dir_name], True

    if dir_name in SUBPOP_QID_PREFIX:
        if not spec_cols:
            raise ValueError(
                f"{dataset_dir.name}: parquet missing canonical persona_spec.yaml columns. "
                "Click 'Parquet -> This dataset' to rebuild with data.subpop_export."
            )
        return "persona_spec.yaml", spec_cols, SUBPOP_QID_PREFIX[dir_name], True

    if spec_cols:
        return "persona_spec.yaml", spec_cols, None, True

    present_persona = [c for c in schema_cols if c in ALL_PERSONA_NAMES]

    ordered = (["c_alphan"] if "c_alphan" in present_persona else []) + [
        c for c in present_persona if c != "c_alphan"
    ]
    return "build_long_records", ordered, None, False


def _clean_scalar(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return None
    if pd.isna(val):
        return None
    return str(val)


def _clean_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [str(x) for x in val]
    raise TypeError(f"Expected sequence column value, got {type(val).__name__}: {val!r}")


def _clean_optional_descriptions(val: Any) -> list[str | None] | None:
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return [None if (x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))) else str(x) for x in val]
    raise TypeError(f"Expected list/None for option_descriptions, got {type(val).__name__}: {val!r}")


def _value_in_spec_levels(spec: PersonaAttribute, val: str) -> bool:
    allowed = set(spec.levels)
    if spec.multi_select:
        return all(part.strip() in allowed for part in val.split(";"))
    return val in allowed


def build_summary(dataset_dir: Path) -> dict[str, Any]:
    parquet_path = _find_long_parquet(dataset_dir)
    pf = pq.ParquetFile(parquet_path)
    schema_cols = pf.schema_arrow.names
    if "CASEID" not in schema_cols:
        raise KeyError(f"{parquet_path.name} is missing required column 'CASEID'")

    loader_kind, persona_cols, _, uses_yaml_spec = _dataset_loader_spec(dataset_dir, schema_cols)
    extra_id_cols = [c for c in EXTRA_ID_CANDIDATES if c in schema_cols]
    q_col = "qid" if "qid" in schema_cols else "v"
    read_cols = ["CASEID", *persona_cols, q_col, *extra_id_cols]

    df = pd.read_parquet(parquet_path, columns=read_cols)
    n_rows = int(len(df))
    n_questions_total = int(df[q_col].nunique(dropna=True))

    # Count answered questions per respondent while preserving first-seen CASEID order.
    q_counts = df.groupby("CASEID", sort=False).size()
    first_rows = df.drop_duplicates(subset=["CASEID"], keep="first").reset_index(drop=True)
    n_respondents = int(len(first_rows))

    # Strict fail-fast check: persona columns must be constant within every CASEID.
    dedup_persona = df.drop_duplicates(subset=["CASEID", *persona_cols])
    if len(dedup_persona) != n_respondents:
        raise ValueError(
            f"{parquet_path.name}: persona values vary within CASEID "
            f"({len(dedup_persona)} unique (CASEID, persona) rows for {n_respondents} respondents)"
        )

    issp_match = ISSP_DIR_RE.match(dataset_dir.name)
    raw_by_caseid: dict[str, dict[str, str | None]] | None = None
    if issp_match:
        from data.persona_loaders import load_raw_issp_personas

        yr = int(issp_match.group(1))
        raw_df = load_raw_issp_personas(DATA_DIR, years=[yr])
        raw_df["_cid"] = raw_df["CASEID"].str.split(":", n=1).str[1]
        raw_by_caseid = {
            str(r["_cid"]): {col: _clean_scalar(r[col]) for col in persona_cols if col in raw_df.columns}
            for r in raw_df.to_dict("records")
        }
    elif dataset_dir.name == WVS7_DIR_NAME:
        from data.persona_loaders import load_raw_wvs7_personas

        raw_df = load_raw_wvs7_personas(DATA_DIR)
        raw_by_caseid = {
            str(r["CASEID"]): {col: _clean_scalar(r[col]) for col in persona_cols if col in raw_df.columns}
            for r in raw_df.to_dict("records")
        }
    elif dataset_dir.name in TWIN2K_QID_PREFIX:
        from data.persona_loaders import load_raw_twin2k_personas

        raw_df = load_raw_twin2k_personas(DATA_DIR)
        raw_by_caseid = {
            f"twin_{r['CASEID']}": {col: _clean_scalar(r[col]) for col in persona_cols if col in raw_df.columns}
            for r in raw_df.to_dict("records")
        }
    elif dataset_dir.name in SUBPOP_QID_PREFIX:
        from data.persona_loaders import load_raw_subpop_personas

        raw_df = load_raw_subpop_personas(DATA_DIR)
        raw_by_caseid = {
            str(r["CASEID"]): {col: _clean_scalar(r[col]) for col in persona_cols if col in raw_df.columns}
            for r in raw_df.to_dict("records")
        }

    is_marginal = "target_probs" in schema_cols

    # Per-attribute summaries in exact canonical order.
    attributes: list[dict[str, Any]] = []
    for col in persona_cols:
        series = first_rows[col]
        present_mask = series.notna()
        n_present = int(present_mask.sum())
        n_missing = n_respondents - n_present
        vc = series.dropna().astype(str).value_counts()

        if uses_yaml_spec:
            spec = PERSONA_SPECS[col]
            prompt_label = spec.prompt
            source_kind = "global"
            levels = [
                {
                    "value": str(val),
                    "count": int(cnt),
                    "pct": round(float(cnt) / n_respondents * 100.0, 2) if n_respondents > 0 else 0.0,
                    "inGlobalLevels": _value_in_spec_levels(spec, str(val)),
                }
                for val, cnt in vc.items()
            ]
        else:
            prompt_label = ALL_PERSONA_NAMES[col]
            source_kind = "global" if col in PERSONA_NAMES else "extra"
            global_levels = GLOBAL_PERSONA_LEVELS.get(col)
            levels = [
                {
                    "value": str(val),
                    "count": int(cnt),
                    "pct": round(float(cnt) / n_respondents * 100.0, 2) if n_respondents > 0 else 0.0,
                    "inGlobalLevels": (str(val) in global_levels) if global_levels is not None else None,
                }
                for val, cnt in vc.items()
            ]

        attributes.append(
            {
                "col": col,
                "promptLabel": prompt_label,
                "source": source_kind,
                "isCountry": col in COUNTRY_COLS,
                "nPresent": n_present,
                "nMissing": n_missing,
                "pctPresent": round(float(n_present) / n_respondents * 100.0, 2) if n_respondents > 0 else 0.0,
                "nLevels": len(levels),
                "levels": levels,
            }
        )

    non_country_cols = [c for c in persona_cols if c not in COUNTRY_COLS]
    present_with_country = first_rows[persona_cols].notna().sum(axis=1)
    present_no_country = (
        first_rows[non_country_cols].notna().sum(axis=1) if non_country_cols else pd.Series(0, index=first_rows.index)
    )

    unique_with_country = int(first_rows[persona_cols].drop_duplicates().shape[0]) if persona_cols else 0
    unique_without_country = int(first_rows[non_country_cols].drop_duplicates().shape[0]) if non_country_cols else 0

    respondents: list[dict[str, Any]] = []
    for idx in range(n_respondents):
        row = first_rows.iloc[idx]
        cid_raw = row["CASEID"]
        cid_str = str(cid_raw)
        extra_ids = {c: _clean_scalar(row[c]) for c in extra_id_cols}
        raw_rec = raw_by_caseid.get(cid_str) if raw_by_caseid is not None else None
        values: dict[str, str | None] = {}
        raw_diffs: dict[str, str | None] = {}
        for col in persona_cols:
            mv = _clean_scalar(row[col])
            values[col] = mv
            if raw_rec is not None and col in raw_rec:
                rv = raw_rec[col]
                if rv != mv:
                    raw_diffs[col] = rv
        entry: dict[str, Any] = {
            "caseId": cid_str,
            "extraIds": extra_ids,
            "nQuestions": int(q_counts.loc[cid_raw]),
            "nPresent": int(present_with_country.iloc[idx]),
            "nPresentNoCountry": int(present_no_country.iloc[idx]),
            "values": values,
        }
        if raw_diffs:
            entry["rawValues"] = raw_diffs
        respondents.append(entry)

    return {
        "parquetFile": parquet_path.name,
        "parquetMtimeMs": int(parquet_path.stat().st_mtime * 1000),
        "loaderKind": loader_kind,
        "isspRuntimeMapped": False,
        "isMarginal": is_marginal,
        "nRespondents": n_respondents,
        "nRows": n_rows,
        "nQuestions": n_questions_total,
        "extraIdCols": extra_id_cols,
        "hasCountry": any(c in COUNTRY_COLS for c in persona_cols),
        "nAttributes": len(persona_cols),
        "uniquePersonasWithCountry": unique_with_country,
        "uniquePersonasWithoutCountry": unique_without_country,
        "avgAttrsPresent": round(float(present_with_country.mean()), 2) if n_respondents > 0 else 0.0,
        "minAttrsPresent": int(present_with_country.min()) if n_respondents > 0 else 0,
        "maxAttrsPresent": int(present_with_country.max()) if n_respondents > 0 else 0,
        "attributes": attributes,
        "respondents": respondents,
    }


def fetch_respondent_questions(dataset_dir: Path, case_id: str, limit: int) -> dict[str, Any]:
    parquet_path = _find_long_parquet(dataset_dir)
    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    schema_cols = schema.names
    _, _, qid_prefix, _ = _dataset_loader_spec(dataset_dir, schema_cols)

    caseid_field = schema.field("CASEID")
    typed_case_id: int | str = int(case_id) if pa.types.is_integer(caseid_field.type) else str(case_id)

    is_marginal = "target_probs" in schema_cols
    wanted = ["CASEID", "v", "question", "type", "option_codes", "option_texts"]
    if is_marginal:
        wanted.append("target_probs")
    else:
        wanted.extend(["answer_code", "answer_text"])
    for opt_col in ("qid", "label", "option_descriptions", "arm"):
        if opt_col in schema_cols:
            wanted.append(opt_col)

    sub = pd.read_parquet(parquet_path, columns=wanted, filters=[("CASEID", "==", typed_case_id)])
    total_questions = int(len(sub))
    if total_questions == 0:
        raise KeyError(f"CASEID={case_id!r} not found in {parquet_path.name}")

    has_qid = "qid" in sub.columns
    has_label = "label" in sub.columns
    has_descs = "option_descriptions" in sub.columns
    has_arm = "arm" in sub.columns

    questions: list[dict[str, Any]] = []
    for row in sub.iloc[:limit].to_dict("records"):
        v_val = str(row["v"])
        qid_val = str(row["qid"]) if has_qid else (f"{qid_prefix}:{v_val}" if qid_prefix else v_val)
        opt_codes = [int(c) for c in row["option_codes"]]
        opt_texts = _clean_list(row["option_texts"])
        opt_descs = _clean_optional_descriptions(row["option_descriptions"]) if has_descs else None
        if len(opt_codes) != len(opt_texts):
            raise ValueError(f"{qid_val}: option_codes ({len(opt_codes)}) != option_texts ({len(opt_texts)})")

        if is_marginal:
            probs = [round(float(p), 6) for p in row["target_probs"]]
            if len(probs) != len(opt_codes):
                raise ValueError(f"{qid_val}: target_probs ({len(probs)}) != option_codes ({len(opt_codes)})")
            ans_idx = int(np.argmax(probs))
            ans_code = opt_codes[ans_idx]
            ans_text = opt_texts[ans_idx]
        else:
            probs = None
            ans_code = int(row["answer_code"])
            ans_idx = opt_codes.index(ans_code)
            ans_text = str(row["answer_text"])

        q_item: dict[str, Any] = {
            "qid": qid_val,
            "v": v_val,
            "label": _clean_scalar(row["label"]) if has_label else None,
            "type": str(row["type"]),
            "question": str(row["question"]),
            "optionCodes": opt_codes,
            "optionTexts": opt_texts,
            "optionDescriptions": opt_descs,
            "answerCode": ans_code,
            "answerText": ans_text,
            "answerIdx": ans_idx,
            "arm": _clean_scalar(row["arm"]) if has_arm else None,
        }
        if probs is not None:
            q_item["targetProbs"] = probs
        questions.append(q_item)

    return {
        "caseId": str(case_id),
        "totalQuestions": total_questions,
        "questions": questions,
    }


PREFERRED_MARGINAL_PERSONAS = (
    "political_party",
    "US_PRTY",
    "political_ideology",
    "PARTY_LR",
    "gender",
    "SEX",
    "age",
    "AGE",
    "education",
    "EDULEVEL",
    "race",
    "RACE",
    "region",
    "US_REG",
    "religion",
    "RELIGGRP",
)


def _match_question_vs(
    all_vs: list[str],
    question_id: str,
    subitem_id: str | None,
    study: str | None,
) -> list[str]:
    v_set = set(all_vs)
    slug = study.replace("_", "-") if study else None

    if subitem_id:
        for cand in [subitem_id, f"{slug}-{subitem_id}" if slug else None]:
            if cand and cand in v_set:
                return [cand]
        if slug:
            pref = f"{slug}-{subitem_id}_"
            hits = [v for v in all_vs if v.startswith(pref)]
            if hits:
                return hits
        pref = f"{subitem_id}_"
        hits = [v for v in all_vs if v.startswith(pref)]
        if hits:
            return hits

    for cand in [question_id, f"{slug}-{question_id}" if slug else None]:
        if cand and cand in v_set:
            return [cand]

    if slug:
        pref = f"{slug}-{question_id}_"
        hits = [v for v in all_vs if v.startswith(pref)]
        if hits:
            return hits

    pref = f"{question_id}_"
    hits = [v for v in all_vs if v.startswith(pref)]
    if hits:
        return hits
    return []


def fetch_question_marginals(
    dataset_dir: Path,
    question_id: str,
    subitem_id: str | None = None,
    study: str | None = None,
    persona: str | None = None,
    arm: str | None = None,
) -> dict[str, Any]:
    parquet_path = _find_long_parquet(dataset_dir)
    pf = pq.ParquetFile(parquet_path)
    schema_cols = pf.schema_arrow.names
    loader_kind, persona_cols, _, uses_yaml_spec = _dataset_loader_spec(dataset_dir, schema_cols)

    # Discover selectable persona columns (exclude constant country columns like US-only).
    selectable_cols = [c for c in persona_cols if c not in ("country", "c_alphan")]
    if not selectable_cols:
        selectable_cols = list(persona_cols)

    # Read respondent-level unique values once to report available personas + level counts.
    resp_df = pd.read_parquet(parquet_path, columns=["CASEID", *selectable_cols]).drop_duplicates(subset=["CASEID"])
    available_personas: list[dict[str, Any]] = []
    for col in selectable_cols:
        s = resp_df[col].dropna()
        n_levels = int(s.nunique())
        if n_levels < 2:
            continue
        prompt_label = PERSONA_SPECS[col].prompt if uses_yaml_spec else ALL_PERSONA_NAMES[col]
        available_personas.append(
            {
                "col": col,
                "promptLabel": prompt_label,
                "nLevels": n_levels,
                "nPresent": int(len(s)),
            }
        )

    avail_col_set = {a["col"] for a in available_personas}
    if persona and persona in avail_col_set:
        selected_col = persona
    else:
        selected_col = next(
            (c for c in PREFERRED_MARGINAL_PERSONAS if c in avail_col_set),
            available_personas[0]["col"] if available_personas else None,
        )

    all_vs = [str(v) for v in pd.read_parquet(parquet_path, columns=["v"])["v"].drop_duplicates().tolist()]
    sibling_vs = _match_question_vs(all_vs, question_id, None, study)
    matched_vs = _match_question_vs(all_vs, question_id, subitem_id, study) if subitem_id else sibling_vs

    if not selected_col:
        return {
            "found": False,
            "reason": "No multi-level persona attributes available in this parquet.",
            "parquetFile": parquet_path.name,
            "loaderKind": loader_kind,
            "availablePersonas": available_personas,
            "selectedPersona": None,
            "matchedVs": matched_vs,
            "availableArms": [],
            "selectedArm": None,
        }

    selected_meta = next(a for a in available_personas if a["col"] == selected_col)

    if not matched_vs:
        return {
            "found": False,
            "reason": f"Question {subitem_id or question_id!r} is not present in {parquet_path.name} (excluded or not yet rebuilt).",
            "parquetFile": parquet_path.name,
            "loaderKind": loader_kind,
            "availablePersonas": available_personas,
            "selectedPersona": selected_meta,
            "matchedVs": [],
            "availableArms": [],
            "selectedArm": None,
        }

    # For matrix questions with multiple sub-items when no specific subitem was passed, use the first v so option_texts are uniform.
    target_vs = matched_vs[:1] if len(matched_vs) > 1 and subitem_id is None else matched_vs
    wanted = ["v", "type", "option_codes", "option_texts", "answer_code", "answer_text", selected_col]
    has_reversed = "reversed" in schema_cols
    has_arm = "arm" in schema_cols
    if has_reversed:
        wanted.append("reversed")
    if has_arm:
        wanted.append("arm")

    sub = pd.read_parquet(parquet_path, columns=wanted, filters=[("v", "in", target_vs)])
    if len(sub) == 0:
        return {
            "found": False,
            "reason": f"No valid respondent answers found for {target_vs[0]!r} in {parquet_path.name}.",
            "parquetFile": parquet_path.name,
            "loaderKind": loader_kind,
            "availablePersonas": available_personas,
            "selectedPersona": selected_meta,
            "matchedVs": matched_vs,
            "availableArms": [],
            "selectedArm": None,
        }

    available_arms: list[str] = []
    selected_arm: str | None = None
    if has_arm:
        available_arms = sorted(str(x) for x in sub["arm"].dropna().unique().tolist())
        if arm and arm in available_arms:
            sub = sub[sub["arm"].astype(str) == arm]
            selected_arm = arm

    q_type = str(sub["type"].iloc[0])
    is_reversed = bool(sub["reversed"].iloc[0]) if has_reversed and pd.notna(sub["reversed"].iloc[0]) else False
    opt_codes = [int(c) for c in sub["option_codes"].iloc[0]]
    opt_texts = _clean_list(sub["option_texts"].iloc[0])
    if len(opt_codes) != len(opt_texts):
        raise ValueError(f"{target_vs[0]}: option_codes ({len(opt_codes)}) != option_texts ({len(opt_texts)})")

    code_to_idx = {c: i for i, c in enumerate(opt_codes)}
    n_opts = len(opt_codes)

    overall_counts = [0] * n_opts
    group_counts: dict[str, list[int]] = {}
    missing_label = "__MISSING__"

    for ans_code, p_val in zip(sub["answer_code"], sub[selected_col]):
        c_int = int(ans_code)
        if c_int not in code_to_idx:
            continue
        oi = code_to_idx[c_int]
        overall_counts[oi] += 1
        g = _clean_scalar(p_val)
        g_key = g if g is not None else missing_label
        if g_key not in group_counts:
            group_counts[g_key] = [0] * n_opts
        group_counts[g_key][oi] += 1

    overall_n = sum(overall_counts)
    options_payload = [
        {
            "code": opt_codes[i],
            "text": opt_texts[i],
            "overallCount": overall_counts[i],
            "overallPct": round(overall_counts[i] / overall_n * 100.0, 2) if overall_n > 0 else 0.0,
        }
        for i in range(n_opts)
    ]

    # Order groups by persona_spec.yaml canonical level order when available, else descending N; __MISSING__ last.
    non_missing_keys = [k for k in group_counts if k != missing_label]
    if uses_yaml_spec and selected_col in PERSONA_SPECS and not PERSONA_SPECS[selected_col].multi_select:
        spec_order = {lvl: idx for idx, lvl in enumerate(PERSONA_SPECS[selected_col].levels)}
        non_missing_keys.sort(key=lambda k: (spec_order.get(k, 10_000), -sum(group_counts[k]), k))
    else:
        non_missing_keys.sort(key=lambda k: (-sum(group_counts[k]), k))

    ordered_keys = non_missing_keys + ([missing_label] if missing_label in group_counts else [])

    groups_payload: list[dict[str, Any]] = []
    for k in ordered_keys:
        cnts = group_counts[k]
        gn = sum(cnts)
        shares = [round(c / gn * 100.0, 2) if gn > 0 else 0.0 for c in cnts]
        mean_score: float | None = None
        if q_type == "score" and gn > 0:
            mean_score = round(sum((i + 1) * cnts[i] for i in range(n_opts)) / gn, 3)
        groups_payload.append(
            {
                "level": "Missing (NA)" if k == missing_label else k,
                "isMissing": k == missing_label,
                "n": gn,
                "counts": cnts,
                "shares": shares,
                "meanScore": mean_score,
            }
        )

    # Compute separation statistics across non-missing groups with n >= 15 (fallback to all non-missing if <2 qualify).
    eligible = [g for g in groups_payload if not g["isMissing"] and g["n"] >= 15]
    if len(eligible) < 2:
        eligible = [g for g in groups_payload if not g["isMissing"] and g["n"] > 0]

    max_tvd_pp = 0.0
    tvd_pair: list[str] | None = None
    max_opt_diff: dict[str, Any] | None = None
    max_mean_diff: dict[str, Any] | None = None

    if len(eligible) >= 2:
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                ga, gb = eligible[i], eligible[j]
                tvd = 0.5 * sum(abs(a - b) for a, b in zip(ga["shares"], gb["shares"]))
                if tvd > max_tvd_pp:
                    max_tvd_pp = round(tvd, 2)
                    tvd_pair = [ga["level"], gb["level"]]
                if ga["meanScore"] is not None and gb["meanScore"] is not None:
                    md = abs(ga["meanScore"] - gb["meanScore"])
                    if max_mean_diff is None or md > max_mean_diff["delta"]:
                        hi, lo = (ga, gb) if ga["meanScore"] >= gb["meanScore"] else (gb, ga)
                        max_mean_diff = {
                            "delta": round(md, 3),
                            "highGroup": hi["level"],
                            "highMean": hi["meanScore"],
                            "lowGroup": lo["level"],
                            "lowMean": lo["meanScore"],
                        }

        for oi in range(n_opts):
            hi = max(eligible, key=lambda g: g["shares"][oi])
            lo = min(eligible, key=lambda g: g["shares"][oi])
            delta = round(hi["shares"][oi] - lo["shares"][oi], 2)
            if max_opt_diff is None or delta > max_opt_diff["deltaPct"]:
                max_opt_diff = {
                    "option": opt_texts[oi],
                    "deltaPct": delta,
                    "highGroup": hi["level"],
                    "highPct": hi["shares"][oi],
                    "lowGroup": lo["level"],
                    "lowPct": lo["shares"][oi],
                }

    overall_mean: float | None = None
    if q_type == "score" and overall_n > 0:
        overall_mean = round(sum((i + 1) * overall_counts[i] for i in range(n_opts)) / overall_n, 3)

    return {
        "found": True,
        "parquetFile": parquet_path.name,
        "loaderKind": loader_kind,
        "availablePersonas": available_personas,
        "selectedPersona": selected_meta,
        "matchedVs": sibling_vs if sibling_vs else matched_vs,
        "activeV": target_vs[0],
        "availableArms": available_arms,
        "selectedArm": selected_arm,
        "questionType": q_type,
        "reversed": is_reversed,
        "overallN": overall_n,
        "overallMean": overall_mean,
        "options": options_payload,
        "groups": groups_payload,
        "separation": {
            "minGroupN": 15,
            "eligibleGroupCount": len(eligible),
            "maxTvdPct": max_tvd_pp,
            "tvdPair": tvd_pair,
            "maxOptionDiff": max_opt_diff,
            "maxMeanDiff": max_mean_diff,
        },
    }


def fetch_persona_marginals_bundle(
    dataset_dir: Path,
    persona: str | None = None,
) -> dict[str, Any]:
    """Compute counts for ALL questions in a dataset for one persona in a single vectorized pass."""
    parquet_path = _find_long_parquet(dataset_dir)
    pf = pq.ParquetFile(parquet_path)
    schema_cols = pf.schema_arrow.names
    loader_kind, persona_cols, _, uses_yaml_spec = _dataset_loader_spec(dataset_dir, schema_cols)

    selectable_cols = [c for c in persona_cols if c not in ("country", "c_alphan")]
    if not selectable_cols:
        selectable_cols = list(persona_cols)

    # Read respondent-level metadata for availablePersonas in one narrow scan.
    resp_df = pd.read_parquet(parquet_path, columns=["CASEID", *selectable_cols]).drop_duplicates(subset=["CASEID"])
    available_personas: list[dict[str, Any]] = []
    for col in selectable_cols:
        s = resp_df[col].dropna()
        n_levels = int(s.nunique())
        if n_levels < 2:
            continue
        prompt_label = PERSONA_SPECS[col].prompt if uses_yaml_spec else ALL_PERSONA_NAMES[col]
        available_personas.append(
            {
                "col": col,
                "promptLabel": prompt_label,
                "nLevels": n_levels,
                "nPresent": int(len(s)),
            }
        )

    avail_col_set = {a["col"] for a in available_personas}
    if persona and persona in avail_col_set:
        selected_col = persona
    else:
        selected_col = next(
            (c for c in PREFERRED_MARGINAL_PERSONAS if c in avail_col_set),
            available_personas[0]["col"] if available_personas else None,
        )

    if not selected_col:
        return {
            "parquetFile": parquet_path.name,
            "loaderKind": loader_kind,
            "availablePersonas": available_personas,
            "selectedPersona": None,
            "canonicalLevelOrder": None,
            "allVs": [],
            "questions": {},
        }

    selected_meta = next(a for a in available_personas if a["col"] == selected_col)
    canonical_order: list[str] | None = None
    if uses_yaml_spec and selected_col in PERSONA_SPECS and not PERSONA_SPECS[selected_col].multi_select:
        canonical_order = list(PERSONA_SPECS[selected_col].levels)

    has_reversed = "reversed" in schema_cols
    has_arm = "arm" in schema_cols

    # Read question metadata (1 row per v) + narrow columns for vectorized groupby.
    meta_wanted = ["v", "type", "option_codes", "option_texts"]
    if has_reversed:
        meta_wanted.append("reversed")
    meta_df = pd.read_parquet(parquet_path, columns=meta_wanted).drop_duplicates(subset=["v"], keep="first")

    all_vs: list[str] = []
    q_meta: dict[str, dict[str, Any]] = {}
    code_to_idx_by_v: dict[str, dict[int, int]] = {}
    for row in meta_df.to_dict("records"):
        v_str = str(row["v"])
        opt_codes = [int(c) for c in row["option_codes"]]
        opt_texts = _clean_list(row["option_texts"])
        if len(opt_codes) != len(opt_texts):
            raise ValueError(f"{v_str}: option_codes ({len(opt_codes)}) != option_texts ({len(opt_texts)})")
        is_rev = bool(row["reversed"]) if has_reversed and pd.notna(row["reversed"]) else False
        all_vs.append(v_str)
        code_to_idx_by_v[v_str] = {c: i for i, c in enumerate(opt_codes)}
        q_meta[v_str] = {
            "questionType": str(row["type"]),
            "reversed": is_rev,
            "optionCodes": opt_codes,
            "optionTexts": opt_texts,
            "arms": [],
            "byArm": {},
        }

    is_marginal = "target_probs" in schema_cols

    if is_marginal:
        # Marginal parquet: each row already holds the exact probability vector for one (v, subgroup).
        marg_cols = ["v", "attribute", selected_col, "target_probs"]
        marg_df = pd.read_parquet(parquet_path, columns=marg_cols)
        marg_df = marg_df[marg_df["attribute"] == selected_col]
        for row in marg_df.to_dict("records"):
            v_str = str(row["v"])
            if v_str not in q_meta:
                continue
            p_val = _clean_scalar(row[selected_col])
            if p_val is None:
                raise ValueError(f"{v_str}: missing {selected_col} on row with attribute={selected_col!r}")
            probs = [float(p) for p in row["target_probs"]]
            if len(probs) != len(q_meta[v_str]["optionCodes"]):
                raise ValueError(
                    f"{v_str} ({p_val}): target_probs len {len(probs)} != optionCodes len {len(q_meta[v_str]['optionCodes'])}"
                )
            all_bucket = q_meta[v_str]["byArm"].setdefault("__all__", {})
            all_bucket[p_val] = probs

        return {
            "parquetFile": parquet_path.name,
            "loaderKind": loader_kind,
            "isMarginal": True,
            "availablePersonas": available_personas,
            "selectedPersona": selected_meta,
            "canonicalLevelOrder": canonical_order,
            "allVs": all_vs,
            "questions": q_meta,
        }

    # Narrow read for vectorized contingency counts.
    cnt_wanted = ["v", "answer_code", selected_col]
    if has_arm:
        cnt_wanted.append("arm")
    cnt_df = pd.read_parquet(parquet_path, columns=cnt_wanted)
    missing_label = "__MISSING__"
    p_ser = cnt_df[selected_col]
    cnt_df["_p"] = (
        p_ser.astype("string")
        .str.strip()
        .replace({"": missing_label, "nan": missing_label, "None": missing_label, "<NA>": missing_label})
        .fillna(missing_label)
    )

    if has_arm:
        a_ser = cnt_df["arm"]
        cnt_df["_arm"] = (
            a_ser.astype("string")
            .str.strip()
            .replace({"": "__NO_ARM__", "nan": "__NO_ARM__", "None": "__NO_ARM__", "<NA>": "__NO_ARM__"})
            .fillna("__NO_ARM__")
        )
        grouped = cnt_df.groupby(["v", "_arm", "_p", "answer_code"], observed=True).size()
        arms_by_v: dict[str, set[str]] = {}
        for (v_val, arm_val, p_val, ans_code), cnt in grouped.items():
            v_str = str(v_val)
            c_map = code_to_idx_by_v.get(v_str)
            if c_map is None:
                continue
            oi = c_map.get(int(ans_code))
            if oi is None:
                continue
            n_opts = len(c_map)
            by_arm = q_meta[v_str]["byArm"]
            cnt_int = int(cnt)

            # Pooled __all__
            all_bucket = by_arm.setdefault("__all__", {})
            if p_val not in all_bucket:
                all_bucket[p_val] = [0] * n_opts
            all_bucket[p_val][oi] += cnt_int

            # Specific arm
            if arm_val != "__NO_ARM__":
                arms_by_v.setdefault(v_str, set()).add(str(arm_val))
                arm_bucket = by_arm.setdefault(str(arm_val), {})
                if p_val not in arm_bucket:
                    arm_bucket[p_val] = [0] * n_opts
                arm_bucket[p_val][oi] += cnt_int

        for v_str, a_set in arms_by_v.items():
            q_meta[v_str]["arms"] = sorted(a_set)
    else:
        grouped = cnt_df.groupby(["v", "_p", "answer_code"], observed=True).size()
        for (v_val, p_val, ans_code), cnt in grouped.items():
            v_str = str(v_val)
            c_map = code_to_idx_by_v.get(v_str)
            if c_map is None:
                continue
            oi = c_map.get(int(ans_code))
            if oi is None:
                continue
            n_opts = len(c_map)
            all_bucket = q_meta[v_str]["byArm"].setdefault("__all__", {})
            if p_val not in all_bucket:
                all_bucket[p_val] = [0] * n_opts
            all_bucket[p_val][oi] += int(cnt)

    return {
        "parquetFile": parquet_path.name,
        "loaderKind": loader_kind,
        "isMarginal": False,
        "availablePersonas": available_personas,
        "selectedPersona": selected_meta,
        "canonicalLevelOrder": canonical_order,
        "allVs": all_vs,
        "questions": q_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect personas and KEV prompts from a dataset parquet.")
    parser.add_argument("--dataset-dir", required=True, help="Absolute path to dataset folder containing *_long.parquet")
    parser.add_argument("--mode", required=True, choices=["summary", "questions", "marginals", "marginals-bundle"])
    parser.add_argument("--caseid", help="Respondent CASEID (required for --mode questions)")
    parser.add_argument("--limit", type=int, default=50, help="Max answered questions to return (default 50)")
    parser.add_argument("--question-id", help="QuestionID / v (required for --mode marginals)")
    parser.add_argument("--subitem-id", help="Optional matrix sub-item ID for --mode marginals")
    parser.add_argument("--study", help="Optional Mega-Study study name for --mode marginals")
    parser.add_argument("--persona", help="Optional persona column name for --mode marginals / marginals-bundle")
    parser.add_argument("--arm", help="Optional treatment arm filter for --mode marginals")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if args.mode == "summary":
        out = build_summary(dataset_dir)
    elif args.mode == "questions":
        if not args.caseid:
            raise ValueError("--caseid is required when --mode questions")
        out = fetch_respondent_questions(dataset_dir, args.caseid, args.limit)
    elif args.mode == "marginals-bundle":
        out = fetch_persona_marginals_bundle(dataset_dir, persona=args.persona or None)
    else:
        if not args.question_id:
            raise ValueError("--question-id is required when --mode marginals")
        out = fetch_question_marginals(
            dataset_dir,
            question_id=args.question_id,
            subitem_id=args.subitem_id or None,
            study=args.study or None,
            persona=args.persona or None,
            arm=args.arm or None,
        )

    json.dump(out, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
