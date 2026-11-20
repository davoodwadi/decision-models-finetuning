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
    apply_persona_mapping,
)

EXTRA_ID_CANDIDATES = ("Response.ID", "ResponseId", "CaseID", "participantid", "respondent", "arm")
ISSP_DIR_RE = re.compile(r"^issp(\d{4})$")


def _find_long_parquet(dataset_dir: Path) -> Path:
    parquets = sorted(dataset_dir.glob("*_long.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No *_long.parquet found in {dataset_dir}. Click 'Parquet -> This dataset' to build it first."
        )
    if len(parquets) > 1:
        raise ValueError(f"Expected exactly 1 *_long.parquet in {dataset_dir}, found {[p.name for p in parquets]}")
    return parquets[0]


def _dataset_loader_spec(dataset_dir: Path, schema_cols: list[str]) -> tuple[str, list[str], str | None]:
    """Return `(loader_kind, canonical_persona_cols, qid_prefix_if_missing)` matching the real Python loader."""
    dir_name = dataset_dir.name
    issp_match = ISSP_DIR_RE.match(dir_name)
    present_persona = [c for c in schema_cols if c in ALL_PERSONA_NAMES]

    if issp_match:
        # `load_issp_eval` -> `build_persona_dataset` iterates `ALL_PERSONA_NAMES` order after `apply_persona_mapping`.
        ordered = [c for c in ALL_PERSONA_NAMES if c in present_persona]
        return "build_persona_dataset", ordered, issp_match.group(1)

    if dir_name == "Twin-2K-500":
        ordered = [c for c in ALL_PERSONA_NAMES if c in present_persona]
        return "build_persona_dataset", ordered, "twin2k"

    if dir_name == "Twin-2K-500-Mega-Study":
        ordered = [c for c in ALL_PERSONA_NAMES if c in present_persona]
        return "build_persona_dataset", ordered, "twin2k-mega"

    # `build_long_records` puts `c_alphan` first (when `include_country=True`) followed by dataset `*_PERSONA_COLS`,
    # which matches the exact column order written into `*_long.parquet`.
    ordered = (["c_alphan"] if "c_alphan" in present_persona else []) + [
        c for c in present_persona if c != "c_alphan"
    ]
    return "build_long_records", ordered, None


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


def build_summary(dataset_dir: Path) -> dict[str, Any]:
    parquet_path = _find_long_parquet(dataset_dir)
    pf = pq.ParquetFile(parquet_path)
    schema_cols = pf.schema_arrow.names
    if "CASEID" not in schema_cols:
        raise KeyError(f"{parquet_path.name} is missing required column 'CASEID'")

    loader_kind, persona_cols, _ = _dataset_loader_spec(dataset_dir, schema_cols)
    extra_id_cols = [c for c in EXTRA_ID_CANDIDATES if c in schema_cols]
    q_col = "qid" if "qid" in schema_cols else "v"
    # Keep persona columns immediately after CASEID and before `v` so `apply_persona_mapping(..., dataset="issp")` matches.
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

    is_issp = ISSP_DIR_RE.match(dataset_dir.name) is not None
    raw_first = first_rows.copy()
    if is_issp:
        mapped_first = apply_persona_mapping(first_rows.copy(), dataset="issp")
    else:
        mapped_first = first_rows

    # Per-attribute summaries in exact canonical KEV order.
    attributes: list[dict[str, Any]] = []
    for col in persona_cols:
        series = mapped_first[col]
        present_mask = series.notna()
        n_present = int(present_mask.sum())
        n_missing = n_respondents - n_present
        vc = series.dropna().astype(str).value_counts()
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
                "promptLabel": ALL_PERSONA_NAMES[col],
                "source": "global" if col in PERSONA_NAMES else "extra",
                "isCountry": col == "c_alphan",
                "nPresent": n_present,
                "nMissing": n_missing,
                "pctPresent": round(float(n_present) / n_respondents * 100.0, 2) if n_respondents > 0 else 0.0,
                "nLevels": len(levels),
                "levels": levels,
            }
        )

    non_country_cols = [c for c in persona_cols if c != "c_alphan"]
    present_with_country = mapped_first[persona_cols].notna().sum(axis=1)
    present_no_country = (
        mapped_first[non_country_cols].notna().sum(axis=1) if non_country_cols else pd.Series(0, index=mapped_first.index)
    )

    unique_with_country = int(mapped_first[persona_cols].drop_duplicates().shape[0]) if persona_cols else 0
    unique_without_country = int(mapped_first[non_country_cols].drop_duplicates().shape[0]) if non_country_cols else 0

    respondents: list[dict[str, Any]] = []
    for idx in range(n_respondents):
        row = mapped_first.iloc[idx]
        raw_row = raw_first.iloc[idx]
        cid_raw = row["CASEID"]
        cid_str = str(cid_raw)
        extra_ids = {c: _clean_scalar(row[c]) for c in extra_id_cols}
        values: dict[str, str | None] = {}
        raw_diffs: dict[str, str | None] = {}
        for col in persona_cols:
            mv = _clean_scalar(row[col])
            values[col] = mv
            if is_issp:
                rv = _clean_scalar(raw_row[col])
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
        "isspRuntimeMapped": is_issp,
        "nRespondents": n_respondents,
        "nRows": n_rows,
        "nQuestions": n_questions_total,
        "extraIdCols": extra_id_cols,
        "hasCountry": "c_alphan" in persona_cols,
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
    _, _, qid_prefix = _dataset_loader_spec(dataset_dir, schema_cols)

    caseid_field = schema.field("CASEID")
    typed_case_id: int | str = int(case_id) if pa.types.is_integer(caseid_field.type) else str(case_id)

    wanted = ["CASEID", "v", "question", "type", "option_codes", "option_texts", "answer_code", "answer_text"]
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
        ans_code = int(row["answer_code"])
        if len(opt_codes) != len(opt_texts):
            raise ValueError(f"{qid_val}: option_codes ({len(opt_codes)}) != option_texts ({len(opt_texts)})")
        ans_idx = opt_codes.index(ans_code)
        questions.append(
            {
                "qid": qid_val,
                "v": v_val,
                "label": _clean_scalar(row["label"]) if has_label else None,
                "type": str(row["type"]),
                "question": str(row["question"]),
                "optionCodes": opt_codes,
                "optionTexts": opt_texts,
                "optionDescriptions": opt_descs,
                "answerCode": ans_code,
                "answerText": str(row["answer_text"]),
                "answerIdx": ans_idx,
                "arm": _clean_scalar(row["arm"]) if has_arm else None,
            }
        )

    return {
        "caseId": str(case_id),
        "totalQuestions": total_questions,
        "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect personas and KEV prompts from a dataset parquet.")
    parser.add_argument("--dataset-dir", required=True, help="Absolute path to dataset folder containing *_long.parquet")
    parser.add_argument("--mode", required=True, choices=["summary", "questions"])
    parser.add_argument("--caseid", help="Respondent CASEID (required for --mode questions)")
    parser.add_argument("--limit", type=int, default=50, help="Max answered questions to return (default 50)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if args.mode == "summary":
        out = build_summary(dataset_dir)
    else:
        if not args.caseid:
            raise ValueError("--caseid is required for --mode questions")
        out = fetch_respondent_questions(dataset_dir, args.caseid, args.limit)

    json.dump(out, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
