"""Unified fail-fast audit bootstrapper and marginal parquet exporter for official SubPOP splits.

Reads the two official upstream `jjssuh/subpop` splits:
  - `code/data/subpop_eval/raw_subpop_eval.jsonl`  (GSS 2022, 133 questions x 22 subgroups)
  - `code/data/subpop_train/raw_subpop_train.jsonl` (Pew ATP, 3,229 questions x ~22 subgroups)

Joins them with:
  - `question_audit.csv` and optional `question_overrides.json`
  - `persona_loaders.load_raw_subpop_personas()` + `persona_spec.yaml`

Writes explicit marginal-format parquets (`subpop_{split}_US_marginals.parquet`) rather than
synthetic respondent microdata so downstream evaluation and the Audit UI preserve exact
aggregate subgroup probability distributions.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from data.persona_loaders import SUBPOP_ATTR_TO_COL, load_raw_subpop_personas
from data.persona_spec import load_persona_spec, standardize_dataframe

DATA_DIR = Path(__file__).resolve().parent
SUBPOP_SPLITS: tuple[str, ...] = ("eval", "train")
SUBPOP_DIR_TO_SPLIT: dict[str, str] = {
    "subpop_eval": "eval",
    "subpop_train": "train",
}

WAVE_SUFFIX_RE = re.compile(r"_(W\d+)$")

NON_SUBSTANTIVE_MARKERS: tuple[str, ...] = (
    "not sure",
    "don't know",
    "don’t know",
    "refused",
    "no opinion",
    "no answer",
    "can't choose",
    "can’t choose",
    "dk ",
    "dk,",
    "unclear",
    "did not receive",
)

# Anchor phrases indicating clearly ordered Likert / rating / frequency / quantity scales.
ORDINAL_ANCHOR_RE = re.compile(
    r"\b("
    r"strongly agree|strongly disagree|somewhat agree|somewhat disagree|"
    r"strongly favor|strongly oppose|somewhat favor|somewhat oppose|"
    r"strongly approve|strongly disapprove|strongly support|strongly oppose|"
    r"very likely|somewhat likely|not too likely|not at all likely|very unlikely|"
    r"very important|somewhat important|not too important|not at all important|"
    r"very concerned|somewhat concerned|not too concerned|not at all concerned|"
    r"very confident|somewhat confident|not too confident|not at all confident|"
    r"very satisfied|somewhat satisfied|very dissatisfied|completely satisfied|"
    r"very favorable|somewhat favorable|very unfavorable|somewhat unfavorable|"
    r"very positive|somewhat positive|very negative|somewhat negative|"
    r"very well|somewhat well|not too well|not at all well|"
    r"very good|somewhat good|very bad|somewhat bad|"
    r"very often|fairly often|sometimes|rarely|never|"
    r"a great deal|a fair amount|not too much|not at all|hardly ever|"
    r"almost all of the time|most of the time|some of the time|"
    r"much more|somewhat more|about the same|somewhat less|much less|"
    r"much better|somewhat better|somewhat worse|much worse|"
    r"too little|about right|too much|"
    r"major problem|minor problem|not a problem|"
    r"major reason|minor reason|not a reason|"
    r"major threat|minor threat|not a threat|"
    r"major factor|minor factor|not a factor|"
    r"excellent|only fair|poor|"
    r"extremely|moderately|slightly|"
    r"always|usually|seldom|"
    r"every day|several times a week|once a week|once a month|"
    r"legal in all cases|illegal in all cases"
    r")\b",
    re.IGNORECASE,
)


def split_dir(split: str, data_dir: Path = DATA_DIR) -> Path:
    if split not in SUBPOP_SPLITS:
        raise ValueError(f"Unknown SubPOP split {split!r}; expected one of {SUBPOP_SPLITS}")
    return data_dir / f"subpop_{split}"


def raw_jsonl_path(split: str, data_dir: Path = DATA_DIR) -> Path:
    p = split_dir(split, data_dir) / f"raw_subpop_{split}.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"Missing official SubPOP raw file: {p}")
    return p


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_multiline_override(text: str) -> str:
    return "\n".join(_norm_ws(line) for line in text.replace("\r\n", "\n").split("\n")).strip()


def _has_non_substantive(opt: str) -> bool:
    low = opt.strip().lower()
    if low in {"dk", "refused", "unclear", "not sure", "don't know", "don’t know", "no opinion"}:
        return True
    return any(marker in low for marker in NON_SUBSTANTIVE_MARKERS)


def _extract_block_name(split: str, qkey: str) -> str:
    if split == "eval":
        return "GSS 2022"
    m = WAVE_SUFFIX_RE.search(qkey)
    return f"Pew ATP {m.group(1)}" if m else "Pew ATP"


def load_raw_subpop_split(
    split: str, data_dir: Path = DATA_DIR
) -> tuple[list[dict[str, Any]], OrderedDict[str, list[dict[str, Any]]]]:
    """Load and validate an official raw `jjssuh/subpop` JSONL split."""
    p = raw_jsonl_path(split, data_dir)
    rows: list[dict[str, Any]] = []
    by_qkey: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            rec = json.loads(line)
            qkey = rec["qkey"]
            attr = rec["attribute"]
            grp = rec["group"]
            if attr not in SUBPOP_ATTR_TO_COL:
                raise KeyError(f"{p.name}:{line_no}: unknown SubPOP attribute {attr!r}")
            opts = rec["options"]
            resps = rec["responses"]
            if len(opts) != len(resps) + 1:
                raise ValueError(
                    f"{p.name}:{line_no} ({qkey}): expected len(options) == len(responses) + 1, "
                    f"got {len(opts)} vs {len(resps)}"
                )
            if any(float(v) < -1e-9 for v in resps):
                raise ValueError(f"{p.name}:{line_no} ({qkey}): negative response probability in {resps}")
            prob_sum = sum(float(v) for v in resps)
            if abs(prob_sum - 1.0) > 1e-4:
                raise ValueError(f"{p.name}:{line_no} ({qkey}): response probabilities sum to {prob_sum:.6f}, expected 1.0")
            rows.append(rec)
            by_qkey.setdefault(qkey, []).append(rec)

    for qkey, grp_rows in by_qkey.items():
        q0 = grp_rows[0]["question"]
        o0 = grp_rows[0]["options"]
        ord0 = grp_rows[0]["ordinal"]
        seen_ag: set[tuple[str, str]] = set()
        for r in grp_rows:
            ag = (r["attribute"], r["group"])
            if ag in seen_ag:
                raise ValueError(f"{p.name} ({qkey}): duplicate subgroup {ag}")
            seen_ag.add(ag)
            if r["question"] != q0:
                raise ValueError(f"{p.name} ({qkey}): inconsistent question text across subgroups")
            if r["options"] != o0:
                raise ValueError(f"{p.name} ({qkey}): inconsistent options across subgroups")
            if r["ordinal"] != ord0:
                raise ValueError(f"{p.name} ({qkey}): inconsistent ordinal across subgroups")

    return rows, by_qkey


def _infer_initial_audit_fields(
    split: str,
    eff_opts: list[str],
    ordinal: list[float],
    stripped_opt: str,
) -> tuple[bool, str, str, str]:
    """Return `(include, kev_type, family, review_note)` for initial question_audit.csv generation."""
    notes: list[str] = []
    if stripped_opt != "Refused":
        notes.append(f"Upstream stripped last option {stripped_opt!r} instead of 'Refused'.")

    if len(eff_opts) < 2:
        notes.append(f"Broken upstream: only {len(eff_opts)} substantive option remains after stripping {stripped_opt!r}.")
        return False, "", "1_Format_Skip", " ".join(notes)

    dk_opts = [o for o in eff_opts if _has_non_substantive(o)]
    if dk_opts:
        notes.append(
            f"Includes non-substantive option(s) {dk_opts}; typed as 'choice' unless added to drop_options."
        )
        return True, "choice", "4_MC_Single", " ".join(notes)

    if len(eff_opts) == 2:
        return True, "choice", "4_MC_Single", " ".join(notes)

    # 3+ substantive options without DK/Refused
    is_strict_monotonic = (
        len(ordinal) == len(eff_opts)
        and len(set(ordinal)) == len(ordinal)
        and (ordinal == sorted(ordinal) or ordinal == sorted(ordinal, reverse=True))
    )
    anchor_hits = sum(1 for o in eff_opts if ORDINAL_ANCHOR_RE.search(o))
    if split == "eval":
        kev_type = "score" if is_strict_monotonic else "choice"
    else:
        kev_type = "score" if anchor_hits >= 2 else "choice"

    return True, kev_type, "4_MC_Single", " ".join(notes)


def bootstrap_subpop_audit_csv(
    split: str,
    overwrite: bool = False,
    data_dir: Path = DATA_DIR,
) -> Path:
    """Create initial `question_audit.csv` for an official SubPOP split."""
    out_dir = split_dir(split, data_dir)
    csv_path = out_dir / "question_audit.csv"
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"{csv_path} already exists; pass overwrite=True to regenerate")

    _, by_qkey = load_raw_subpop_split(split, data_dir)
    records: list[dict[str, Any]] = []

    for idx, (qkey, grp_rows) in enumerate(by_qkey.items()):
        first = grp_rows[0]
        raw_opts = [_norm_ws(str(o)) for o in first["options"]]
        eff_opts = raw_opts[:-1]
        stripped_opt = raw_opts[-1]
        ordinal = [float(x) for x in first["ordinal"]]
        q_text = _norm_ws(str(first["question"]))
        include, kev_type, family, review_note = _infer_initial_audit_fields(
            split, eff_opts, ordinal, stripped_opt
        )
        records.append(
            {
                "catalog_idx": str(idx),
                "QuestionID": qkey,
                "n_groups": str(len(grp_rows)),
                "family": family,
                "BlockName": _extract_block_name(split, qkey),
                "include": "True" if include else "False",
                "kev_type": kev_type,
                "drop_options": "",
                "review_note": review_note,
                "QuestionText": q_text,
                "Options_preview": f"[{len(eff_opts)}] " + " | ".join(eff_opts),
            }
        )

    out_df = pd.DataFrame(records)
    tmp_path = csv_path.with_suffix(".csv.tmp")
    out_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)
    return csv_path


def _load_overrides(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    ov_path = dataset_dir / "question_overrides.json"
    if not ov_path.is_file():
        return {}
    return json.loads(ov_path.read_text(encoding="utf-8"))


def _apply_option_transforms(
    qkey: str,
    raw_eff_opts: list[str],
    drop_set: set[str],
    opt_ov: dict[str, str] | None,
) -> tuple[list[int], list[str]]:
    """Return `(kept_indices, final_option_texts)` after `drop_options` and `options` overrides."""
    raw_set = set(raw_eff_opts)
    unknown_drops = sorted(drop_set - raw_set)
    if unknown_drops:
        raise ValueError(f"{qkey}: drop_options {unknown_drops} not found in options {raw_eff_opts}")

    kept_indices = [i for i, o in enumerate(raw_eff_opts) if o not in drop_set]
    if len(kept_indices) < 2:
        raise ValueError(
            f"{qkey}: fewer than 2 options remain ({len(kept_indices)}) after drop_options={sorted(drop_set)}"
        )

    final_opts: list[str] = []
    for i in kept_indices:
        raw_o = raw_eff_opts[i]
        if opt_ov is not None and raw_o in opt_ov:
            label = _norm_ws(str(opt_ov[raw_o]))
            if not label:
                raise ValueError(f"{qkey}: option override for {raw_o!r} is empty")
            final_opts.append(label)
        else:
            final_opts.append(raw_o)

    if len(set(final_opts)) != len(final_opts):
        raise ValueError(f"{qkey}: duplicate option labels after overrides: {final_opts}")

    return kept_indices, final_opts


def _renormalize_probs(qkey: str, group_id: str, raw_probs: list[float], kept_indices: list[int]) -> list[float]:
    kept = [float(raw_probs[i]) for i in kept_indices]
    total = sum(kept)
    if total <= 0.0:
        raise ValueError(
            f"{qkey} ({group_id}): remaining probability mass is 0 after drop_options"
        )
    return [p / total for p in kept]


def subpop_audit_details(
    split: str,
    audit_df: pd.DataFrame,
    data_dir: Path = DATA_DIR,
) -> dict[str, Any]:
    """Build `.audit_details.json` payload for an official SubPOP split."""
    dataset_dir = split_dir(split, data_dir)
    _, by_qkey = load_raw_subpop_split(split, data_dir)
    overrides = _load_overrides(dataset_dir)

    items: dict[str, Any] = {}
    for _, row in audit_df.iterrows():
        rec = {col: str(row[col]) for col in audit_df.columns}
        qkey = rec["QuestionID"]
        row_key = f"{rec['catalog_idx']}:{qkey}"
        if qkey not in by_qkey:
            raise KeyError(f"subpop_{split}: QuestionID {qkey!r} in question_audit.csv missing from raw JSONL")

        grp_rows = by_qkey[qkey]
        raw_eff_opts = [_norm_ws(str(o)) for o in grp_rows[0]["options"][:-1]]
        n_opts = len(raw_eff_opts)

        # Unweighted mean probability across available subgroups, scaled to per-10,000 basis points
        # so the UI's existing Share % and Distribution bar reflect exact mean shares (to 0.01%).
        mean_probs = np.mean([[float(x) for x in r["responses"]] for r in grp_rows], axis=0)
        opt_counts = [
            {
                "option": raw_eff_opts[i],
                "count": int(round(float(mean_probs[i]) * 10_000)),
            }
            for i in range(n_opts)
        ]

        drop_set = {s.strip() for s in rec["drop_options"].split("|") if s.strip()}
        ov = overrides[row_key] if row_key in overrides else None
        opt_ov = ov["options"] if ov is not None and "options" in ov else None

        export_err: str | None = None
        try:
            kept_idx, _ = _apply_option_transforms(qkey, raw_eff_opts, drop_set, opt_ov)
            for r in grp_rows:
                gid = f"SUBPOP/{r['attribute']}/{r['group']}"
                _renormalize_probs(qkey, gid, r["responses"], kept_idx)
        except ValueError as exc:
            export_err = str(exc)

        items[row_key] = {
            "QuestionType": "MC",
            "Selector": "SAVR",
            "rows": [],
            "options": opt_counts,
            "row_breakdowns": [],
            "qtext_variants": [],
            "db_variants": [],
            "opt_variant_sets": [raw_eff_opts],
            "paired_variants": [],
            "paired_variants_total": 0,
            "sample_texts": [],
            "in_block_db_text": "",
            "export_problem_if_included": {
                "choice": export_err,
                "score": export_err,
            },
        }

    return {
        "version": 2,
        "schema_kind": "question_audit",
        "study_contexts": {},
        "items": items,
    }


def _load_standardized_subpop_personas(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, list[str]]:
    """Return `(standardized_df_indexed_by_CASEID, canonical_persona_cols_in_spec_order)`."""
    specs = load_persona_spec()
    raw_df = load_raw_subpop_personas(data_dir)
    std_df = standardize_dataframe(raw_df, dataset="subpop", specs=specs)
    persona_cols = [c for c in specs if c in std_df.columns]
    if "country" not in persona_cols:
        raise ValueError("Standardized SubPOP personas missing 'country' column")
    std_df = std_df[["CASEID", *persona_cols]].copy()
    if std_df["CASEID"].duplicated().any():
        raise ValueError("Duplicate CASEID in standardized SubPOP personas")
    return std_df.set_index("CASEID"), persona_cols


def export_subpop_parquet(
    split: str,
    write: bool = True,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Export an official SubPOP split to `subpop_{split}_US_marginals.parquet`."""
    dataset_dir = split_dir(split, data_dir)
    csv_path = dataset_dir / "question_audit.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}; run bootstrap_subpop_audit_csv({split!r}) first")

    audit_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required_cols = {"catalog_idx", "QuestionID", "BlockName", "include", "kev_type", "drop_options"}
    missing_cols = required_cols - set(audit_df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path}: missing required columns {sorted(missing_cols)}")

    _, by_qkey = load_raw_subpop_split(split, data_dir)
    overrides = _load_overrides(dataset_dir)
    personas_by_id, persona_cols = _load_standardized_subpop_personas(data_dir)

    out_rows: list[dict[str, Any]] = []
    included_df = audit_df[audit_df["include"] == "True"]
    if len(included_df) == 0:
        raise ValueError(f"{csv_path}: no rows have include == 'True'")

    for _, row in included_df.iterrows():
        qkey = str(row["QuestionID"])
        row_key = f"{row['catalog_idx']}:{qkey}"
        kev_type = str(row["kev_type"]).strip()
        if kev_type not in ("choice", "score"):
            raise ValueError(f"subpop_{split} {row_key}: invalid kev_type {kev_type!r} for included question")
        if qkey not in by_qkey:
            raise KeyError(f"subpop_{split} {row_key}: missing from raw JSONL")

        grp_rows = by_qkey[qkey]
        raw_stem = _norm_ws(str(grp_rows[0]["question"]))
        raw_eff_opts = [_norm_ws(str(o)) for o in grp_rows[0]["options"][:-1]]
        drop_set = {s.strip() for s in str(row["drop_options"]).split("|") if s.strip()}

        ov = overrides[row_key] if row_key in overrides else None
        stem = _normalize_multiline_override(str(ov["stem"])) if ov is not None and "stem" in ov else raw_stem
        if not stem:
            raise ValueError(f"subpop_{split} {row_key}: empty question stem")
        opt_ov = ov["options"] if ov is not None and "options" in ov else None

        kept_indices, final_opts = _apply_option_transforms(qkey, raw_eff_opts, drop_set, opt_ov)
        option_codes = list(range(1, len(final_opts) + 1))
        label = f"{row['BlockName']}: {qkey}"

        for r in grp_rows:
            attr_raw = r["attribute"]
            grp_raw = r["group"]
            attr_col = SUBPOP_ATTR_TO_COL[attr_raw]
            case_id = f"SUBPOP/{attr_raw}/{grp_raw}"
            if case_id not in personas_by_id.index:
                raise KeyError(f"subpop_{split} {qkey}: subgroup {case_id!r} missing from standardized personas")

            probs = _renormalize_probs(qkey, case_id, r["responses"], kept_indices)
            p_row = personas_by_id.loc[case_id]
            out_rec: dict[str, Any] = {
                "CASEID": case_id,
                "attribute": attr_col,
            }
            for col in persona_cols:
                val = p_row[col]
                out_rec[col] = None if pd.isna(val) else str(val)
            out_rec["v"] = qkey
            out_rec["label"] = label
            out_rec["question"] = stem
            out_rec["type"] = kev_type
            out_rec["reversed"] = False
            out_rec["option_codes"] = option_codes
            out_rec["option_texts"] = final_opts
            out_rec["target_probs"] = probs
            out_rec["refusal_rate"] = float(r["refusal_rate"])
            out_rows.append(out_rec)

    out_df = pd.DataFrame(out_rows)
    if write:
        out_path = dataset_dir / f"subpop_{split}_US_marginals.parquet"
        tmp_path = out_path.with_suffix(".parquet.tmp")
        out_df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap and export official SubPOP splits.")
    parser.add_argument("--split", choices=["eval", "train", "all"], default="all")
    parser.add_argument("--bootstrap", action="store_true", help="Create initial question_audit.csv")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing question_audit.csv on bootstrap")
    args = parser.parse_args()

    splits = list(SUBPOP_SPLITS) if args.split == "all" else [args.split]
    for sp in splits:
        if args.bootstrap:
            p = bootstrap_subpop_audit_csv(sp, overwrite=args.overwrite)
            print(f"Bootstrapped {p}")
        df = export_subpop_parquet(sp, write=True)
        nc = df[df["type"] == "choice"]["v"].nunique()
        ns = df[df["type"] == "score"]["v"].nunique()
        print(
            f"subpop_{sp}_US_marginals.parquet: {len(df):,} rows | "
            f"{df['CASEID'].nunique()} subgroups | {df['v'].nunique():,} questions ({nc:,} choice, {ns:,} score)"
        )


if __name__ == "__main__":
    main()
