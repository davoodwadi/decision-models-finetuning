"""Fail-fast exporter for the NCI HINTS 7 (2024) long parquet (official Stata archive only).

Reads, from `code/data/HINTS/`:
- `HINTS7_STATA_20250731.zip` (`hints7_public.dta`): answers, variable labels, and value labels.
- `question_audit.csv`: include, kev_type, QuestionText (untruncated codebook wording, see
  `HINTS/build_question_catalog.py`), and drop_options per question.
- `question_overrides.json` (optional): stem / option text edits made in the audit app.

Attaches personas only via `standardize_dataframe(load_raw_hints7_personas(), dataset="hints7")`,
keeping only the `persona_spec.yaml` attributes. Strictly refuses to export any Section R /
persona column in `HINTS7_PERSONA_RAW_COLS`. Writes `hints7_long.parquet` and
`questions_verified.csv` atomically.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re
import zipfile

import pandas as pd
from pandas.io.stata import StataReader

from data.option_labels import format_option_texts, is_option_relabeled
from data.persona_loaders import (
    DATA_DIR,
    HINTS7_DTA_MEMBER,
    HINTS7_MISSING_CODES,
    HINTS7_PERSONA_RAW_COLS,
    HINTS7_ZIP_REL,
    load_raw_hints7_personas,
)
from data.persona_spec import load_persona_spec, standardize_dataframe

HINTS_FOLDER = "HINTS"
PARQUET_NAME = "hints7_long.parquet"
COUNTRY = "US"
MAX_OPTIONS = 15
SECTION_AQ_RE = re.compile(r'^"?([A-Q])(\d+[A-Za-z]?)\.\s*')
OVERRIDE_KEYS = {"stem", "options"}
OPTION_ORIGINS = ("stata_label", "stata_label_code", "stata_label_filled", "code_only", "override")


def _hints_dir(data_dir: Path) -> Path:
    return data_dir / HINTS_FOLDER


def _load_audit(hints_dir: Path) -> pd.DataFrame:
    audit_path = hints_dir / "question_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing question_audit.csv: {audit_path}")
    df = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
    required = {"catalog_idx", "QuestionID", "include", "kev_type", "QuestionText", "data_label", "drop_options"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"{audit_path}: missing required columns {missing_cols}")
    if df["QuestionID"].duplicated().any():
        dups = sorted(set(df.loc[df["QuestionID"].duplicated(), "QuestionID"]))
        raise ValueError(f"{audit_path}: duplicate QuestionID rows {dups}")

    types: list[str] = []
    for row in df.to_dict("records"):
        if row["include"] not in ("True", "False"):
            raise ValueError(f"{audit_path}: bad include={row['include']!r} on {row['QuestionID']!r}")
        if row["include"] == "True":
            if row["kev_type"] not in ("score", "choice"):
                raise ValueError(f"{audit_path}: included {row['QuestionID']!r} has bad kev_type={row['kev_type']!r}")
            if not row["QuestionText"].strip():
                raise ValueError(f"{audit_path}: included {row['QuestionID']!r} has empty QuestionText")
            types.append(row["kev_type"])
        else:
            types.append("exclude")
    return pd.DataFrame(
        {
            "_row_key": [f"{c}:{q}" for c, q in zip(df["catalog_idx"], df["QuestionID"])],
            "v": df["QuestionID"],
            "type": types,
            "question": df["QuestionText"],
            "data_label": df["data_label"],
            "drop_options": df["drop_options"],
        }
    )


def _load_answers(data_dir: Path) -> tuple[dict[str, str], dict[str, dict[int, str]], pd.DataFrame]:
    """Variable labels, offered (non-negative) value labels per variable, and answers with string CASEID."""
    zip_path = data_dir / HINTS7_ZIP_REL
    if not zip_path.is_file():
        raise FileNotFoundError(f"Official HINTS 7 Stata archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        dta_bytes = zf.read(HINTS7_DTA_MEMBER)

    with StataReader(io.BytesIO(dta_bytes)) as reader:
        var_labels = reader.variable_labels()
        label_sets = reader.value_labels()
        value_labels = {
            v: (
                {int(k): str(t) for k, t in sorted(label_sets[s].items()) if int(k) >= 0}
                if s in label_sets
                else {}
            )
            for v, s in zip(reader._varlist, reader._lbllist, strict=True)
        }

    raw = pd.read_stata(io.BytesIO(dta_bytes), convert_categoricals=False).copy()
    if raw["HHID"].isna().any():
        raise ValueError(f"{zip_path}: missing HHID values")
    raw["CASEID"] = raw["HHID"].astype(str)
    if raw["CASEID"].duplicated().any():
        raise ValueError(f"{zip_path}: duplicate HHID values")
    return var_labels, value_labels, raw


def _question_columns(raw: pd.DataFrame, var_labels: dict[str, str]) -> set[str]:
    """All Section A-Q columns in `hints7_public.dta` excluding persona/Section R variables."""
    return {
        c
        for c in raw.columns
        if c not in HINTS7_PERSONA_RAW_COLS and c in var_labels and SECTION_AQ_RE.match(str(var_labels[c]))
    }


def _valid_series(raw: pd.DataFrame, v: str) -> pd.Series:
    """Validate negative missing codes strictly and return non-null valid integer responses (`code >= 0`)."""
    s = raw[v].dropna()
    neg = s[s < 0]
    unexpected = set(int(x) for x in neg.unique()) - HINTS7_MISSING_CODES
    if unexpected:
        raise ValueError(f"hints7.{v}: unexpected negative code(s) {sorted(unexpected)}")
    valid = s[s >= 0]
    return valid.astype("int64")


def _all_options(
    v: str,
    is_score: bool,
    value_labels: dict[str, dict[int, str]],
    raw: pd.DataFrame,
) -> dict[int, object]:
    """Labelled positive options for `v`, plus any answered-but-unlabelled non-negative codes."""
    opts: dict[int, object] = dict(value_labels[v])
    given = set(_valid_series(raw, v).unique())
    for c in given - set(opts):
        opts[c] = float("nan")
    return dict(sorted(opts.items()))


def _export_problem(
    v: str,
    is_score: bool,
    question_cols: set[str],
    value_labels: dict[str, dict[int, str]],
    raw: pd.DataFrame,
) -> str | None:
    """Why `v` cannot be exported if included with this kev_type; None when it can."""
    if v in HINTS7_PERSONA_RAW_COLS:
        return f"{v} is a persona/demographic column (Section R / geo / derived)"
    if v not in question_cols:
        return f"{v} is not a Section A-Q question column in hints7_public.dta"
    opts = _all_options(v, is_score, value_labels, raw)
    if len(opts) < 2:
        return f"only {len(opts)} valid answer option(s)"
    if len(opts) > MAX_OPTIONS:
        return f"{len(opts)} answer options (limit {MAX_OPTIONS})"
    return None


def _option_texts(codes: list[int], texts: list[object]) -> list[str]:
    """Format Stata option labels via the shared rule in `data.option_labels`."""
    return format_option_texts(codes, texts)


def _option_origins(codes: list[int], texts: list[object], rendered: list[str]) -> dict[int, str]:
    origins: dict[int, str] = {}
    for c, t, r in zip(codes, texts, rendered):
        if not isinstance(t, str) or not t.strip() or (r == f"{c}." and str(t) != r):
            origins[c] = "code_only"
        elif r == str(t):
            origins[c] = "stata_label"
        else:
            origins[c] = "stata_label_code"
    return origins


def _apply_question_overrides(hints_dir: Path, bank: pd.DataFrame) -> pd.DataFrame:
    overrides_path = hints_dir / "question_overrides.json"
    if not overrides_path.exists():
        return bank
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    if not isinstance(overrides, dict):
        raise TypeError(f"{overrides_path}: expected JSON object, got {type(overrides).__name__}")
    bank = bank.copy()
    key_to_idx = {k: idx for idx, k in enumerate(bank["_row_key"])}
    for row_key, ov in overrides.items():
        if not isinstance(ov, dict):
            raise TypeError(f"{overrides_path}: {row_key!r} must map to an object, got {type(ov).__name__}")
        unknown = sorted(set(ov) - OVERRIDE_KEYS)
        if unknown:
            raise ValueError(f"{overrides_path}: {row_key!r} has unsupported keys {unknown}")
        if row_key not in key_to_idx:
            continue
        idx = key_to_idx[row_key]
        if "stem" in ov:
            stem = str(ov["stem"]).strip()
            if not stem:
                raise ValueError(f"{overrides_path}: {row_key!r} has an empty stem override")
            bank.at[idx, "question"] = stem
        if "options" in ov:
            opts = list(bank.at[idx, "answer_text"])
            for raw_opt, new_opt in ov["options"].items():
                if raw_opt not in opts:
                    raise ValueError(f"{overrides_path}: option {raw_opt!r} not found in {row_key} options {opts!r}")
                opts = [str(new_opt) if o == raw_opt else o for o in opts]
            bank.at[idx, "answer_text"] = opts
    return bank


def format_not_fielded(not_fielded: list[str]) -> str:
    if not not_fielded:
        return f"hints7_{COUNTRY}: every included question has {COUNTRY} answers"
    return (
        f"hints7_{COUNTRY}: {len(not_fielded)} included question(s) with no valid {COUNTRY} answers, "
        f"left out of the parquet: {', '.join(not_fielded)}"
    )


def _final_bank(
    rows: pd.DataFrame,
    value_labels: dict[str, dict[int, str]],
    raw: pd.DataFrame,
    hints_dir: Path,
) -> tuple[pd.DataFrame, set[tuple[str, int]]]:
    bank = rows.reset_index(drop=True).copy()
    codes_col: list[list[int]] = []
    texts_col: list[list[str]] = []
    for v, q_type in zip(bank["v"], bank["type"]):
        opts = _all_options(v, q_type == "score", value_labels, raw)
        codes = list(opts)
        codes_col.append(codes)
        texts_col.append(_option_texts(codes, list(opts.values())))
    bank["answer_code"] = codes_col
    bank["answer_text"] = texts_col
    bank["options_rendered"] = [list(t) for t in texts_col]

    dropped_pairs: set[tuple[str, int]] = set()
    for idx in range(len(bank)):
        raw_drop = str(bank.at[idx, "drop_options"]).strip()
        if not raw_drop:
            continue
        v = str(bank.at[idx, "v"])
        drop_set = {s.strip() for s in raw_drop.split("|") if s.strip()}
        codes = list(bank.at[idx, "answer_code"])
        texts = list(bank.at[idx, "answer_text"])
        unknown_drops = sorted(drop_set - set(texts))
        if unknown_drops:
            raise ValueError(f"hints7.{v}: drop_options {unknown_drops} not in options {texts!r}")
        kept = [(c, t) for c, t in zip(codes, texts) if t not in drop_set]
        dropped_pairs |= {(v, int(c)) for c, t in zip(codes, texts) if t in drop_set}
        if len(kept) < 2:
            raise ValueError(f"hints7.{v}: drop_options leaves fewer than 2 options ({[t for _, t in kept]!r})")
        bank.at[idx, "answer_code"] = [c for c, _ in kept]
        bank.at[idx, "answer_text"] = [t for _, t in kept]

    bank["options_before_override"] = [list(t) for t in bank["answer_text"]]
    bank = _apply_question_overrides(hints_dir, bank)
    return bank, dropped_pairs


def hints7_question_checks(*, data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """Per-row export checks, valid US counts, and rendered option counts for `build_index.py`."""
    hints_dir = _hints_dir(data_dir)
    bank = _load_audit(hints_dir)
    var_labels, value_labels, raw = _load_answers(data_dir)
    question_cols = _question_columns(raw, var_labels)
    checks: dict[str, dict] = {}
    for row_key, v, q_type in zip(bank["_row_key"], bank["v"], bank["type"]):
        options: list[dict[str, object]] = []
        n_valid: int | None = None
        relabeled = False
        if v in question_cols:
            valid = _valid_series(raw, v)
            n_valid = int(len(valid))
            opts = _all_options(v, q_type == "score", value_labels, raw)
            counts = valid.value_counts()
            codes = list(opts)
            raw_vals = list(opts.values())
            relabeled = is_option_relabeled(codes, raw_vals)
            texts = _option_texts(codes, raw_vals)
            options = [{"option": t, "count": int(counts[c]) if c in counts.index else 0} for c, t in zip(opts, texts)]
        checks[row_key] = {
            "export_problem_if_included": {
                "choice": _export_problem(v, False, question_cols, value_labels, raw),
                "score": _export_problem(v, True, question_cols, value_labels, raw),
            },
            "n_valid_by_country": {COUNTRY: n_valid},
            "options": options,
            "is_persona": v in HINTS7_PERSONA_RAW_COLS,
            "option_relabeled": relabeled,
        }
    return checks


def hints7_final_questions(*, data_dir: Path = DATA_DIR) -> dict[str, dict[str, object]]:
    """Per-row final exported stem/options/provenance for `source_check.py`."""
    hints_dir = _hints_dir(data_dir)
    audit = _load_audit(hints_dir)
    var_labels, value_labels, raw = _load_answers(data_dir)
    overrides_path = hints_dir / "question_overrides.json"
    stem_override_keys: set[str] = set()
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise TypeError(f"{overrides_path}: expected JSON object, got {type(overrides).__name__}")
        stem_override_keys = {k for k, ov in overrides.items() if isinstance(ov, dict) and "stem" in ov}

    question_cols = _question_columns(raw, var_labels)
    renderable = audit[audit["v"].isin(question_cols)]
    bank, _ = _final_bank(renderable, value_labels, raw, hints_dir)
    by_key = {r["_row_key"]: r for r in bank.to_dict("records")}
    out: dict[str, dict[str, object]] = {}
    for row in audit.to_dict("records"):
        key = row["_row_key"]
        included = row["type"] != "exclude"
        persona = row["v"] in HINTS7_PERSONA_RAW_COLS
        data_label = str(var_labels[row["v"]]) if row["v"] in var_labels else ""
        has_stem_override = key in stem_override_keys
        problem = (
            _export_problem(row["v"], row["type"] == "score", question_cols, value_labels, raw) if included else None
        )
        if key in by_key:
            r = by_key[key]
            opts = _all_options(row["v"], row["type"] == "score", value_labels, raw)
            origin_by_code = _option_origins(list(opts), list(opts.values()), list(r["options_rendered"]))
            origins = [
                "override" if final != before else origin_by_code[int(code)]
                for code, final, before in zip(r["answer_code"], r["answer_text"], r["options_before_override"])
            ]
            out[key] = {
                "v": row["v"],
                "included": included,
                "is_persona": persona,
                "type": row["type"],
                "question": r["question"],
                "data_label": data_label,
                "has_stem_override": has_stem_override,
                "options": list(r["answer_text"]),
                "options_rendered": list(r["options_rendered"]),
                "options_before_override": list(r["options_before_override"]),
                "options_origin": origins,
                "options_unfilled": list(r["options_rendered"]),
                "export_problem": problem,
            }
        else:
            out[key] = {
                "v": row["v"],
                "included": included,
                "is_persona": persona,
                "type": row["type"],
                "question": row["question"],
                "data_label": data_label,
                "has_stem_override": has_stem_override,
                "options": [],
                "options_rendered": [],
                "options_before_override": [],
                "options_origin": [],
                "options_unfilled": [],
                "export_problem": problem,
            }
    return out


def export_hints7(*, data_dir: Path = DATA_DIR, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Export NCI HINTS 7 (2024) to `hints7_long.parquet` and `questions_verified.csv`."""
    hints_dir = _hints_dir(data_dir)
    audit = _load_audit(hints_dir)
    var_labels, value_labels, raw = _load_answers(data_dir)
    question_cols = _question_columns(raw, var_labels)

    included = audit[audit["type"] != "exclude"].reset_index(drop=True)
    failing: dict[str, str] = {}
    for row_key, v, q_type in zip(included["_row_key"], included["v"], included["type"]):
        problem = _export_problem(v, q_type == "score", question_cols, value_labels, raw)
        if problem is not None:
            failing[row_key] = problem
    if failing:
        lines = "\n".join(f"  {row_key}: {problem}" for row_key, problem in failing.items())
        raise ValueError(
            f"hints7: {len(failing)} included question(s) cannot be exported. "
            f"Exclude them in question_audit.csv (audit app: 'Blocks export' filter):\n{lines}"
        )

    bank, dropped_pairs = _final_bank(included, value_labels, raw, hints_dir)
    kept_vs = list(bank["v"])

    # Strictly validate negative missing codes and mask them to NaN before melting.
    clean_cols: dict[str, pd.Series] = {"CASEID": raw["CASEID"]}
    for v in kept_vs:
        s = raw[v]
        neg = s.dropna()
        neg = neg[neg < 0]
        unexpected = set(int(x) for x in neg.unique()) - HINTS7_MISSING_CODES
        if unexpected:
            raise ValueError(f"hints7.{v}: unexpected negative code(s) {sorted(unexpected)}")
        clean_cols[v] = s.where(s.isna() | (s >= 0), float("nan"))
    clean_df = pd.DataFrame(clean_cols)

    long = clean_df.melt(id_vars=["CASEID"], value_vars=kept_vs, var_name="v", value_name="answer_code")
    long = long.dropna(subset=["answer_code"]).copy()
    long["answer_code"] = long["answer_code"].astype("int64")
    if dropped_pairs:
        keep_mask = [(v, c) not in dropped_pairs for v, c in zip(long["v"], long["answer_code"])]
        long = long.loc[keep_mask].copy()

    answered_vs = set(long["v"])
    not_fielded = [v for v in kept_vs if v not in answered_vs]
    long["label"] = [var_labels[v] for v in long["v"]]

    text_of = {
        (v, int(c)): str(t)
        for v, codes, texts in zip(bank["v"], bank["answer_code"], bank["answer_text"])
        for c, t in zip(codes, texts)
    }
    missing_pairs = [(v, c) for v, c in zip(long["v"], long["answer_code"]) if (v, c) not in text_of]
    if missing_pairs:
        raise ValueError(f"hints7: unmapped (v, answer_code) pairs: {missing_pairs[:5]}")
    long["answer_text"] = [text_of[(v, c)] for v, c in zip(long["v"], long["answer_code"])]
    long = long.merge(
        bank[["v", "question", "type", "answer_code", "answer_text"]].rename(
            columns={"answer_code": "option_codes", "answer_text": "option_texts"}
        ),
        on="v",
        how="left",
    )

    specs = load_persona_spec()
    std_personas = standardize_dataframe(load_raw_hints7_personas(data_dir=data_dir), dataset="hints7", specs=specs)
    persona_cols = [c for c in specs if c in std_personas.columns and std_personas[c].notna().any()]
    std_personas = std_personas[["CASEID", *persona_cols]].copy()
    std_personas["CASEID"] = std_personas["CASEID"].astype(str)
    if std_personas["CASEID"].duplicated().any():
        raise ValueError("hints7: duplicate CASEID in standardized personas")
    missing_cids = set(long["CASEID"]) - set(std_personas["CASEID"])
    if missing_cids:
        raise ValueError(f"hints7: {len(missing_cids)} CASEIDs missing from standardized personas")

    long = std_personas.merge(long, on="CASEID", how="inner")
    ordered_cols = [
        "CASEID",
        *persona_cols,
        "v",
        "label",
        "question",
        "type",
        "option_codes",
        "option_texts",
        "answer_code",
        "answer_text",
    ]
    long = long[ordered_cols].reset_index(drop=True)

    verified_bank = bank[["v", "data_label", "question", "answer_code", "answer_text", "type"]].copy()
    verified_bank = verified_bank[~verified_bank["v"].isin(not_fielded)].reset_index(drop=True)

    if write:
        long_path = hints_dir / PARQUET_NAME
        tmp_pq = hints_dir / f".{PARQUET_NAME}.tmp"
        long.to_parquet(tmp_pq, index=False)
        os.replace(tmp_pq, long_path)

        verified_path = hints_dir / "questions_verified.csv"
        tmp_csv = hints_dir / ".questions_verified.csv.tmp"
        verified_bank.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, verified_path)

    return long, verified_bank, not_fielded


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the NCI HINTS 7 long parquet with standardized personas.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate without writing files.")
    args = parser.parse_args()

    long, _, not_fielded = export_hints7(write=not args.dry_run)
    print(format_not_fielded(not_fielded))
    n_choice = long.loc[long["type"] == "choice", "v"].nunique()
    n_score = long.loc[long["type"] == "score", "v"].nunique()
    persona_cols = list(long.columns[1 : list(long.columns).index("v")])
    print(
        f"{PARQUET_NAME}: {len(long):,} rows | {long['CASEID'].nunique():,} respondents | "
        f"{long['v'].nunique():,} questions ({n_choice} choice, {n_score} score) | "
        f"{len(persona_cols)} persona columns: {', '.join(persona_cols)}"
    )


if __name__ == "__main__":
    main()
