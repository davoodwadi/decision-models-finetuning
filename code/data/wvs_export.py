"""Fail-fast exporter for the WVS Wave 7 United States long parquet (official WVSA sources only).

Reads, from `WVS-Wave7/`:
- `WVS_Wave_7_United_States_Stata_v5.1.dta`: answers, variable labels and value labels (answer options and codes).
- `question_audit.csv`: include, kev_type, QuestionText (US codebook PDF wording, see `build_question_catalog.py`)
  and drop_options per question.
- `question_overrides.json` (optional): stem / option text edits made in the audit app.

Fills the questionnaire's placeholders ([your country], [nationality], ...) at export time from US_FILLS, as
issp_export fills {country}; the audit keeps the codebook wording with its placeholders.

Attaches personas only via `standardize_dataframe(load_raw_wvs7_personas(), dataset="wvs7")`, keeping only the
`persona_spec.yaml` attributes. Writes `wvs7_US_long.parquet` and `questions_verified.csv` atomically.
WorldValuesBench is not read.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

import pandas as pd
from pandas.io.stata import StataReader

from data.option_labels import format_option_texts, is_option_relabeled
from data.persona_loaders import DATA_DIR, WVS_PERSONA_RAW_COLS, load_raw_wvs7_personas
from data.persona_spec import load_persona_spec, standardize_dataframe

WVS_FOLDER = "WVS-Wave7"
DTA_NAME = "WVS_Wave_7_United_States_Stata_v5.1.dta"
PARQUET_NAME = "wvs7_US_long.parquet"
COUNTRY = "US"
MAX_OPTIONS = 15
# Stata extended missing values (.a, .b, ...) carry value labels at codes >= this bound; they read as NaN.
STATA_MISSING_MIN = 2147483621
QUESTION_COL_RE = re.compile(r"Q\d+[A-Z0-9_]*")
PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")
# Template placeholders of the WVSA master questionnaire, filled at export time for the US (like ISSP's
# `{country}` fill): the audit keeps the codebook wording with its placeholders. Question and option text.
US_FILLS: dict[str, str] = {
    "[your country]": "the United States",
    "[my country]": "my country",
    "[Country]": "the United States",
    "[nationality]": "American",
    "[Continent; e.g. Europe, Asia etc.]": "North America",
    "[village, town or city]": "village, town or city",
    "[county, region, district]": "county, region, district",
}
OVERRIDE_KEYS = {"stem", "options"}


def _wvs_dir(data_dir: Path) -> Path:
    return data_dir / WVS_FOLDER


def _load_audit(wvs_dir: Path) -> pd.DataFrame:
    audit_path = wvs_dir / "question_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing question_audit.csv: {audit_path}")
    df = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
    required = {"catalog_idx", "QuestionID", "family", "include", "kev_type", "QuestionText", "data_label", "drop_options"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"{audit_path}: missing required columns {missing_cols}")
    if df["QuestionID"].duplicated().any():
        raise ValueError(f"{audit_path}: duplicate QuestionID rows {sorted(set(df.loc[df['QuestionID'].duplicated(), 'QuestionID']))}")

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
            "family": df["family"],
            "type": types,
            "question": df["QuestionText"],
            "data_label": df["data_label"],
            "drop_options": df["drop_options"],
        }
    )


def is_persona_row(v: str, family: str) -> bool:
    """True when `v` is a WVS persona column (or its label/recode variant) or in the Demographics section."""
    return v in WVS_PERSONA_RAW_COLS or v.split("_", 1)[0] in WVS_PERSONA_RAW_COLS or family.endswith("_Demographics")


def _load_answers(wvs_dir: Path) -> tuple[dict[str, str], dict[str, dict[int, str]], pd.DataFrame]:
    """Variable labels, offered (non-missing) value labels per variable, and the answers with CASEID."""
    dta_path = wvs_dir / DTA_NAME
    if not dta_path.exists():
        raise FileNotFoundError(f"Official WVS Wave 7 US Stata file not found: {dta_path}")
    with StataReader(dta_path) as reader:
        var_labels = reader.variable_labels()
        label_sets = reader.value_labels()
        value_labels = {
            v: {int(k): str(t) for k, t in label_sets[s].items() if int(k) < STATA_MISSING_MIN} if s in label_sets else {}
            for v, s in zip(reader._varlist, reader._lbllist)
        }
    raw = pd.read_stata(dta_path, convert_categoricals=False).copy()
    if raw["D_INTERVIEW"].isna().any():
        raise ValueError(f"{dta_path}: missing D_INTERVIEW values")
    raw["CASEID"] = raw["D_INTERVIEW"].astype("int64")
    if raw["CASEID"].duplicated().any():
        raise ValueError(f"{dta_path}: duplicate D_INTERVIEW values")
    return var_labels, value_labels, raw


def _question_columns(raw: pd.DataFrame) -> set[str]:
    return {c for c in raw.columns if QUESTION_COL_RE.fullmatch(c)}


def _all_options(v: str, is_score: bool, value_labels: dict[str, dict[int, str]], raw: pd.DataFrame) -> dict[int, object]:
    """Labelled options for `v`, plus unlabelled interior scale points (score) and answered-but-unlabelled codes."""
    opts: dict[int, object] = dict(value_labels[v])
    given = {int(x) for x in raw[v].dropna()}
    span = set(range(min(opts), max(opts) + 1)) if opts and is_score else set()
    for c in (span | given) - set(opts):
        opts[c] = float("nan")
    return dict(sorted(opts.items()))


def _export_problem(
    v: str,
    is_score: bool,
    question_cols: set[str],
    value_labels: dict[str, dict[int, str]],
    raw: pd.DataFrame,
) -> str | None:
    """Why `v` cannot be exported if included with this kev_type; None when it can.

    Only properties of the question block the export. No valid US answers is not one: the question is left out of
    the parquet and reported (see `export_wvs7`).
    """
    if v not in question_cols:
        return f"{v} is not a question column in the .dta"
    if v in WVS_PERSONA_RAW_COLS:
        return f"{v} is a persona column read by load_raw_wvs7_personas"
    n_opts = len(_all_options(v, is_score, value_labels, raw))
    if n_opts > MAX_OPTIONS:
        return f"{n_opts} answer options (limit {MAX_OPTIONS})"
    return None


def wvs_question_checks(*, data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """For every `question_audit.csv` row key: why export would fail if included as choice / as score,
    the number of valid US answers (None when the question is not in the .dta), and the option texts as the
    exporter renders them (before drop_options and overrides, for the row's kev_type; choice when excluded)
    with their US answer counts."""
    wvs_dir = _wvs_dir(data_dir)
    bank = _load_audit(wvs_dir)
    _, value_labels, raw = _load_answers(wvs_dir)
    question_cols = _question_columns(raw)
    checks: dict[str, dict] = {}
    for row_key, v, family, q_type in zip(bank["_row_key"], bank["v"], bank["family"], bank["type"]):
        options: list[dict[str, object]] = []
        relabeled = False
        if v in question_cols:
            opts = _all_options(v, q_type == "score", value_labels, raw)
            counts = raw[v].dropna().astype("int64").value_counts()
            codes = list(opts)
            raw_vals = list(opts.values())
            relabeled = is_option_relabeled(codes, raw_vals)
            texts = _render_options(codes, raw_vals)
            options = [{"option": t, "count": int(counts[c]) if c in counts.index else 0} for c, t in zip(opts, texts)]
        checks[row_key] = {
            "export_problem_if_included": {
                "choice": _export_problem(v, False, question_cols, value_labels, raw),
                "score": _export_problem(v, True, question_cols, value_labels, raw),
            },
            "n_valid_by_country": {COUNTRY: int(raw[v].notna().sum()) if v in question_cols else None},
            "options": options,
            "is_persona": is_persona_row(str(v), str(family)),
            "option_relabeled": relabeled,
        }
    return checks


def _fill_placeholders(text: str) -> str:
    """Fill the US values for the questionnaire's placeholders; anything left is caught by the export check."""
    for placeholder, value in US_FILLS.items():
        text = text.replace(placeholder, value)
    return text


def _render_options(codes: list[int], texts: list[object]) -> list[str]:
    """Option texts as exported, before drop_options and overrides: `_option_texts`, then placeholder fills."""
    return [_fill_placeholders(t) for t in _option_texts(codes, texts)]


def _option_texts(codes: list[int], texts: list[object]) -> list[str]:
    """Format Stata option labels via the shared rule in `data.option_labels`."""
    return format_option_texts(codes, texts)


# Where each exported option text comes from (shown as provenance tags in the audit app's Source tab).
OPTION_ORIGINS = ("stata_label", "stata_label_code", "stata_label_filled", "code_only", "override")


def _option_origins(codes: list[int], texts: list[object], rendered: list[str]) -> dict[int, str]:
    """Per code: `stata_label` (value label as written), `stata_label_code` (label with the exporter's code prefix),
    `stata_label_filled` (label with a placeholder filled by US_FILLS) or `code_only` (no label, or an interior
    endpoint-anchored point rendered as `{c}.`)."""
    origins: dict[int, str] = {}
    for c, t, r in zip(codes, texts, rendered):
        if not isinstance(t, str) or not t.strip() or (r == f"{c}." and str(t) != r):
            origins[c] = "code_only"
        elif _fill_placeholders(str(t)) != str(t):
            origins[c] = "stata_label_filled"
        elif r == str(t):
            origins[c] = "stata_label"
        else:
            origins[c] = "stata_label_code"
    return origins


def _apply_question_overrides(wvs_dir: Path, bank: pd.DataFrame) -> pd.DataFrame:
    overrides_path = wvs_dir / "question_overrides.json"
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
            # Overrides on excluded or not-fielded questions do not reach the parquet.
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
    """One-line report of included questions left out of the parquet because they have no valid US answers."""
    if not not_fielded:
        return f"wvs7_{COUNTRY}: every included question has {COUNTRY} answers"
    return (
        f"wvs7_{COUNTRY}: {len(not_fielded)} included question(s) with no valid {COUNTRY} answers, "
        f"left out of the parquet: {', '.join(not_fielded)}"
    )


def _final_bank(
    rows: pd.DataFrame,
    value_labels: dict[str, dict[int, str]],
    raw: pd.DataFrame,
    wvs_dir: Path,
) -> tuple[pd.DataFrame, set[tuple[str, int]]]:
    """Question and option texts exactly as the exporter emits them, for audit rows whose `v` is a .dta column.

    Adds `answer_code` / `answer_text` (after drop_options, then question_overrides.json), `options_rendered`
    (before both) and `options_before_override` (after drops, before overrides). Rows typed `exclude` are rendered
    as choice. Returns the bank and the dropped (v, code) pairs.
    """
    bank = rows.reset_index(drop=True).copy()
    # Placeholders are filled before stem overrides, as in issp_export (an override replaces the filled text).
    bank["question"] = [_fill_placeholders(q) for q in bank["question"]]
    codes_col: list[list[int]] = []
    texts_col: list[list[str]] = []
    for v, q_type in zip(bank["v"], bank["type"]):
        opts = _all_options(v, q_type == "score", value_labels, raw)
        codes = list(opts)
        codes_col.append(codes)
        texts_col.append(_render_options(codes, list(opts.values())))
    bank["answer_code"] = codes_col
    bank["answer_text"] = texts_col
    bank["options_rendered"] = [list(t) for t in texts_col]

    # drop_options (before option text overrides, matching audit-app semantics).
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
            raise ValueError(f"wvs7.{v}: drop_options {unknown_drops} not in options {texts!r}")
        kept = [(c, t) for c, t in zip(codes, texts) if t not in drop_set]
        dropped_pairs |= {(v, int(c)) for c, t in zip(codes, texts) if t in drop_set}
        if len(kept) < 2:
            raise ValueError(f"wvs7.{v}: drop_options leaves fewer than 2 options ({[t for _, t in kept]!r})")
        bank.at[idx, "answer_code"] = [c for c, _ in kept]
        bank.at[idx, "answer_text"] = [t for _, t in kept]

    bank["options_before_override"] = [list(t) for t in bank["answer_text"]]
    bank = _apply_question_overrides(wvs_dir, bank)
    return bank, dropped_pairs


def wvs_final_questions(*, data_dir: Path = DATA_DIR) -> dict[str, dict[str, object]]:
    """For every audit row: the question text and options the exporter would emit now (from the audit CSV,
    drop_options and question_overrides.json, not from the parquet), each option's origin (OPTION_ORIGINS), and the
    export problem if it is included. Rows whose `v` is not in the .dta have no options."""
    wvs_dir = _wvs_dir(data_dir)
    audit = _load_audit(wvs_dir)
    var_labels, value_labels, raw = _load_answers(wvs_dir)
    overrides_path = wvs_dir / "question_overrides.json"
    stem_override_keys: set[str] = set()
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise TypeError(f"{overrides_path}: expected JSON object, got {type(overrides).__name__}")
        stem_override_keys = {k for k, ov in overrides.items() if isinstance(ov, dict) and "stem" in ov}
    question_cols = _question_columns(raw)
    renderable = audit[audit["v"].isin(question_cols)]
    bank, _ = _final_bank(renderable, value_labels, raw, wvs_dir)
    by_key = {r["_row_key"]: r for r in bank.to_dict("records")}
    out: dict[str, dict[str, object]] = {}
    for row in audit.to_dict("records"):
        key = row["_row_key"]
        included = row["type"] != "exclude"
        persona = is_persona_row(str(row["v"]), str(row["family"]))
        data_label = str(var_labels[row["v"]]) if row["v"] in var_labels else ""
        has_stem_override = key in stem_override_keys
        problem = (
            _export_problem(row["v"], row["type"] == "score", question_cols, value_labels, raw) if included else None
        )
        if key in by_key:
            r = by_key[key]
            opts = _all_options(row["v"], row["type"] == "score", value_labels, raw)
            origin_by_code = _option_origins(list(opts), list(opts.values()), list(r["options_rendered"]))
            unfilled_rendered = _option_texts(list(opts), list(opts.values()))
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
                # Stata text before placeholder fills, aligned with `options_rendered` (covers both kept and dropped).
                "options_unfilled": unfilled_rendered,
                "export_problem": problem,
            }
        else:
            out[key] = {
                "v": row["v"],
                "included": included,
                "is_persona": persona,
                "type": row["type"],
                "question": _fill_placeholders(row["question"]),
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


def export_wvs7(*, data_dir: Path = DATA_DIR, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Export WVS Wave 7 US to `wvs7_US_long.parquet` and `questions_verified.csv`.

    Returns `(long, verified_bank, not_fielded)`; `not_fielded` lists included questions with no valid US answers
    (missing codes only, or only answers in `drop_options`). They are absent from the parquet and never block.
    """
    wvs_dir = _wvs_dir(data_dir)
    audit = _load_audit(wvs_dir)
    var_labels, value_labels, raw = _load_answers(wvs_dir)
    question_cols = _question_columns(raw)

    included = audit[audit["type"] != "exclude"].reset_index(drop=True)
    failing: dict[str, str] = {}
    for row_key, v, q_type in zip(included["_row_key"], included["v"], included["type"]):
        problem = _export_problem(v, q_type == "score", question_cols, value_labels, raw)
        if problem is not None:
            failing[row_key] = problem
    if failing:
        lines = "\n".join(f"  {row_key}: {problem}" for row_key, problem in failing.items())
        raise ValueError(
            f"wvs7: {len(failing)} included question(s) cannot be exported. "
            f"Exclude them in question_audit.csv (audit app: 'Blocks export' filter):\n{lines}"
        )

    bank, dropped_pairs = _final_bank(included, value_labels, raw, wvs_dir)

    placeholder_rows = [
        f"{v}: {text!r}"
        for v, q, opts in zip(bank["v"], bank["question"], bank["answer_text"])
        for text in [q, *opts]
        if PLACEHOLDER_RE.search(text)
    ]
    if placeholder_rows:
        raise ValueError("wvs7: unfilled [placeholder] in included question or option text:\n  " + "\n  ".join(placeholder_rows))

    kept_vs = list(bank["v"])
    long = raw[["CASEID", *kept_vs]].melt(id_vars=["CASEID"], value_vars=kept_vs, var_name="v", value_name="answer_code")
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
        raise ValueError(f"wvs7: unmapped (v, answer_code) pairs: {missing_pairs[:5]}")
    long["answer_text"] = [text_of[(v, c)] for v, c in zip(long["v"], long["answer_code"])]
    long = long.merge(
        bank[["v", "question", "type", "answer_code", "answer_text"]].rename(
            columns={"answer_code": "option_codes", "answer_text": "option_texts"}
        ),
        on="v",
        how="left",
    )

    # Personas: persona_loaders + persona_spec.yaml only (spec attributes with at least one value).
    specs = load_persona_spec()
    std_personas = standardize_dataframe(load_raw_wvs7_personas(data_dir=data_dir), dataset="wvs7", specs=specs)
    persona_cols = [c for c in specs if c in std_personas.columns and std_personas[c].notna().any()]
    std_personas = std_personas[["CASEID", *persona_cols]].copy()
    std_personas["CASEID"] = std_personas["CASEID"].astype("int64")
    if std_personas["CASEID"].duplicated().any():
        raise ValueError("wvs7: duplicate CASEID in standardized personas")
    missing_cids = set(long["CASEID"]) - set(std_personas["CASEID"])
    if missing_cids:
        raise ValueError(f"wvs7: {len(missing_cids)} CASEIDs missing from standardized personas")

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
        long_path = wvs_dir / PARQUET_NAME
        tmp_pq = wvs_dir / f".{PARQUET_NAME}.tmp"
        long.to_parquet(tmp_pq, index=False)
        os.replace(tmp_pq, long_path)

        verified_path = wvs_dir / "questions_verified.csv"
        tmp_csv = wvs_dir / ".questions_verified.csv.tmp"
        verified_bank.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, verified_path)

    return long, verified_bank, not_fielded


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the WVS Wave 7 US long parquet with standardized personas.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate without writing files.")
    args = parser.parse_args()

    long, _, not_fielded = export_wvs7(write=not args.dry_run)
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
