"""Unified fail-fast exporter for ISSP (2015-2024 US) long parquets.

Replaces the 10 per-wave `data_check_<year>.ipynb` notebooks with a single
deterministic Python pipeline that:
1. Reads each wave's raw GESIS `.dta` file, `question_audit.csv`, optional
   `label_fixes.csv`, optional `*_missing.txt`, and optional `question_overrides.json`.
2. Builds verified substantive questions, option codes, option texts, and answers.
3. Attaches standardized personas directly from `persona_loaders.py` + `persona_spec.yaml`
   (zero legacy hardcoded collapsing or stale persona columns).
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
from data.persona_loaders import DATA_DIR, load_raw_issp_personas
from data.persona_spec import load_persona_spec, standardize_dataframe

ISSP_YEARS: tuple[int, ...] = tuple(range(2015, 2025))
VALID_BANK_TYPES = {"score", "choice", "exclude"}
MAX_OPTIONS = 15

# Labels for unlabelled interior scale points (identified by the scale's labelled endpoints).
LABEL_BANK: dict[tuple[str, ...], dict[int, str]] = {
    ("None", "1 child", "2 children"): {
        3: "3 children",
        4: "4 children",
        5: "5 children",
        6: "6 children",
        7: "7 children",
        8: "8 children",
        9: "9 children",
        10: "10 children",
    },
    ("Yes, they definitely should", "No, they definitely should not"): {
        2: "Yes, they should",
        3: "Yes, they probably should",
        4: "Undecided",
        5: "No, they probably should not",
        6: "No, they should not",
    },
    (
        "You cannot rely on either of them",
        "You can definitely rely more on close relatives",
        "You can rely equally on both",
        "You can definitely rely more on close friends",
    ): {
        2: "You can rely more on close relatives",
        3: "You can probably rely more on close relatives",
        5: "You can probably rely more on close friends",
        6: "You can rely more on close friends",
    },
    (
        "Close relatives should definitely be the most important",
        "Both should be equally important",
        "Close friends should definitely be the most important",
    ): {
        2: "Close relatives should be the most important",
        3: "Close relatives should probably be the most important",
        5: "Close friends should probably be the most important",
        6: "Close friends should be the most important",
    },
}

BRACKETS = {
    "[COUNTRY's]": "{country}'s",
    "[COUNTRY]": "{country}",
    "[CNTRY]": "{country}",
    "[EU/ USMCA/ etc.]": "{association}",
}

COUNTRY_NOTE = re.compile(r"^([A-Z]{2}(?:\s*,\s*[A-Z]{2})*)(?:\s+also)?\s*:\s*")
NUMBER_PREFIX = re.compile(r"^-?\d+\.\s*")
PLACEHOLDER_RE = re.compile(r"\{[^}]*\}")


def _fill_placeholders(text: str, fields: dict[str, str], where: str) -> str:
    name = fields["country"]
    values = {
        "{country}'s": name + ("'" if name.endswith("s") else "'s"),
        **{f"{{{k}}}": v for k, v in fields.items()},
    }
    for bracket, brace in BRACKETS.items():
        text = text.replace(bracket, brace)
    for placeholder, value in values.items():
        if placeholder in text:
            text = text.replace(placeholder, value)
    text = re.sub(r"(^|[.?!…]\s+)the ", lambda m: m.group(1) + "The ", text)
    leftover = PLACEHOLDER_RE.findall(text)
    if leftover:
        raise ValueError(f"{where}: unfilled placeholder(s) {leftover} in {text!r}")
    return text


def _unlabelled(text: object) -> bool:
    return not isinstance(text, str) or not re.search(r"[A-Za-z]", text)


def _fill_labels(codes: list[int], texts: list[object]) -> list[object]:
    key = tuple(t for t in texts if not _unlabelled(t))
    fixes = LABEL_BANK.get(key, {})
    return [fixes.get(c, t) if _unlabelled(t) else t for c, t in zip(codes, texts)]


def _country_specific_text(label: str, country: str) -> str:
    text = re.sub(r"^-?\d+\.\s+", "", str(label)).strip()
    m = re.fullmatch(r"(.*?)\s*\(([A-Z]{2}(?:\s*,\s*[A-Z]{2})*)\s*:\s*(.*)\)", text)
    if m:
        return m[3] if country in re.split(r"\s*,\s*", m[2]) else m[1]
    kept: list[str] = []
    ours: str | None = None
    for part in re.split(r";|(?<![A-Z]{2}),\s*(?=[A-Z]{2}(?:\s*,\s*[A-Z]{2})*\s*:)", text):
        part = part.strip()
        cm = COUNTRY_NOTE.match(part)
        if not cm:
            kept.append(part)
        elif country in re.split(r"\s*,\s*", cm[1]):
            ours = part[cm.end() :]
            if re.match(r"(incl|\+|also)", ours, re.I):
                ours = f"{kept[0]}, {ours.lstrip('+ ')}" if kept else ours
    return ours if ours is not None else "; ".join(kept)


def _offered_options(labels: dict[int, str], country: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for k, t in sorted(labels.items()):
        if k < 0:
            continue
        text = NUMBER_PREFIX.sub("", t)
        cm = COUNTRY_NOTE.match(text)
        if cm:
            if country not in re.split(r"\s*,\s*", cm[1]):
                continue
            text = text[cm.end() :]
        resolved = _country_specific_text(text, country)
        if re.fullmatch(r"not\s+used", resolved.strip(), re.I):
            continue
        out[int(k)] = resolved
    return out


def _read_missing_codes(path: Path) -> dict[str, set[int]]:
    missing: dict[str, set[int]] = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        match = re.match(r"\s*(\S+)\s*\((.*)\)", line.strip())
        if not match:
            continue
        codes: set[int] = set()
        expr = match.group(2).upper().replace("LOWEST", "-999").replace("HIGHEST", "99999")
        for part in expr.split(","):
            if "THRU" in part:
                low, high = part.split("THRU")
                codes |= set(range(int(low), int(high) + 1))
            elif part.strip():
                codes.add(int(float(part)))
        missing[match.group(1)] = codes
    return missing


def _find_wave_dta(wave_dir: Path) -> Path:
    raw_dir = wave_dir / "raw"
    dtas = sorted(raw_dir.glob("*.dta"))
    if len(dtas) != 1:
        raise FileNotFoundError(f"Expected exactly 1 .dta file in {raw_dir}, found {dtas}")
    return dtas[0]


def _load_country_fields(data_dir: Path, country: str) -> dict[str, str]:
    countries = pd.read_csv(data_dir / "issp_countries.csv", keep_default_na=False).set_index("c_alphan")
    if country not in countries.index:
        raise KeyError(f"Country {country!r} not found in issp_countries.csv")
    row = countries.loc[country, ["country", "nationality", "parliament"]].to_dict()
    return {**row, "association": "the United States-Mexico-Canada Agreement (USMCA)"}


def _apply_label_fixes(
    wave_dir: Path,
    value_labels: dict[str, dict[int, str]],
) -> None:
    fixes_path = wave_dir / "label_fixes.csv"
    if not fixes_path.exists():
        return
    for fix in pd.read_csv(fixes_path, keep_default_na=False).itertuples():
        if fix.variable not in value_labels:
            raise KeyError(f"{fixes_path}: variable {fix.variable!r} not found in .dta")
        vl = value_labels[fix.variable]
        code = int(fix.code)
        if code not in vl:
            raise KeyError(f"{fixes_path}: code {code} not found in value labels for {fix.variable!r}")
        old = vl[code]
        if NUMBER_PREFIX.sub("", old) != NUMBER_PREFIX.sub("", fix.data_label):
            raise ValueError(
                f"{fixes_path}: label mismatch for {fix.variable} {code}: got {old!r}, expected {fix.data_label!r}"
            )
        prefix = NUMBER_PREFIX.match(old)
        new = (prefix.group(0) if prefix else "") + NUMBER_PREFIX.sub("", fix.fixed_label)
        value_labels[fix.variable] = {**vl, code: new}


def _apply_question_overrides(
    wave_dir: Path,
    bank: pd.DataFrame,
) -> pd.DataFrame:
    overrides_path = wave_dir / "question_overrides.json"
    if not overrides_path.exists():
        return bank
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    if not isinstance(overrides, dict):
        raise TypeError(f"{overrides_path}: expected JSON object, got {type(overrides).__name__}")
    bank = bank.copy()
    key_to_idx = {k: idx for idx, k in enumerate(bank["_row_key"])}
    for row_key, ov in overrides.items():
        if row_key not in key_to_idx or not isinstance(ov, dict):
            continue
        idx = key_to_idx[row_key]
        if "stem" in ov and ov["stem"]:
            bank.at[idx, "question"] = str(ov["stem"]).strip()
        if "options" in ov and isinstance(ov["options"], dict):
            opts = list(bank.at[idx, "answer_text"])
            for raw_opt, new_opt in ov["options"].items():
                if raw_opt not in opts:
                    raise ValueError(
                        f"{overrides_path}: option {raw_opt!r} not found in {row_key} options {opts!r}"
                    )
                opts = [str(new_opt) if o == raw_opt else o for o in opts]
            bank.at[idx, "answer_text"] = opts
    return bank


def _load_wave_audit(wave_dir: Path) -> pd.DataFrame:
    audit_path = wave_dir / "question_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing question_audit.csv: {audit_path}")

    df = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
    required = {"catalog_idx", "QuestionID", "include", "kev_type", "reversed", "QuestionText", "data_label"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"{audit_path}: missing required columns {missing_cols}")

    rev_map = {"True": True, "False": False}
    types: list[str] = []
    revs: list[bool] = []
    for row in df.to_dict("records"):
        inc = row["include"]
        if inc not in ("True", "False"):
            raise ValueError(f"{audit_path}: bad include={inc!r} on {row['QuestionID']!r}")
        kt = row["kev_type"]
        if inc == "True":
            if kt not in ("score", "choice"):
                raise ValueError(f"{audit_path}: included {row['QuestionID']!r} has bad kev_type={kt!r}")
            types.append(kt)
        else:
            types.append("exclude")

        rv = row["reversed"]
        if rv not in rev_map:
            raise ValueError(f"{audit_path}: bad reversed={rv!r} on {row['QuestionID']!r}")
        revs.append(rev_map[rv])

    return pd.DataFrame(
        {
            "_row_key": [f"{c}:{q}" for c, q in zip(df["catalog_idx"], df["QuestionID"])],
            "v": df["QuestionID"],
            "type": types,
            "reversed": revs,
            "question": df["QuestionText"],
            "options": df["Options_preview"] if "Options_preview" in df.columns else "",
            "data_label": df["data_label"],
            "drop_options": df["drop_options"] if "drop_options" in df.columns else "",
        }
    )


def _check_year_country(year: int | str, country: str) -> int:
    yr = int(year)
    if yr not in ISSP_YEARS:
        raise ValueError(f"Unsupported ISSP year {year!r}; expected one of {ISSP_YEARS}")
    if country != "US":
        raise ValueError(f"Standardized persona export currently supports country='US' only, got {country!r}")
    return yr


def _load_wave_answers(
    wave_dir: Path,
    country: str,
) -> tuple[Path, dict[str, str], dict[str, dict[int, str]], pd.DataFrame]:
    """Read one wave's `.dta`: variable labels, fixed value labels, and the country's rows with missing codes made negative."""
    dta_path = _find_wave_dta(wave_dir)

    with StataReader(dta_path) as reader:
        var_labels = reader.variable_labels()
        label_sets = reader.value_labels()
        value_labels: dict[str, dict[int, str]] = {
            v: {int(k): str(lbl) for k, lbl in label_sets[s].items()} if s in label_sets else {}
            for v, s in zip(reader._varlist, reader._lbllist)
        }

    _apply_label_fixes(wave_dir, value_labels)

    raw = pd.read_stata(dta_path, convert_categoricals=False)

    # Shift GESIS non-negative missing codes (from *_missing.txt) to negative sentinel codes.
    missing_path = dta_path.with_name(f"{dta_path.stem}_missing.txt")
    if missing_path.exists():
        for col, codes in _read_missing_codes(missing_path).items():
            pos_codes = {k for k in codes if k >= 0}
            if col not in raw.columns or not pos_codes or raw[col].dtype.kind not in "iuf":
                continue
            s = raw[col].astype("float64")
            raw[col] = s.where(~s.isin(pos_codes), -1000 - s)
            value_labels[col] = {
                (-1000 - int(k) if k in pos_codes else int(k)): t for k, t in value_labels[col].items()
            }

    # Filter to target country using Stata categorical `c_alphan`.
    c_alphan_col = pd.read_stata(dta_path, columns=["c_alphan"])["c_alphan"]
    us_mask = c_alphan_col == country
    raw_us = raw.loc[us_mask].reset_index(drop=True)
    if raw_us["CASEID"].isna().any():
        raise ValueError(f"{dta_path}: missing CASEID values in {country} rows")
    return dta_path, var_labels, value_labels, raw_us


def _question_columns(raw_us: pd.DataFrame) -> list[str]:
    return [c for c in raw_us.columns if re.fullmatch(r"[vV]\d+", c)]


def _all_options(
    v: str,
    is_score: bool,
    value_labels: dict[str, dict[int, str]],
    raw_us: pd.DataFrame,
    country: str,
) -> dict[int, object]:
    """Offered options for `v`, plus unlabelled interior scale points (score) and answered-but-unlabelled codes (NaN text)."""
    opts: dict[int, object] = dict(_offered_options(value_labels[v], country))
    valid_vals = raw_us.loc[raw_us[v] >= 0, v].dropna()
    given = {int(x) for x in valid_vals}
    labelled = {int(k) for k in value_labels[v] if k >= 0}
    span = set(range(min(opts), max(opts) + 1)) if opts and is_score else set()
    for c in (span - labelled) | (given - set(opts)):
        opts[c] = float("nan")
    return dict(sorted(opts.items()))


def _export_problem(
    v: str,
    is_score: bool,
    question_cols: set[str],
    value_labels: dict[str, dict[int, str]],
    raw_us: pd.DataFrame,
    country: str,
) -> str | None:
    """Why `v` cannot be exported if included with this kev_type; None when it can.

    Only properties of the question block the export. A country with no valid answers is not one:
    the question is left out of that country's parquet and reported (see `export_issp_wave`).
    """
    if v not in question_cols:
        return f"{v} is not a question column in the .dta"
    n_opts = len(_all_options(v, is_score, value_labels, raw_us, country))
    if n_opts > MAX_OPTIONS:
        return f"{n_opts} answer options (limit {MAX_OPTIONS})"
    return None


def issp_question_checks(
    year: int | str,
    *,
    country: str = "US",
    data_dir: Path = DATA_DIR,
) -> dict[str, dict]:
    """For every `question_audit.csv` row key: why export would fail if included as choice / as score,
    and the number of valid answers per country (None when the question is not in the .dta)."""
    yr = _check_year_country(year, country)
    wave_dir = data_dir / f"issp{yr}"
    bank = _load_wave_audit(wave_dir)
    _, _, value_labels, raw_us = _load_wave_answers(wave_dir, country)
    question_cols = set(_question_columns(raw_us))
    checks: dict[str, dict] = {}
    for row_key, v, q_type in zip(bank["_row_key"], bank["v"], bank["type"]):
        n_valid = int((raw_us[v] >= 0).sum()) if v in question_cols else None
        relabeled = False
        if v in question_cols:
            opts = _all_options(v, q_type == "score", value_labels, raw_us, country)
            codes = list(opts)
            after_vals = _fill_labels(codes, list(opts.values()))
            relabeled = is_option_relabeled(codes, after_vals)
        checks[row_key] = {
            "export_problem_if_included": {
                "choice": _export_problem(v, False, question_cols, value_labels, raw_us, country),
                "score": _export_problem(v, True, question_cols, value_labels, raw_us, country),
            },
            "n_valid_by_country": {country: n_valid},
            "option_relabeled": relabeled,
        }
    return checks


def format_not_fielded(year: int, country: str, not_fielded: list[str]) -> str:
    """One-line report of included questions left out of a country's parquet because it has no valid answers."""
    if not not_fielded:
        return f"issp{year}_{country}: every included question has {country} answers"
    return (
        f"issp{year}_{country}: {len(not_fielded)} included question(s) not fielded in {country}, "
        f"left out of the parquet: {', '.join(not_fielded)}"
    )


def export_issp_wave(
    year: int | str,
    *,
    country: str = "US",
    data_dir: Path = DATA_DIR,
    write: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Export one ISSP wave to `issp{year}_{country}_long.parquet` and `questions_verified.csv`.

    Returns `(long, verified_bank, not_fielded)`: `not_fielded` lists included questions with no valid
    answers for `country` (missing codes only, or only answers in `drop_options`); they are absent from
    the parquet. This is a fact about the country's sample, not about the question, so it never blocks.
    """
    yr = _check_year_country(year, country)
    wave_dir = data_dir / f"issp{yr}"
    fields = _load_country_fields(data_dir, country)
    bank = _load_wave_audit(wave_dir)
    _, var_labels, value_labels, raw_us = _load_wave_answers(wave_dir, country)
    caseids = raw_us["CASEID"].astype("int64")
    raw_q_cols = _question_columns(raw_us)
    question_cols = set(raw_q_cols)

    included = bank[bank["type"] != "exclude"]
    dup_vs = sorted(set(included.loc[included["v"].duplicated(), "v"]))
    if dup_vs:
        raise ValueError(f"issp{yr}: QuestionID included more than once in question_audit.csv: {dup_vs}")

    failing: dict[str, str] = {}
    for row_key, v, q_type in zip(included["_row_key"], included["v"], included["type"]):
        problem = _export_problem(v, q_type == "score", question_cols, value_labels, raw_us, country)
        if problem is not None:
            failing[row_key] = problem
    if failing:
        lines = "\n".join(f"  {row_key}: {problem}" for row_key, problem in failing.items())
        raise ValueError(
            f"issp{yr}: {len(failing)} included question(s) cannot be exported. "
            f"Exclude them in question_audit.csv (audit app: 'Blocks export' filter):\n{lines}"
        )

    score_vs = set(bank.loc[bank["type"] == "score", "v"])
    included_bank_vs = set(included["v"])
    kept_vs = [v for v in raw_q_cols if v in included_bank_vs]
    all_opts = {v: _all_options(v, v in score_vs, value_labels, raw_us, country) for v in kept_vs}

    bank = bank[bank["v"].isin(kept_vs)].reset_index(drop=True)
    bank["question"] = [_fill_placeholders(t, fields, f"issp{yr}.{v}") for v, t in zip(bank["v"], bank["question"])]

    filled_rows = []
    for v in kept_vs:
        opts = {
            c: _fill_placeholders(t, fields, f"issp{yr}.{v}[{c}]") if isinstance(t, str) else t
            for c, t in all_opts[v].items()
        }
        codes = list(opts.keys())
        after_vals = _fill_labels(codes, list(opts.values()))
        after_texts = format_option_texts(codes, after_vals)
        filled_rows.append({"v": v, "answer_code": codes, "answer_text": after_texts})

    bank = bank.merge(pd.DataFrame(filled_rows), on="v", how="inner")

    # Apply drop_options (before option text overrides, matching audit-app semantics).
    dropped_pairs: set[tuple[str, int]] = set()
    for idx in range(len(bank)):
        raw_drop = str(bank.at[idx, "drop_options"]).strip()
        if not raw_drop:
            continue
        drop_set = {s.strip() for s in raw_drop.split("|") if s.strip()}
        codes = list(bank.at[idx, "answer_code"])
        texts = list(bank.at[idx, "answer_text"])
        unknown_drops = sorted(drop_set - set(texts))
        if unknown_drops:
            raise ValueError(
                f"issp{yr}.{bank.at[idx, 'v']}: drop_options {unknown_drops} not in options {texts!r}"
            )
        v_id = str(bank.at[idx, "v"])
        kept_codes: list[int] = []
        kept_texts: list[str] = []
        for c, t in zip(codes, texts):
            if t in drop_set:
                dropped_pairs.add((v_id, int(c)))
            else:
                kept_codes.append(c)
                kept_texts.append(t)
        if len(kept_codes) < 2:
            raise ValueError(f"issp{yr}.{v_id}: drop_options leaves fewer than 2 options ({kept_texts!r})")
        bank.at[idx, "answer_code"] = kept_codes
        bank.at[idx, "answer_text"] = kept_texts

    bank = _apply_question_overrides(wave_dir, bank)

    # Melt substantive answers for kept questions (keeping only non-negative valid numeric codes).
    q_frame = raw_us[kept_vs].copy()
    q_frame.insert(0, "CASEID", caseids)
    long = q_frame.melt(id_vars=["CASEID"], value_vars=kept_vs, var_name="v", value_name="answer_code")
    long = long.dropna(subset=["answer_code"])
    long = long[long["answer_code"] >= 0].copy()
    long["answer_code"] = long["answer_code"].astype("int64")
    if dropped_pairs:
        keep_mask = [(v, int(c)) not in dropped_pairs for v, c in zip(long["v"], long["answer_code"])]
        long = long.loc[keep_mask].copy()
    answered_vs = set(long["v"])
    not_fielded = [v for v in kept_vs if v not in answered_vs]
    long["label"] = [var_labels[v] for v in long["v"]]

    text_of = {
        (r["v"], int(c)): str(t)
        for _, r in bank.iterrows()
        for c, t in zip(r["answer_code"], r["answer_text"])
    }
    missing_pairs = [
        (v, int(c)) for v, c in zip(long["v"], long["answer_code"]) if (v, int(c)) not in text_of
    ]
    if missing_pairs:
        raise ValueError(f"issp{yr}: unmapped (v, answer_code) pairs: {missing_pairs[:5]}")
    long["answer_text"] = [text_of[(v, int(c))] for v, c in zip(long["v"], long["answer_code"])]

    long = long.merge(
        bank[["v", "question", "type", "reversed", "answer_code", "answer_text"]].rename(
            columns={"answer_code": "option_codes", "answer_text": "option_texts"}
        ),
        on="v",
        how="left",
    )

    # Attach standardized personas from persona_loaders + persona_spec.yaml (single source of truth).
    specs = load_persona_spec()
    raw_personas = load_raw_issp_personas(data_dir=data_dir, shared_only=False, years=(yr,))
    std_personas = standardize_dataframe(raw_personas, dataset="issp", specs=specs)
    persona_cols = [c for c in specs if c in std_personas.columns and std_personas[c].notna().any()]

    std_personas = std_personas[["CASEID", *persona_cols]].copy()
    std_personas["CASEID"] = [int(cid.split(":", 1)[1]) for cid in std_personas["CASEID"]]
    if std_personas["CASEID"].duplicated().any():
        raise ValueError(f"issp{yr}: duplicate CASEID in standardized personas")

    missing_cids = set(long["CASEID"].unique()) - set(std_personas["CASEID"].unique())
    if missing_cids:
        raise ValueError(f"issp{yr}: {len(missing_cids)} CASEIDs missing from standardized personas")

    long = std_personas.merge(long, on="CASEID", how="inner")
    ordered_cols = [
        "CASEID",
        *persona_cols,
        "v",
        "label",
        "question",
        "type",
        "reversed",
        "option_codes",
        "option_texts",
        "answer_code",
        "answer_text",
    ]
    long = long[ordered_cols].reset_index(drop=True)

    verified_bank = bank.copy()
    verified_bank["options_old"] = verified_bank["options"]
    verified_bank = verified_bank[
        ["v", "data_label", "question", "options_old", "answer_code", "answer_text", "type", "reversed"]
    ]

    if write:
        long_path = wave_dir / f"issp{yr}_{country}_long.parquet"
        tmp_pq = wave_dir / f".issp{yr}_{country}_long.parquet.tmp"
        long.to_parquet(tmp_pq, index=False)
        os.replace(tmp_pq, long_path)

        verified_path = wave_dir / "questions_verified.csv"
        tmp_csv = wave_dir / ".questions_verified.csv.tmp"
        verified_bank.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, verified_path)

    return long, verified_bank, not_fielded


def export_all_issp_waves(
    years: tuple[int, ...] = ISSP_YEARS,
    *,
    country: str = "US",
    data_dir: Path = DATA_DIR,
    write: bool = True,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """Export all requested ISSP waves."""
    return {yr: export_issp_wave(yr, country=country, data_dir=data_dir, write=write) for yr in years}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export clean ISSP long parquets with standardized personas.")
    parser.add_argument("--year", type=int, help="Single ISSP wave year (2015-2024); omit for all waves.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate without writing files.")
    args = parser.parse_args()

    years = (args.year,) if args.year is not None else ISSP_YEARS
    for yr in years:
        long, _, not_fielded = export_issp_wave(yr, write=not args.dry_run)
        print(format_not_fielded(yr, "US", not_fielded))
        n_choice = long.loc[long["type"] == "choice", "v"].nunique()
        n_score = long.loc[long["type"] == "score", "v"].nunique()
        print(
            f"issp{yr}_US_long.parquet: {len(long):,} rows | "
            f"{long['CASEID'].nunique():,} respondents | "
            f"{long['v'].nunique():,} questions ({n_choice} choice, {n_score} score)"
        )


if __name__ == "__main__":
    main()
