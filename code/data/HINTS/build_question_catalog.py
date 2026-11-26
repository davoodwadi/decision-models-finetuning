#!/usr/bin/env python3
"""Build `code/data/HINTS/question_audit.csv` from official NCI HINTS 7 sources.

Reads:
- `HINTS7_STATA_20250731.zip` (`hints7_public.dta`): answers, Stata variable labels, value labels,
  and valid respondent counts (`code >= 0`).
- `HINTS 7 Public Codebook.pdf`: official section titles, untruncated `Variable Label` stems,
  and exact skip-logic criteria (`Criteria to receive Question`, `Criteria description`).

Strictly includes ONLY discrete single-choice survey questions from Sections A through Q.
Excludes all Section R persona/demographic items, geographic/admin/sampling/weight columns,
derived recodes, check-all-that-apply checkbox grids, and numeric/open-text items.
"""

from __future__ import annotations

import io
from pathlib import Path
import re
import sys
import zipfile

import pandas as pd
from pandas.io.stata import StataReader
import pymupdf

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data.persona_loaders import (
    HINTS7_DTA_MEMBER,
    HINTS7_MISSING_CODES,
    HINTS7_PERSONA_RAW_COLS,
    HINTS7_ZIP_REL,
)

HINTS_DIR = Path(__file__).resolve().parent
DATA_DIR = HINTS_DIR.parent
CODEBOOK_PDF = HINTS_DIR / "HINTS 7 Public Codebook.pdf"
OUT_CSV = HINTS_DIR / "question_audit.csv"

SECTION_ITEM_RE = re.compile(r'^"?([A-Q])(\d+[A-Za-z]?)\.\s*(.*)')
ITEM_PREFIX_RE = re.compile(r'^"?([A-Q]\d+[A-Za-z]?)\.\s*')
WS_RE = re.compile(r"\s+")

# Pure ordinal response scales (exact sorted positive Stata option tuples) -> default kev_type='score'.
# Scales containing trailing opt-outs ('Don't know', 'Not applicable', 'I do not use...') stay 'choice'
# until the reviewer decides whether to drop the opt-out in `drop_options`.
PURE_ORDINAL_SETS: set[tuple[str, ...]] = {
    ("Strongly agree", "Somewhat agree", "Somewhat disagree", "Strongly disagree"),
    ("Strongly agree", "Somewhat agree", "Neither agree nor disagree", "Somewhat disagree", "Strongly disagree"),
    ("A lot", "Some", "A little", "Not at all"),
    ("Always", "Usually", "Sometimes", "Never"),
    ("Never", "Rarely", "Sometimes", "Usually", "Always"),
    ("Never", "Rarely", "Often", "Very Often"),
    ("Almost everyday", "At least once a week", "A few times a month", "Less than once a month", "Never"),
    ("More than once per day", "About once per day", "A few times a week", "Less than once per week", "Rarely", "Never"),
    ("Very often true", "Often true", "Sometimes true", "Rarely true", "Never true"),
    ("Nearly everyday", "More than half the days", "Several days", "Not at all"),
    ("Excellent", "Very good", "Good", "Fair", "Poor"),
    ("Completely confident", "Very confident", "Somewhat confident", "A little confident", "Not confident at all"),
    ("Very", "Somewhat", "A little", "Not at all"),
    ("Very comfortable", "Somewhat comfortable", "Somewhat uncomfortable", "Very uncomfortable"),
    ("Strongly oppose", "Oppose", "Neither support nor oppose", "Support", "Strongly support"),
    ("Extremely satisfied", "Very satisfied", "Somewhat satisfied", "Not very satisfied", "Not at all satisfied"),
    ("Very willing", "Somewhat willing", "Somewhat unwilling", "Very unwilling"),
    ("Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult"),
    ("Very well", "Well", "Fairly well", "Poorly"),
    ("Not at all valuable", "A little valuable", "Somewhat valuable", "Very valuable"),
    ("A lot worse", "A little worse", "No impact on our interaction", "A little better", "A lot better"),
    ("Everyday", "Some days", "Not at all"),
    ("Never", "1 or 2 times", "More than 2 times"),
    ("Never", "1 or 2 times", "3 to 5 times", "6 to 10 times", "11 or more times"),
    ("0", "1 to 2 times", "3 to 5 times", "6 to 9 times", "10 or more times"),
    ("None", "1 time", "2 times", "3 times", "4 times", "5-9 times", "10 or more times"),
    (
        "None",
        "1 day per week",
        "2 days per week",
        "3 days per week",
        "4 days per week",
        "5 days per week",
        "6 days per week",
        "7 days per week",
    ),
    (
        "None",
        "1/2 cup or less",
        "1/2 cup to less than 1 cup",
        "1 to less than 2 cups",
        "2 to less than 3 cups",
        "3 to less than 4 cups",
        "4 or more cups",
    ),
}

OPTOUT_PATTERNS = (
    "don't know",
    "not sure",
    "not applicable",
    "i do not use",
    "i don't have",
    "i do not have",
    "i have never",
    "i did not",
    "i do not buy",
    "other (specify)",
)


def _collapse(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def _parse_codebook_meta(pdf_path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Return `(section_titles, var_meta)` from `HINTS 7 Public Codebook.pdf`."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing codebook PDF: {pdf_path}")

    section_titles: dict[str, str] = {}
    var_meta: dict[str, dict[str, str]] = {}

    with pymupdf.open(pdf_path) as doc:
        # TOC lives on pages 2..20 (1-based).
        for pno in range(1, min(25, len(doc))):
            txt = doc[pno].get_text("text")
            for m in re.finditer(r"^SECTION ([A-Q]):\s*(.+?)\s*$", txt, re.M):
                sec = m.group(1)
                section_titles[sec] = f"{sec}_{_collapse(m.group(2))}"

        for pno in range(len(doc)):
            lines: list[str] = []
            for blk in doc[pno].get_text("dict")["blocks"]:
                for ln in blk.get("lines", []):
                    t = "".join(s["text"] for s in ln["spans"]).strip()
                    if t:
                        lines.append(t)
            vname = None
            for t in lines:
                m = re.match(r"^Variable Name:\s*(\S+)$", t)
                if m:
                    vname = m.group(1)
                    break
            if vname is None:
                continue

            i_lbl = next((i for i, t in enumerate(lines) if t.startswith("Variable Label:")), None)
            i_fmt = next((i for i, t in enumerate(lines) if t.startswith("Variable Format:")), None)
            i_crit = next((i for i, t in enumerate(lines) if t.startswith("Criteria to receive Question:")), None)
            i_cdesc = next((i for i, t in enumerate(lines) if t.startswith("Criteria description:")), None)
            i_back = next((i for i, t in enumerate(lines) if t == "Back to Table of Contents"), None)
            if None in (i_lbl, i_fmt, i_crit, i_cdesc, i_back):
                continue

            vlabel_raw = _collapse(
                " ".join([lines[i_lbl][len("Variable Label:") :].strip(), *lines[i_lbl + 1 : i_fmt]])
            )
            crit = _collapse(
                " ".join([lines[i_crit][len("Criteria to receive Question:") :].strip(), *lines[i_crit + 1 : i_cdesc]])
            )
            cdesc = _collapse(
                " ".join([lines[i_cdesc][len("Criteria description:") :].strip(), *lines[i_cdesc + 1 : i_back]])
            )
            var_meta[vname] = {
                "variable_label": vlabel_raw,
                "criteria": crit,
                "criteria_desc": cdesc,
            }

    expected_sections = {chr(c) for c in range(ord("A"), ord("Q") + 1)}
    missing_secs = sorted(expected_sections - set(section_titles))
    if missing_secs:
        raise ValueError(f"{pdf_path}: missing TOC section headers for {missing_secs}")
    return section_titles, var_meta


def _strip_item_prefix(text: str) -> str:
    cleaned = ITEM_PREFIX_RE.sub("", text, count=1).strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    return _collapse(cleaned)


def build_catalog() -> pd.DataFrame:
    zip_path = DATA_DIR / HINTS7_ZIP_REL
    if not zip_path.is_file():
        raise FileNotFoundError(f"Official HINTS 7 Stata archive not found: {zip_path}")

    section_titles, var_meta = _parse_codebook_meta(CODEBOOK_PDF)

    with zipfile.ZipFile(zip_path) as zf:
        dta_bytes = zf.read(HINTS7_DTA_MEMBER)

    with StataReader(io.BytesIO(dta_bytes)) as reader:
        var_labels = reader.variable_labels()
        label_sets = reader.value_labels()
        col_to_lbl = dict(zip(reader._varlist, reader._lbllist, strict=True))

    raw = pd.read_stata(io.BytesIO(dta_bytes), convert_categoricals=False)

    records: list[dict[str, object]] = []
    for col in raw.columns:
        if col in HINTS7_PERSONA_RAW_COLS:
            continue
        data_label = str(var_labels[col])
        m = SECTION_ITEM_RE.match(data_label)
        if not m:
            continue
        sec = m.group(1)
        qnum = f"{sec}{m.group(2)}"

        lbl_name = col_to_lbl[col]
        if not lbl_name or lbl_name not in label_sets:
            continue
        pos_opts = {int(k): str(v) for k, v in sorted(label_sets[lbl_name].items()) if int(k) >= 0}
        opt_values = tuple(pos_opts.values())
        if len(pos_opts) < 2:
            continue
        if set(opt_values) in ({"Selected", "Not selected"}, {"Selected", "Not Selected"}):
            continue

        # Validate negative missing codes strictly.
        s = raw[col].dropna()
        neg = s[s < 0]
        unexpected_neg = set(int(x) for x in neg.unique()) - HINTS7_MISSING_CODES
        if unexpected_neg:
            raise ValueError(f"hints7.{col}: unexpected negative code(s) {sorted(unexpected_neg)}")
        n_answered = int((s >= 0).sum())

        if col not in var_meta:
            raise KeyError(f"hints7.{col}: missing from {CODEBOOK_PDF.name}")
        meta = var_meta[col]
        qtext = _strip_item_prefix(meta["variable_label"])
        if not qtext:
            raise ValueError(f"hints7.{col}: empty question stem extracted from codebook")

        is_score = opt_values in PURE_ORDINAL_SETS
        kev_type = "score" if is_score else "choice"
        include = sec != "Q"

        notes: list[str] = []
        if meta["criteria"] and meta["criteria"] != "N/A":
            notes.append(f"Skip-gated ({meta['criteria']}): {meta['criteria_desc']}.")
        else:
            notes.append("Full sample.")
        if is_score:
            notes.append("Ordered response scale.")
        else:
            optouts = [o for o in opt_values if any(p in o.lower() for p in OPTOUT_PATTERNS)]
            if optouts:
                notes.append(f"Has opt-out/non-substantive option(s): {'; '.join(optouts)}.")
        if sec == "Q":
            notes.append("Excluded by default: personal/family cancer diagnosis history (Section Q).")

        records.append(
            {
                "catalog_idx": len(records),
                "QuestionID": col,
                "family": section_titles[sec],
                "BlockName": qnum,
                "n_answered": n_answered,
                "include": str(include),
                "kev_type": kev_type,
                "drop_options": "",
                "review_note": " ".join(notes),
                "QuestionText": qtext,
                "Options_preview": " | ".join(opt_values),
                "data_label": data_label,
                "codes": " | ".join(str(c) for c in pos_opts),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("hints7: built empty question catalog")
    if df["QuestionID"].duplicated().any():
        raise ValueError("hints7: duplicate QuestionID in built catalog")
    # Double-check zero overlap with persona columns.
    leaked = sorted(set(df["QuestionID"]) & HINTS7_PERSONA_RAW_COLS)
    if leaked:
        raise ValueError(f"hints7: persona column(s) leaked into question catalog: {leaked}")
    return df


def main() -> None:
    df = build_catalog()
    df.to_csv(OUT_CSV, index=False)
    n_inc = int((df["include"] == "True").sum())
    n_score = int(((df["include"] == "True") & (df["kev_type"] == "score")).sum())
    n_choice = int(((df["include"] == "True") & (df["kev_type"] == "choice")).sum())
    print(f"Wrote {OUT_CSV}: {len(df)} questions ({n_inc} included: {n_score} score, {n_choice} choice; {len(df) - n_inc} excluded)")


if __name__ == "__main__":
    main()
