#!/usr/bin/env python3
"""Build `code/data/SHED/question_audit.csv` from official Federal Reserve SHED 2025 sources.

Reads:
- `SHED_public_use_data_2025_(STATA).zip` (`public2025.dta`): answers, Stata variable labels,
  value labels, hot-deck imputation flags (`*_iflag`), and valid non-imputed respondent counts.
- `SHED_2025codebook.pdf`: untruncated variable question stems.

Strictly includes ONLY discrete survey questions from the 2025 SHED questionnaire.
Excludes all KnowledgePanel profile variables (`pp*`), persona/demographic columns in
`SHED2025_PERSONA_RAW_COLS`, imputation flags (`*_iflag`), weights/admin/DOV/derived variables,
continuous dollar open-ends, and single-option check-all-that-apply (`Not selected`) checkboxes.
"""

from __future__ import annotations

import io
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import pandas as pd
from pandas.io.stata import StataReader

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data.persona_loaders import (
    SHED2025_DTA_MEMBER,
    SHED2025_MISSING_CODES,
    SHED2025_PERSONA_RAW_COLS,
    SHED2025_ZIP_REL,
)

SHED_DIR = Path(__file__).resolve().parent
DATA_DIR = SHED_DIR.parent
CODEBOOK_PDF = SHED_DIR / "SHED_2025codebook.pdf"
OUT_CSV = SHED_DIR / "question_audit.csv"

WS_RE = re.compile(r"\s+")
PREFIX_RE = re.compile(r"^([A-Za-z]+)")

# Matrix items in SHED codebook are formatted "<sub-item> - <question stem>" (or en-dash).
MATRIX_SPLIT_RE = re.compile(
    r"^(.+?)\s+[-–]\s+((?:\[?Think|How|Do|Did|Are|Is|In|During|Compared|Suppose|Have|Were|Would|"
    r"Which|What|You|Has|Was|When|Still|For|Please)\b.*)$"
)

# Non-survey admin, weight, sampling, experimental-flag, derived-summary, and ACS profile vars.
SHED2025_NON_SURVEY_COLS: frozenset[str] = frozenset({
    "shedid",
    "duration",
    "weight",
    "weight_pop",
    "panel_weight",
    "panel_weight_pop",
    "xlaptop",
    "xspanish",
    "field_month",
    "field_day",
    "control",
    "year",
    "DOV_X11",
    "DOV_D36_AMOUNT",
    "dov_ef7",
    "pay_casheqv",
    "atleast_okay",
    "X11_none",
    "BK58_no",
    "ACSNET2_mob",
    "ACSNET2_hi",
    "ACSNET2_sat",
    "ACSNET2_du",
    "ACSNET2_oth",
})

SECTION_NAMES: dict[str, str] = {
    "L": "L_Living_Arrangements",
    "B": "B_Financial_Wellbeing",
    "X": "X_Financial_Challenges",
    "CG": "CG_Childcare_and_Caregiving",
    "D": "D_Employment_and_AI",
    "GH": "GH_Housing_and_Insurance",
    "R": "R_Renting_and_Moving",
    "M": "M_Mortgage",
    "ND": "ND_Natural_Disasters",
    "BK": "BK_Banking_and_Fraud",
    "A": "A_Credit_Applications",
    "C": "C_Credit_Cards",
    "BNPL": "BNPL_Buy_Now_Pay_Later",
    "S": "S_Cryptocurrency",
    "SL": "SL_Student_Loans",
    "K": "K_Retirement",
    "DC": "DC_Investment_Comfort",
    "FL": "FL_Financial_Risk",
    "I": "I_Income_and_Spending",
    "FS": "FS_Financial_Support",
    "INF": "INF_Inflation_and_Prices",
    "EF": "EF_Emergency_Funds",
    "FD": "FD_Food_Sufficiency",
    "E": "E_Medical_and_Expenses",
}

# Pure ordinal response scales (exact clean sorted positive Stata option tuples) -> default kev_type='score'.
PURE_ORDINAL_SETS: set[tuple[str, ...]] = {
    ("Finding it difficult to get by", "Just getting by", "Doing okay", "Living comfortably"),
    ("Much worse off", "Somewhat worse off", "About the same", "Somewhat better off", "Much better off"),
    ("Poor", "Only fair", "Good", "Excellent"),
    ("Not a concern", "Minor concern", "Major concern"),
    ("Never", "Rarely", "Sometimes", "Often", "Always"),
    ("Always", "Often", "Sometimes", "Rarely", "Never"),
    ("Agree", "Neither agree nor disagree", "Disagree"),
    ("Better", "About the same", "Worse"),
    ("None", "Some", "All"),
    ("None", "1 or 2 days", "3 or 4 days", "5 or more days"),
    ("Very likely", "Somewhat likely", "Not that likely", "Not at all likely"),
    ("No", "Yes, slightly", "Yes, moderately", "Yes, substantially"),
    ("Very confident", "Somewhat confident", "Not confident"),
    ("Never carried an unpaid balance (always pay in full)", "Once", "Some of the time", "Most or all of the time"),
    ("Very comfortable", "Mostly comfortable", "Slightly comfortable", "Not comfortable"),
    ("0. Not at all willing to take financial risks", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10. Very willing to take financial risks"),
    ("Roughly the same amount each month", "Occasionally varies from month to month", "Varies quite often from month to month"),
    ("Less than your income", "The same as your income", "More than your income"),
    ("Decreased", "About the same", "Increased"),
    ("Much worse", "Somewhat worse", "Little or no effect", "Somewhat better", "Much better"),
    ("Under $100", "$100 to $499", "$500 to $999", "$1,000 to $1,999", "$2,000 or more"),
    ("Under $100", "$100 to $499", "$500 to $999", "$1,000 to $1,999", "$2,000 to $4,999", "$5,000 or more"),
    ("Completely", "Very well", "Somewhat", "Very little", "Not at all"),
    ("Enough of the kinds of food we wanted to eat", "Enough, but not always the kinds of food we wanted to eat", "Sometimes not enough to eat", "Often not enough to eat"),
    ("$1 to $499", "$500 to $999", "$1,000 to $1,999", "$2,000 to $4,999", "$5,000 or higher"),
    ("None", "1 to 9 hours", "10 to 19 hours", "20 to 39 hours", "40 hours or more"),
    ("1 to 9 hours", "10 to 19 hours", "20 to 39 hours", "40 hours or more"),
    ("Daily", "Several days per week", "Several days per month", "Once per month", "Less than once per month"),
    ("Multiple times per day", "Daily", "Several days per week", "Weekly", "Less than one day per week"),
    ("less than 1 hour", "1 to 9 hours", "10 to 39 hours", "40 to 79 hours", "80 hours or more"),
    ("59 or younger", "60 or 61", "62, 63, or 64", "65 or 66", "67, 68, or 69", "70 or older"),
    ("1", "2", "3", "4", "5 or more"),
}

# Variables excluded by default with explicit reason (still visible in Audit UI for override).
DEFAULT_EXCLUDE_REASONS: dict[str, str] = {
    "D36B": "Randomized pay-change wording injected via DOV_D36_AMOUNT.",
    "GH14A": "Payment period modifier for open-ended dollar item GH14.",
    "D1G": "Current student enrollment (overlaps demographic/employment status).",
    "D1I": "Self-identified retired status (overlaps demographic/employment status).",
}

# Random 50% split-ballot items (not skip-gated on prior answers).
RANDOM_SPLIT_PREFIXES = ("X12_",)
RANDOM_SPLIT_VARS = frozenset({"EF7", "EF7A"})

# Minimum non-imputed sample size for default inclusion when not a random split.
DEFAULT_MIN_N = 10_000


def _collapse(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def _parse_codebook_headers(pdf_path: Path) -> dict[str, str]:
    """Parse untruncated variable descriptions from `SHED_2025codebook.pdf`."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing SHED 2025 codebook PDF: {pdf_path}")

    txt = subprocess.check_output(["pdftotext", "-layout", str(pdf_path), "-"], text=True)
    lines = txt.replace("\x0c", "\n").splitlines()
    entries: dict[str, str] = {}
    i = 0
    while i < len(lines):
        if re.match(r"^\s*-{50,}\s*$", lines[i]):
            while i + 1 < len(lines) and re.match(r"^\s*-{50,}\s*$", lines[i + 1]):
                i += 1
            j = i + 1
            hdr: list[str] = []
            while j < len(lines) and not re.match(r"^\s*-{50,}\s*$", lines[j]):
                if lines[j].strip():
                    hdr.append(lines[j])
                j += 1
            if hdr and j < len(lines):
                m = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(.*)$", hdr[0])
                if m and m.group(1) not in {"CODEBOOK", "ANALYSIS", "DEMOGRAPHIC", "VARIABLE", "The"}:
                    var = m.group(1)
                    if var not in entries:
                        entries[var] = _collapse(" ".join([m.group(2).strip(), *(x.strip() for x in hdr[1:])]))
            i = j
        else:
            i += 1
    return entries


def _clean_question_text(raw_text: str) -> str:
    """Reorder SHED matrix '<item> - <stem>' into natural '<stem> <item>' reading order."""
    text = _collapse(raw_text)
    m = MATRIX_SPLIT_RE.match(text)
    if not m:
        return text
    item, stem = m.group(1).strip(), m.group(2).strip()
    if stem.endswith((":", "?", ".")):
        return f"{stem} {item}"
    return f"{stem}: {item}"


def _clean_positive_options(
    col: str,
    raw_opts: dict[int, str],
    used_codes: set[int],
) -> dict[int, str]:
    """Keep non-negative option labels, dropping zero-count duplicate-label artifacts (e.g. D22 code 4)."""
    clean: dict[int, str] = {}
    seen_labels: set[str] = set()
    for code, label in sorted(raw_opts.items()):
        if code < 0:
            continue
        if label in seen_labels and code not in used_codes:
            continue
        clean[code] = label
        seen_labels.add(label)
    return clean


def build_catalog() -> pd.DataFrame:
    zip_path = DATA_DIR / SHED2025_ZIP_REL
    if not zip_path.is_file():
        raise FileNotFoundError(f"Official SHED 2025 Stata archive not found: {zip_path}")

    codebook_headers = _parse_codebook_headers(CODEBOOK_PDF)

    with zipfile.ZipFile(zip_path) as zf:
        dta_bytes = zf.read(SHED2025_DTA_MEMBER)

    with StataReader(io.BytesIO(dta_bytes)) as reader:
        var_labels = reader.variable_labels()
        label_sets = reader.value_labels()
        col_to_lbl = dict(zip(reader._varlist, reader._lbllist, strict=True))

    raw = pd.read_stata(io.BytesIO(dta_bytes), convert_categoricals=False)
    total_n = len(raw)

    records: list[dict[str, object]] = []
    for col in raw.columns:
        if col.endswith("_iflag") or col.lower().startswith("pp"):
            continue
        if col in SHED2025_PERSONA_RAW_COLS or col in SHED2025_NON_SURVEY_COLS:
            continue

        lbl_name = col_to_lbl[col]
        if not lbl_name or lbl_name not in label_sets:
            continue

        # Validate negative missing codes strictly and mask hot-deck imputed values.
        s = raw[col]
        iflag_col = f"{col}_iflag"
        n_imputed = 0
        if iflag_col in raw.columns:
            flag = raw[iflag_col]
            bad_flags = set(flag.dropna().unique()) - {0, 1}
            if bad_flags:
                raise ValueError(f"shed2025.{iflag_col}: unexpected values {sorted(bad_flags)}")
            n_imputed = int((flag == 1).sum())
            s = s.where(flag != 1)

        s_nonnull = s.dropna()
        neg = s_nonnull[s_nonnull < 0]
        unexpected_neg = set(int(x) for x in neg.unique()) - SHED2025_MISSING_CODES
        if unexpected_neg:
            raise ValueError(f"shed2025.{col}: unexpected negative code(s) {sorted(unexpected_neg)}")
        valid = s_nonnull[s_nonnull >= 0].astype("int64")
        used_codes = set(int(x) for x in valid.unique())
        n_answered = int(len(valid))

        pos_opts = _clean_positive_options(col, label_sets[lbl_name], used_codes)
        if len(pos_opts) < 2 or len(pos_opts) > 15:
            continue
        opt_values = tuple(pos_opts.values())
        if "Not selected" in opt_values:
            continue

        if col not in codebook_headers:
            raise KeyError(f"shed2025.{col}: missing from {CODEBOOK_PDF.name}")
        raw_cb_text = codebook_headers[col]
        qtext = _clean_question_text(raw_cb_text)
        if not qtext:
            raise ValueError(f"shed2025.{col}: empty question stem extracted from codebook")

        pref_m = PREFIX_RE.match(col)
        if not pref_m or pref_m.group(1) not in SECTION_NAMES:
            raise ValueError(f"shed2025.{col}: unexpected section prefix")
        pref = pref_m.group(1)
        family = SECTION_NAMES[pref]

        is_score = opt_values in PURE_ORDINAL_SETS
        kev_type = "score" if is_score else "choice"

        is_random_split = col in RANDOM_SPLIT_VARS or col.startswith(RANDOM_SPLIT_PREFIXES)
        is_select_all = "select all that apply" in raw_cb_text.lower()
        is_other_specify = "other (" in raw_cb_text.lower() or "please specify" in raw_cb_text.lower()

        notes: list[str] = []
        exclude_reason: str | None = None

        if col in DEFAULT_EXCLUDE_REASONS:
            exclude_reason = DEFAULT_EXCLUDE_REASONS[col]
        elif pref == "L":
            exclude_reason = "Living-arrangement roster (overlaps marital_status / household_size / children_in_household)."
        elif is_select_all:
            exclude_reason = "Skip-gated select-all-that-apply follow-up."
        elif is_other_specify:
            exclude_reason = "Catch-all 'Other (please specify)' item."
        elif not is_random_split and n_answered < DEFAULT_MIN_N:
            exclude_reason = f"Skip-gated subsample (N={n_answered:,}/{total_n:,})."

        include = exclude_reason is None

        if is_random_split:
            notes.append(f"Random 50% split ballot (N={n_answered:,}/{total_n:,}).")
        elif n_answered >= total_n - 800:
            notes.append("Full sample.")
        else:
            notes.append(f"Subsample (N={n_answered:,}/{total_n:,}).")

        if n_imputed > 0:
            notes.append(f"Masks {n_imputed} imputed rows via {iflag_col}.")
        if is_score:
            notes.append("Ordered response scale.")
        if exclude_reason:
            notes.append(f"Excluded by default: {exclude_reason}")

        records.append(
            {
                "catalog_idx": len(records),
                "QuestionID": col,
                "family": family,
                "BlockName": col.split("_")[0],
                "n_answered": n_answered,
                "include": str(include),
                "kev_type": kev_type,
                "drop_options": "",
                "review_note": " ".join(notes),
                "QuestionText": qtext,
                "Options_preview": " | ".join(opt_values),
                "data_label": str(var_labels[col]),
                "codebook_text": raw_cb_text,
                "codes": " | ".join(str(c) for c in pos_opts),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("shed2025: built empty question catalog")
    if df["QuestionID"].duplicated().any():
        raise ValueError("shed2025: duplicate QuestionID in built catalog")
    leaked = sorted(set(df["QuestionID"]) & SHED2025_PERSONA_RAW_COLS)
    if leaked:
        raise ValueError(f"shed2025: persona column(s) leaked into question catalog: {leaked}")
    return df


def main() -> None:
    df = build_catalog()
    df.to_csv(OUT_CSV, index=False)
    n_inc = int((df["include"] == "True").sum())
    n_score = int(((df["include"] == "True") & (df["kev_type"] == "score")).sum())
    n_choice = int(((df["include"] == "True") & (df["kev_type"] == "choice")).sum())
    print(
        f"Wrote {OUT_CSV}: {len(df)} questions "
        f"({n_inc} included: {n_score} score, {n_choice} choice; {len(df) - n_inc} excluded)"
    )


if __name__ == "__main__":
    main()
