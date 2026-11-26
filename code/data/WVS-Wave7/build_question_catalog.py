"""Build the WVS Wave 7 United States question catalog and the starting `question_audit.csv`.

Sources (official WVSA releases only; WorldValuesBench is not read):
- Question wording: the US codebook PDF
  `F00010562-World_Values_Survey_Wave_7_2017-2020_United_States_v3.0.pdf` (text via `pdftotext -layout`).
- Answer options and codes: the value labels in `WVS_Wave_7_United_States_Stata_v5.1.dta`.

Outputs, in this folder:
- `wvs7_questions.json`: for every Q-variable in the .dta, the PDF text and PDF response categories as printed,
  the Stata variable label, value labels and valid-answer count, and where PDF and Stata categories disagree.
- `question_audit.csv`: starting include / kev_type / QuestionText per row, with the reason in `review_note`
  (Claude's evaluation; the user verifies in the audit app). It is only written when absent, or with --overwrite,
  so the user's verified audit is never replaced by accident.

Run from `code/`:  python data/WVS-Wave7/build_question_catalog.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import pandas as pd
from pandas.io.stata import StataReader

WVS_DIR = Path(__file__).resolve().parent
PDF_PATH = WVS_DIR / "F00010562-World_Values_Survey_Wave_7_2017-2020_United_States_v3.0.pdf"
DTA_PATH = WVS_DIR / "WVS_Wave_7_United_States_Stata_v5.1.dta"
CATALOG_PATH = WVS_DIR / "wvs7_questions.json"
AUDIT_PATH = WVS_DIR / "question_audit.csv"
TAGS_PATH = WVS_DIR / "question_tags.json"

# Stata extended missing values (.a, .b, ...) carry value labels at codes >= this bound.
STATA_MISSING_MIN = 2147483621

HEAD_RE = re.compile(r"^(Q\d+[A-Z0-9_]*|[A-Z]\d?_[A-Z_]+|[A-Z]\d{3}[A-Z0-9]*)- (.*)$")
PAGE_FURNITURE_RE = re.compile(
    r"^(World Values Survey Wave 7|United States: Distributions|Page \d+ of|Sex\s+Age|TOTAL\b|Male\s+Female|"
    r"Male$|Female$|Up to 29|50 and|more$)"
)
NUMBER_CELL_RE = re.compile(r"[\d.,]+|-")
# PDF response categories that are missing codes, not answer options.
PDF_MISSING_RE = re.compile(r"^(Don.t know|No answer|Not asked|Missing|Other missing|Not applicable|INAP)", re.I)

# WVSA questionnaire sections (master questionnaire headings), by first and last Q number.
SECTIONS: list[tuple[int, int, str]] = [
    (1, 45, "01_Social values, attitudes & stereotypes"),
    (46, 56, "02_Happiness and well-being"),
    (57, 105, "03_Social capital, trust & organizational membership"),
    (106, 111, "04_Economic values"),
    (112, 120, "05_Corruption"),
    (121, 130, "06_Migration"),
    (131, 151, "07_Security"),
    (152, 157, "08_Postmaterialist index"),
    (158, 163, "09_Science & technology"),
    (164, 175, "10_Religious values"),
    (176, 198, "11_Ethical values and norms"),
    (199, 234, "12_Political interest & political participation"),
    (235, 259, "13_Political culture & political regimes"),
    (260, 290, "14_Demographics"),
]

# Columns read by `load_raw_wvs7_personas` outside the demographics block; they are personas, not questions.
PERSONA_QUESTION_VARS = {"Q97", "Q171", "Q222", "Q223", "Q223_ABREV", "Q223_LOCAL", "Q240"}

# Template placeholders of the master questionnaire ([your country], [nationality], ...) are kept in QuestionText,
# as ISSP keeps {country}; code/data/wvs_export.py (US_FILLS) fills them at export time.
PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")

# Codebook-stem defects kept verbatim per user rule (2026-11-25: never silently reword codebook text because it looks
# wrong; flag it for the user's review instead):
# - Q144: trailing bare row label 'Respondent' ('Have you been the victim of a crime during the past year? Respondent').
# - Q221: codebook omits the question stem entirely and prints only the variable label ('Vote in elections: Local level').
# - Q254: garbled codebook stem ('How proud are you to be of nationality of this country?').
# - Q256, Q257, Q259: codebook prints an agree/disagree stem while options are Very close ... Not close at all.
# - Q154, Q156: card-only stems ('which one of the things on this card...' / 'Here is another list...') that omit the
#   national-aims topic introduced in Q152 (tagged 'backref').
STEM_FLAGGED = ("Q144", "Q221", "Q254", "Q256", "Q257", "Q259")
BACKREF_FLAGGED = ("Q154", "Q156")
REVIEW_TAG = "review-codebook-stem"
BACKREF_TAG = "backref"

# Batteries where the PDF lost the separator between the shared stem and the item.
SEPARATOR_REPAIRS: dict[str, tuple[str, str]] = {
    **{
        f"Q{i}": ("or not important at all ", "or not important at all: ")
        for i in range(1, 7)
    },
    **{
        f"Q{i}": ("an essential characteristic of democracy ", "an essential characteristic of democracy”: ")
        for i in range(241, 250)
    },
}

EXCLUDE_REASONS: dict[str, str] = {
    "Q33_3": "Derived 3-point recode of Q33 (same answers, different codes); Q33 is the item.",
    "Q34_3": "Derived 3-point recode of Q34 (same answers, different codes); Q34 is the item.",
    "Q35_3": "Derived 3-point recode of Q35 (same answers, different codes); Q35 is the item.",
    "Q82": "Cross-national combination of the regional-organization item; identical answers to Q82_NAFTA, "
    "which is the US item and has its own PDF wording.",
    "Q153": "Second choice: depends on the answer to Q152 (the first choice cannot be picked again).",
    "Q155": "Second choice: depends on the answer to Q154 (the first choice cannot be picked again).",
    "Q157": "Second choice: depends on the answer to Q156 (the first choice cannot be picked again).",
}

CHOICE_REASONS: dict[str, str] = {
    **{f"Q{i}": "Binary item (Important / Not mentioned) from a pick-up-to-five list." for i in range(7, 18)},
    **{f"Q{i}": "Binary item (Mentioned / Not mentioned) from a would-not-like-as-neighbors list." for i in range(18, 27)},
    "Q42": "Three unordered statements about society (revolution / reform / defend).",
    "Q56": "Better off / Worse off / Or about the same: listed order is not a scale order.",
    "Q57": "Binary (Most people can be trusted / Need to be very careful).",
    "Q91": "Factual knowledge question; options are unordered countries.",
    "Q92": "Factual knowledge question; options are unordered cities.",
    "Q93": "Factual knowledge question; options are unordered topics.",
    "Q111": "Environment vs growth vs Other answer: unordered.",
    "Q119": "Hard to say (code 0) is listed before the agree-disagree scale, so the listed order is not a scale.",
    "Q139": "Binary Yes / No.",
    "Q140": "Binary Yes / No.",
    "Q141": "Binary Yes / No.",
    "Q144": "Binary Yes / No. REVIEW (flagged, wording unchanged): codebook appends trailing row label 'Respondent' "
    "to the question stem.",
    "Q145": "Binary Yes / No. PDF text 'And what about your immediate family' only makes sense after Q144, so "
    "QuestionText prefixes Q144's question (adds words, removes none).",
    "Q149": "Binary (Freedom / Equality).",
    "Q150": "Binary (Freedom / Security).",
    "Q151": "Binary Yes / No.",
    "Q152": "Unordered national aims (first choice). Stata labels code 3 'how  are done' (word missing, same in "
    "PDF); question_overrides.json adds 'things'. PDF also lists 'Trying to make our cities and countryside more "
    "beautiful' at 0%; the .dta has no label or answers for it, so it is not an option here.",
    "Q154": "Unordered national aims (first choice). REVIEW (flagged #backref, wording unchanged): stem refers to "
    "'the things on this card' without stating the Q152 topic (aims of this country).",
    "Q156": "Unordered national aims (first choice). REVIEW (flagged #backref, wording unchanged): stem starts with "
    "'Here is another list.' without stating the Q152 topic (aims of this country).",
    "Q165": "Binary Yes / No.",
    "Q166": "Binary Yes / No.",
    "Q167": "Binary Yes / No.",
    "Q168": "Binary Yes / No.",
    "Q173": "A religious person / Not a religious person / An atheist: categories, not a scale.",
    "Q174": "Binary statement choice.",
    "Q175": "Binary statement choice.",
    "Q221": "Always / Usually / Never plus 'Not allowed to vote' (off-scale); kept as choice so no answers are dropped. "
    "REVIEW (flagged, wording unchanged): codebook omits the question stem ('When elections take place, do you vote "
    "always, usually or never? ...') and prints only the label 'Vote in elections: Local level'.",
    "Q254": "Very proud ... Not at all proud plus 'I am not [nationality]' (off-scale); kept as choice. "
    "wvs_export.py fills [nationality] with 'American' at export time. REVIEW (flagged, wording unchanged): PDF stem "
    "'How proud are you to be of nationality of this country?' is garbled but kept verbatim.",
}

SCORE_NOTES: dict[str, str] = {
    "Q33": "US fielded 3 points (codes 2-4); PDF lists Agree strongly / Disagree strongly at 0%.",
    "Q34": "US fielded 3 points (codes 2-4); PDF lists Agree strongly / Disagree strongly at 0%.",
    "Q35": "US fielded 3 points (codes 2-4); PDF lists Strongly agree / Disagree strongly at 0%.",
    "Q106": "PDF endpoint labels 'Incomes more equal' / 'Larger income differences' differ from the Stata labels "
    "used here.",
    "Q121": "Stata and PDF label code 1 'Rather bad' (master questionnaire: 'Very bad'); Stata label kept.",
    **{
        v: "REVIEW (flagged, wording unchanged): the codebook prints the agree/disagree 'how you see yourself' stem, "
        "while the options are Very close ... Not close at all and the Stata label reads 'Feel close to ...'. "
        "QuestionText is the codebook wording as printed; the user decides."
        for v in ("Q256", "Q257", "Q259")
    },
}


def _q_number(v: str) -> int:
    return int(re.match(r"Q(\d+)", v)[1])


def _section(v: str) -> str:
    n = _q_number(v)
    for lo, hi, name in SECTIONS:
        if lo <= n <= hi:
            return name
    raise ValueError(f"{v}: Q number {n} outside every WVS7 section")


def parse_codebook_pdf(pdf_path: Path) -> dict[str, dict[str, object]]:
    """Each variable block of the US codebook: header text (joined lines) and printed response categories."""
    text = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"], check=True, capture_output=True, text=True
    ).stdout
    lines = text.splitlines()
    blocks: dict[str, dict[str, object]] = {}
    i = 0
    while i < len(lines):
        m = HEAD_RE.match(lines[i])
        if not m:
            i += 1
            continue
        var = m[1]
        header = [m[2].strip()]
        i += 1
        # The wording ends at a blank line or, when the table follows directly, at the table header.
        while (
            i < len(lines)
            and lines[i].strip()
            and not HEAD_RE.match(lines[i])
            and not PAGE_FURNITURE_RE.match(lines[i].strip())
        ):
            header.append(lines[i].strip())
            i += 1
        options: list[str] = []
        while i < len(lines) and not HEAD_RE.match(lines[i]):
            line = lines[i].strip()
            i += 1
            if not line or PAGE_FURNITURE_RE.match(line):
                continue
            if line.startswith("(N)"):
                break
            cells = re.split(r"\s{2,}", line)
            if len(cells) >= 2 and NUMBER_CELL_RE.fullmatch(cells[1]):
                options.append(cells[0])
            else:
                # Long category labels wrap onto a line without numbers.
                if not options:
                    raise ValueError(f"{var}: wrapped label line before any category: {line!r}")
                options[-1] = f"{options[-1]} {line}"
        if var in blocks:
            raise ValueError(f"{var}: appears twice in the codebook PDF")
        blocks[var] = {"text": " ".join(header), "options": options}
    return blocks


def read_stata() -> tuple[list[str], dict[str, str], dict[str, dict[int, str]], pd.DataFrame]:
    with StataReader(DTA_PATH) as reader:
        var_labels = reader.variable_labels()
        label_sets = reader.value_labels()
        value_labels = {
            v: {int(k): str(t) for k, t in label_sets[s].items()} if s in label_sets else {}
            for v, s in zip(reader._varlist, reader._lbllist)
        }
        varlist = list(reader._varlist)
    raw = pd.read_stata(DTA_PATH, convert_categoricals=False)
    q_vars = [v for v in varlist if re.fullmatch(r"Q\d+[A-Z0-9_]*", v)]
    return q_vars, var_labels, value_labels, raw


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("´", "'")).strip()


def build_catalog() -> dict[str, dict[str, object]]:
    pdf = parse_codebook_pdf(PDF_PATH)
    q_vars, var_labels, value_labels, raw = read_stata()
    catalog: dict[str, dict[str, object]] = {}
    for v in q_vars:
        offered = {k: t for k, t in sorted(value_labels[v].items()) if k < STATA_MISSING_MIN}
        valid = raw[v].dropna()
        entry: dict[str, object] = {
            "section": _section(v),
            "data_label": var_labels[v],
            "stata_codes": list(offered),
            "stata_labels": list(offered.values()),
            "n_answered": int(len(valid)),
            "answered_codes": sorted({int(x) for x in valid}) if len(valid.unique()) <= 30 else None,
            "in_pdf": v in pdf,
        }
        if v in pdf:
            pdf_opts = [str(o) for o in pdf[v]["options"]]
            substantive = [o for o in pdf_opts if not PDF_MISSING_RE.match(o)]
            stata_set = {_norm(t) for t in offered.values()}
            pdf_set = {_norm(o) for o in substantive}
            entry["pdf_text"] = pdf[v]["text"]
            entry["pdf_options"] = pdf_opts
            entry["pdf_only_options"] = sorted(pdf_set - stata_set)
            entry["stata_only_options"] = sorted(stata_set - pdf_set) if offered else []
        catalog[v] = entry
    return catalog


def question_text(v: str, catalog: dict[str, dict[str, object]]) -> str:
    """PDF wording with only the documented repairs: separators, Q145 context.
    Placeholders stay; the exporter fills them."""
    text = str(catalog[v]["pdf_text"])
    if v in SEPARATOR_REPAIRS:
        old, new = SEPARATOR_REPAIRS[v]
        if text.count(old) != 1:
            raise ValueError(f"{v}: separator anchor {old!r} found {text.count(old)} times in {text!r}")
        text = text.replace(old, new)
    if v == "Q145":
        q144 = str(catalog["Q144"]["pdf_text"])
        if not q144.endswith("? Respondent"):
            raise ValueError(f"Q144 text changed: {q144!r}")
        text = f"{q144.removesuffix(' Respondent')} {text}"
    return text


def audit_row(idx: int, v: str, catalog: dict[str, dict[str, object]]) -> dict[str, str]:
    entry = catalog[v]
    labels = [str(t) for t in entry["stata_labels"]]
    n = _q_number(v)
    include = "False"
    kev_type = ""
    if v in PERSONA_QUESTION_VARS or n >= 260:
        note = "Persona column (read by load_raw_wvs7_personas / demographics block), not a question."
    elif not entry["in_pdf"]:
        if v not in EXCLUDE_REASONS:
            raise ValueError(f"{v}: not in the codebook PDF and no exclusion reason")
        note = EXCLUDE_REASONS[v]
    elif v in EXCLUDE_REASONS:
        note = EXCLUDE_REASONS[v]
    elif v in CHOICE_REASONS:
        include, kev_type, note = "True", "choice", CHOICE_REASONS[v]
    else:
        include, kev_type = "True", "score"
        note = "Ordered response scale listed in scale order."
        if v in SCORE_NOTES:
            note = f"{note} {SCORE_NOTES[v]}"
    disagreements = []
    if entry["in_pdf"] and entry["pdf_only_options"]:
        disagreements.append(f"PDF-only categories: {' | '.join(entry['pdf_only_options'])}")
    if entry["in_pdf"] and entry["stata_only_options"]:
        disagreements.append(f"Stata-only labels: {' | '.join(entry['stata_only_options'])}")
    if disagreements:
        note = f"{note} [PDF vs Stata: {'; '.join(disagreements)}]"
    text = question_text(v, catalog) if entry["in_pdf"] else ""
    placeholders = sorted(set(PLACEHOLDER_RE.findall(text)) | {p for t in labels for p in PLACEHOLDER_RE.findall(t)})
    if include == "True" and placeholders:
        note = f"{note} Placeholders {' '.join(placeholders)} are filled at export (wvs_export.US_FILLS)."
    return {
        "catalog_idx": str(idx),
        "QuestionID": v,
        "family": str(entry["section"]),
        "BlockName": v,
        "n_answered": str(entry["n_answered"]),
        "include": include,
        "kev_type": kev_type,
        "drop_options": "",
        "review_note": note,
        "QuestionText": text,
        "Options_preview": " | ".join(labels),
        "data_label": str(entry["data_label"]),
        "codes": " | ".join(str(c) for c in entry["stata_codes"]),
    }


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    os.replace(tmp, path)


def _flag_for_review(audit: pd.DataFrame) -> None:
    """Add review tags to flagged rows in question_tags.json (audit app tags), keeping existing tags."""
    tags: dict[str, list[str]] = {}
    if TAGS_PATH.exists():
        tags = json.loads(TAGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(tags, dict):
            raise TypeError(f"{TAGS_PATH}: expected a JSON object")
    for tag, vars_to_flag in ((REVIEW_TAG, STEM_FLAGGED), (BACKREF_TAG, BACKREF_FLAGGED)):
        for v in vars_to_flag:
            row = audit[audit["QuestionID"] == v]
            if len(row) != 1:
                raise ValueError(f"{v}: expected one audit row, found {len(row)}")
            row_key = f"{row['catalog_idx'].iloc[0]}:{v}"
            if tag not in tags.setdefault(row_key, []):
                tags[row_key].append(tag)
    _write_atomic(TAGS_PATH, lambda p: p.write_text(json.dumps(tags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the WVS7 US question catalog and starting audit CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing question_audit.csv.")
    args = parser.parse_args()

    catalog = build_catalog()
    _write_atomic(
        CATALOG_PATH,
        lambda p: p.write_text(json.dumps(catalog, indent=1, ensure_ascii=False), encoding="utf-8"),
    )
    print(f"{CATALOG_PATH.name}: {len(catalog)} Q-variables")

    if AUDIT_PATH.exists() and not args.overwrite:
        print(f"{AUDIT_PATH.name} exists; left unchanged (pass --overwrite to rebuild it).")
        return
    rows = [audit_row(i, v, catalog) for i, v in enumerate(catalog)]
    audit = pd.DataFrame(rows)
    _write_atomic(AUDIT_PATH, lambda p: audit.to_csv(p, index=False))
    _flag_for_review(audit)
    inc = audit[audit["include"] == "True"]
    print(
        f"{AUDIT_PATH.name}: {len(audit)} rows, {len(inc)} included "
        f"({(inc['kev_type'] == 'choice').sum()} choice, {(inc['kev_type'] == 'score').sum()} score)"
    )


if __name__ == "__main__":
    main()
