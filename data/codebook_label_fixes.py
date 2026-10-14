"""Find value labels that a year's .dta file cuts off, and add their missing end from the GESIS codebook.

The .dta files store value labels with a length limit (60 characters in ISSP 2015 and 2017, about 120 from 2020 on),
so some answer options and attribute values end mid-word ("...decided at s"). The codebook (raw/*_cdb.pdf) prints them
in full. For every label that looks cut (at the year's limit, unclosed brackets, or ending in ',', '/', '-', '&'), only
the missing end is added: the label's last words are found in the codebook's label for the same variable and code,
and what follows them is appended. The data's own wording is kept as it is ("incl." stays "incl." where the codebook
writes "including"): wording variation is wanted in training data, never unified. Non-answer codes are skipped.
Output: data/issp<year>/label_fixes.csv (variable, code, data_label, fixed_label, source), applied by
issp_common.read_dta(). Where the codebook cuts a label too, BY_HAND gives the full text from another year's data file
(same variable and code), and the missing end is taken from it the same way. Cut labels without a source are printed
for review; so are endings where the PDF text lost an "ft" ligature (the ISSP 2020 codebook prints "after" as
"a er"), which are not applied.

Needs pdftotext (poppler). Run from the project root, for the years that have a codebook:
  kev/.venv/bin/python data/codebook_label_fixes.py 2015 2017 2020
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent
sys.path.insert(0, str(DATA))
import unicodedata  # noqa: E402

from issp_common import read_dta, read_missing  # noqa: E402

LIMIT = {2015: 59, 2017: 59}                    # stripped label length at which a label counts as cut; 115 otherwise
# (year, variable, code) -> (full label, source): labels the codebook cuts as well
BY_HAND = {
    # the codebook writes it differently ("Jobbik Magyarországért Mozgalom"), so no ending matches; the data's spelling
    (2015, "HU_PRTY", 3): ("Movement for a Better Hungary - Jobbik (Magyarorszagert Mozgalom)",
                           "ZA6770_cdb.pdf, HU_PRTY 3 (Jobbik Magyarországért Mozgalom), in the data's spelling"),
    (2017, "ES_PRTY", 11): ("Nationalist coalition of left wing parties Euskal Herria Bildu - EH Bildu",
                            "ISSP 2016 data file, ES_PRTY 11"),
    (2017, "HU_PRTY", 3): ("Jobbik, the Movement for a Better Hungary - Jobbik Magyarorszagert Mozgalom",
                           "ISSP 2016 data file, HU_PRTY 3"),
    (2020, "PH_RELIG", 79): ("JLFCCI (Jesus Lord of Freedom Christian Community)", "ISSP 2022 data file, PH_RELIG 79"),
    # the 2021 codebook's text layer loses the space at a line break ("odborne smaturitou")
    (2021, "SK_ISCD", 354): ("Completed apprenticeship training with maturity level certif./ Completed voc upper - "
                             "Vyuceny/a s maturitou/ uplne stredoskolske odborne s maturitou (napr. priemyslovka, "
                             "ekonomicka skola, obchodna akademia, stredna zdravotna skola)",
                             "ISSP 2020 codebook (ZA7650_cdb.pdf), SK_ISCD 354"),
}
LOST_FT = re.compile(r"\b(a er|cra |le wing|shi |dra |so ware|e ir|o en)\b")   # words that lost their "ft" glyph
CODE_LINE = re.compile(r"^\s*(-?\d+)\s+(\S.*)$")   # "22 Election Bloc 'Georgian Dream' (...)" in the codebook
# the frequency table the codebook prints after a label on the same line: count, percent, valid percent ("32 2.6 2.6")
FREQUENCIES = re.compile(r"(\s+\d[\d.,]*)?(\s+\d+[.,]\d+){1,2}\s*$")   # decimal point or comma (2021)
MANGLED = re.compile("[\u2044\ufffd]")   # characters the PDF text garbles: "1\u20442" for "\u00bd", unknown glyphs


def strip_number(text):
    """'1. Male' -> 'Male' (labels from 2018 on carry their code in front)."""
    return re.sub(r"^-?\d+\.\s*", "", str(text)).strip()


def squeeze(text):
    """One space between words; ligature characters (\ufb00 'ff', ...) as plain letters."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def looks_cut(text, year):
    """Whether a value label (without its code) ends where the .dta file's length limit cut it."""
    return (len(text) >= LIMIT.get(year, 115) or text.count("(") > text.count(")")
            or text.rstrip().endswith((",", "/", "-", "&", "(")))


def completion(label, full):
    """(the data label with its cut-off end added from `full`, None) or (None, why not). The longest tail of the label
    (8+ characters) found in `full` where the label ends, then the text after it; the label's own wording is kept.

    Rejected: the tail found only well before the label's end (it would repeat text), nothing or only a country note
    after it (the label was not cut), an addition that leaves unbalanced brackets (the label was worded differently,
    not cut) or contains characters the PDF garbles."""
    text = squeeze(strip_number(label))
    for n in range(min(len(text), 40), 7, -1):
        at = full.rfind(text[-n:])
        if at < 0:
            continue
        if at + n < len(text) - 10:                      # matched far from where the label ends: not a continuation
            return None, "the label's ending occurs only earlier in the codebook label"
        rest = full[at + n:]
        if not rest.strip() or rest.lstrip().startswith(";"):
            return None, "nothing missing"
        fixed = label.rstrip() + rest
        if fixed.count("(") < fixed.count(")") or fixed.count("[") < fixed.count("]"):
            return None, "the addition leaves unbalanced brackets (worded differently, not cut)"
        if MANGLED.search(rest):
            return None, "the addition contains characters the PDF garbles"
        return fixed, None
    return None, "no ending of the label found in the codebook label"


def codebook_sections(pdf, variables):
    """{variable: [lines of its codebook sections]}; a section starts at a line 'VAR - title'."""
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    sections, current = {}, None
    for line in text.splitlines():
        header = re.match(r"^(\S+) - ", line)
        if header:
            current = header.group(1) if header.group(1) in variables else None
        if current:
            sections.setdefault(current, []).append(line)
    return sections


def codebook_label(lines, code, start):
    """The codebook's label for `code` that begins like `start`. The codebook prints each label twice: in a full list
    (wrapped lines joined here) and in a frequency table (count and percentages beside it, often cut, its wrapped
    fragments scattered between columns). The full-list label is used; the frequency table's only when there is no
    full list, as its own line with the frequency columns removed."""
    table = None
    for i, line in enumerate(lines):
        match = CODE_LINE.match(line)
        if not match or int(match.group(1)) != code or not squeeze(match.group(2)).startswith(start[:12]):
            continue
        if FREQUENCIES.search(match.group(2)) or re.search(r"\s{3,}\d+\s*$", match.group(2)):
            table = table or squeeze(FREQUENCIES.sub("", match.group(2)).rstrip("0123456789 "))
            continue
        parts = [match.group(2)]
        for following in lines[i + 1:]:
            if not following.strip() or CODE_LINE.match(following) or re.match(r"^\S+ - ", following):
                break
            parts.append(following)
        return squeeze(" ".join(parts))
    return table


def main(year):
    here = DATA / f"issp{year}"
    (dta,) = (here / "raw").glob("*.dta")
    (pdf,) = (here / "raw").glob("*_cdb.pdf")
    _, _, value_labels = read_dta(dta, apply_fixes=False)
    missing = read_missing(next((here / "raw").glob("*_missing.txt"), here / "raw" / "none"))
    cut = [(v, int(k), str(t)) for v, labels in value_labels.items() for k, t in labels.items()
           if k >= 0 and k not in missing.get(v, set()) and looks_cut(squeeze(strip_number(t)), year)]
    sections = codebook_sections(pdf, {v for v, _, _ in cut})
    rows, unresolved = [], []
    for v, code, label in cut:
        data_text = squeeze(strip_number(label))
        full, source = BY_HAND.get((year, v, code)) or (codebook_label(sections.get(v, []), code, data_text), pdf.name)
        fixed, why = completion(label, full) if full else (None, "not in the codebook")
        if fixed and LOST_FT.search(fixed[len(label.rstrip()):]):
            print(f"  not applied, the PDF text lost 'ft': {v} {code} {fixed!r}")
        elif fixed:
            rows.append({"variable": v, "code": code, "data_label": label, "fixed_label": fixed, "source": source})
        else:
            unresolved.append((v, code, label, full, why))
    pd.DataFrame(rows, columns=["variable", "code", "data_label", "fixed_label", "source"]).to_csv(
        here / "label_fixes.csv", index=False)
    print(f"ISSP {year}: {len(cut)} labels look cut, {len(rows)} completed -> {here / 'label_fixes.csv'}")
    for v, code, label, full, why in unresolved:
        if why != "nothing missing":
            print(f"  not completed ({why}): {v} {code} {label!r} (codebook: {full!r})")


if __name__ == "__main__":
    for year in sys.argv[1:]:
        main(int(year))
