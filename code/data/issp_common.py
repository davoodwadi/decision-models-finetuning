"""Shared builder for the ISSP training waves: every year 2015-2024.

Each year's folder holds only what differs between years:
- build_question_bank.py: the source-questionnaire wording of every item (SOURCE), which items are `choice`, `noul` or
  excluded, and which `score` scales are printed high -> low; it calls write_question_bank().
- build_dataset.py: which questionnaire blocks were asked of only part of the sample (ASKED_OF); it calls
  build_dataset(), which writes the respondent-level training data (one row per respondent with every attribute and
  answer, plus the questions per country) that issp_sampler.py draws training records from.

Attributes (PERSONA_KEY) are the same in every year. Country names, nationality words and parliament names for the
question placeholders come from data/issp_countries.csv.
"""
import json
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent                     # data/
COUNTRIES = DATA / "issp_countries.csv"

# a value label that starts with country codes ("TW: Women should decide", "NL, US: Employers") is an extra answer
# that only those countries offered: not used
COUNTRY_ONLY = re.compile(r"^[A-Z]{2}(-[A-Z]+)?(\s*,\s*[A-Z]{2}(-[A-Z]+)?)*\s*:")


# ---------------------------------------------------------------- reading the data file

def read_missing(path):
    """{variable: set of codes that are not answers} from GESIS's *_missing.txt, e.g. ' v1 (8, 9)',
    ' v46 (LOWEST THRU -4, -1)'. Without the file ({}), only negative codes are non-answers."""
    missing = {}
    if not Path(path).exists():
        return missing
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        match = re.match(r"\s*(\S+)\s*\((.*)\)", line)
        if not match:
            continue
        codes = set()
        for part in match.group(2).upper().replace("LOWEST", "-999").replace("HIGHEST", "99999").split(","):
            if "THRU" in part:
                low, high = part.split("THRU")
                codes |= set(range(int(low), int(high) + 1))
            elif part.strip():
                codes.add(int(float(part)))
        missing[match.group(1)] = codes
    return missing


def read_dta(path, apply_fixes=True):
    """(data, variable labels, {variable: {code: value label}}).

    Value labels the .dta file cuts off get their missing end, from label_fixes.csv in the year's folder (made by
    data/codebook_label_fixes.py), when that file exists."""
    reader = pd.io.stata.StataReader(path)
    raw = reader.read(convert_categoricals=False)
    labels_by_name = reader.value_labels()
    value_labels = {v: dict(labels_by_name.get(name, {})) for v, name in zip(reader._varlist, reader._lbllist)}
    fixes = Path(path).parent.parent / "label_fixes.csv"
    if apply_fixes and fixes.exists():
        for fix in pd.read_csv(fixes, keep_default_na=False).itertuples():
            labels = value_labels[fix.variable]
            assert labels.get(fix.code) == fix.data_label, f"{fixes}: {fix.variable} {fix.code} no longer matches the data"
            labels[fix.code] = fix.fixed_label
    return raw, reader.variable_labels(), value_labels


def clean(text):
    """'1. Strongly agree' -> 'Strongly agree'; a bare number such as '02' -> '2'; 'Bottom, Lowest, 01' -> 'Bottom, Lowest';
    'Never or almost never; PL: 1-2 times a year' -> 'Never or almost never'."""
    text = re.sub(r"^-?\d+\.\s*", "", str(text)).strip()
    text = re.sub(r"(?<=\D),\s*\d+$", "", text)
    text = re.sub(r"^\d+\s+-\s+", "", text)                    # '10 - Complete trust' -> 'Complete trust'
    text = re.sub(r"[;,]\s*[A-Z]{2}(\s*,\s*[A-Z]{2})*\s*[:;].*$", "", text)   # drop country notes: "...; KR: About once a year"
    return str(int(text)) if text.isdigit() else text


def answers(labels, not_answers, extra=None):
    """(code, option text) for every answer code: value labels without the non-answer codes, negative codes and
    answers only some countries offered. `extra` ({code: text}) adds answers the data codes as non-answers although
    the questionnaire prints them as an option (e.g. "Does not apply"): they are real answers.

    Numeric scales label only some points ('1. Not at all important', '7. Very important'): the codes in between get
    their number as text, and labelled points keep their number in front ('1 Not at all important'), as in ISSP 2020."""
    country_only = {k for k, t in labels.items() if COUNTRY_ONLY.match(re.sub(r"^-?\d+\.\s*", "", str(t)))}
    texts = {int(k): clean(t) for k, t in labels.items() if k >= 0 and k not in not_answers and k not in country_only}
    texts |= extra or {}
    if not texts:                                        # open answers (amounts, counts) have no value labels
        return []
    scale = range(min(texts), max(texts) + 1)
    if not extra and len(texts) <= 3 < len(scale) <= 11:  # a scale labelled only at its ends (and middle): fill in
        texts = {k: texts.get(k, str(k)) for k in scale}
    if any(t == str(k) for k, t in texts.items()):       # a numeric scale: labelled points keep their number
        texts = {k: t if t == str(k) or t.startswith(f"{k} ") else f"{k} {t}" for k, t in texts.items()}
    return sorted(texts.items())


# ---------------------------------------------------------------- question bank

# first option of a `score` scale that is printed high -> low: such scales are reversed so levels run low -> high
HIGH_FIRST = ("Strongly agree", "Agree strongly", "Very ", "Always", "Completely", "Definitely", "Much ", "A great deal",
              "Almost all", "Almost always", "Every day", "Several times a day", "Extremely", "Increased a lot")


COUNTRY_NOTE = re.compile(r"([A-Z]{2}(?:\s*,\s*[A-Z]{2})*)\s*[:;]\s*([^;]+)")
NOT_OFFERED = ("not used", "not asked", "not available")   # a country note saying the option was not offered there


def country_options(label):
    """{country: option text} from a value label's country notes: 'About once or twice a year; KR: About once a year'
    -> {'KR': 'About once a year'}; a note starting with '+' or 'also' extends the master text ('Father unknown; IL, VE:
    + father already passed away' -> 'Father unknown + father already passed away'), and so does one starting with
    'incl' ('No, NZ: incl. was not eligible' -> 'No, incl. was not eligible'); 'Not used' marks an option the country
    did not offer (kept as the note's text; write_question_bank drops it for that country)."""
    text = re.sub(r"^-?\d+\.\s*", "", str(label)).strip()
    start = re.search(r"[;,]\s*[A-Z]{2}(\s*,\s*[A-Z]{2})*\s*[:;]", text)
    if not start:
        return {}
    master, notes = text[:start.start()].strip(), text[start.start() + 1:]
    variants = {}
    for countries, variant in COUNTRY_NOTE.findall(notes):
        variant = variant.strip()
        for country in re.split(r"\s*,\s*", countries):
            if variant.startswith(("+", "also")):
                variants[country] = f"{master} {variant}"
            elif variant.lower().startswith("incl"):
                variants[country] = f"{master}, {variant}"
            else:
                variants[country] = variant
    return variants


def write_question_bank(here, dta, missing_txt, source, choice=(), noul=(), exclude=None, high_first=HIGH_FIRST,
                        low_first=(), extra_answers=None, country_not_answers=None):
    """One row per item: question number (from the data label), wording, type, answer codes and options.
    `low_first`: score items whose first option matches `high_first` but already run low -> high.
    `extra_answers`: {v: {code: option text}} printed answers the data codes as non-answers (see answers()).
    Country notes in the value labels ('...; KR: About once a year') give that country's own option wording
    (`country_options`, used only for that country's records; not for `extra_answers`, whose text is given by hand).
    `country_not_answers`: {v: {country: [codes]}} codes that are non-answers in one country although the label
    names an answer. Those, and options a country did not offer ('KR, US: Not used'), are not counted for that
    country (`country_dropped`); for a `score` item only codes at the ends of the scale may be dropped."""
    raw, var_labels, value_labels = read_dta(dta)
    missing = read_missing(missing_txt)
    exclude, extra_answers, country_not_answers = exclude or {}, extra_answers or {}, country_not_answers or {}
    rows = []
    for v, question in source.items():
        pairs = answers(value_labels[v], missing.get(v, set()), extra_answers.get(v))
        pairs = [(k, re.sub(r"\[COUNTRY('S)?\]", lambda m: "{country}'s" if m.group(1) else "{country}", t, flags=re.I))
                 for k, t in pairs]                      # "[COUNTRY]" / "[COUNTRY's]" in answer labels -> placeholders
        pairs = [(k, re.sub(r"\[EU/ ?USMCA/ ?etc\.?\]", "{association}", t)) for k, t in pairs]   # ISSP 2023 Q39
        pairs = [(k, re.sub(r"\[([^\]]*)\]", r"(\1)", t)) for k, t in pairs]   # other "[...]" (national values) -> "(...)"
        qtype = "exclude" if v in exclude else "choice" if v in choice else "noul" if v in noul else "score"
        reverse = qtype == "score" and v not in low_first and bool(pairs) and pairs[0][1].startswith(high_first)
        if reverse:
            pairs = pairs[::-1]
        dropped = {country: [int(k) for k in codes] for country, codes in country_not_answers.get(v, {}).items()}
        own = {}                                         # {country: {code: option text}}
        for k, _ in pairs:
            if k in extra_answers.get(v, {}) or qtype == "exclude":
                continue
            for country, text in country_options(value_labels[v].get(k, "")).items():
                if text.lower().startswith(NOT_OFFERED):
                    dropped.setdefault(country, []).append(k)
                elif k not in dropped.get(country, []):
                    own.setdefault(country, {})[str(k)] = text
        codes_in_order = [k for k, _ in pairs]
        for country, codes in dropped.items():           # a score scale may lose levels only at its ends
            kept = [i for i, k in enumerate(codes_in_order) if k not in codes]
            assert qtype != "score" or kept == list(range(kept[0], kept[-1] + 1)), f"{v} {country}: gap in the scale"
        rows.append({"v": v, "source_q": var_labels[v].split()[0], "data_label": var_labels[v], "type": qtype,
                     "question": question,
                     "codes": " | ".join(str(k) for k, _ in pairs),        # data code of each option, in option order
                     "options": " | ".join(t for _, t in pairs), "reversed": reverse, "note": exclude.get(v, ""),
                     "country_options": json.dumps(own, ensure_ascii=False) if own else "",
                     "country_dropped": json.dumps(dropped) if dropped else ""})
    pd.DataFrame(rows).to_csv(here / "question_bank.csv", index=False)
    print(f"wrote {len(rows)} items to {here / 'question_bank.csv'}")


# ---------------------------------------------------------------- questions per country

def fill(text, c):
    """Insert a country's names into a question template; capitalise a leading 'the'."""
    name = c["country"]
    possessive = name + ("'" if name.endswith("s") else "'s")
    for key, value in [("{country}'s", possessive), ("{country}", name), ("{nationality}", c["nationality"]),
                       ("{parliament}", c["parliament"]), ("{association}", c.get("association") or "")]:
        text = text.replace(key, value)
    return re.sub(r"(^|[.?!…]\s+)the ", lambda m: m.group(1) + "The ", text)


def note_country(code):
    """The two-letter code country notes use for a data country code ('GB-GBN' -> 'GB')."""
    return code.split("-")[0]


def country_field(row, name):
    """A question bank row's JSON column ({} when empty or absent)."""
    cell = getattr(row, name, "")
    return json.loads(cell) if isinstance(cell, str) and cell else {}


def question_spec(row, c, code):
    """Kev question (without label/target) for country c (country code `code`), and a map from data code to option
    key. The country's own option wording (`country_options`) replaces the master wording where the data notes one;
    codes that are not answers in that country, or options it did not offer (`country_dropped`), are left out."""
    opts, codes = fill(row.options, c).split(" | "), [int(k) for k in str(row.codes).split(" | ")]
    own = country_field(row, "country_options").get(note_country(code), {})
    opts = [fill(own[str(k)], c) if str(k) in own else option for k, option in zip(codes, opts)]
    dropped = set(country_field(row, "country_dropped").get(note_country(code), []))
    opts, codes = [o for o, k in zip(opts, codes) if k not in dropped], [k for k in codes if k not in dropped]
    if row.type == "score":
        return {"type": "score", "criteria": opts}, {code: str(i) for i, code in enumerate(codes)}
    if row.type == "choice":
        return {"type": "choice", "criteria": {o: None for o in opts}}, dict(zip(codes, opts))
    yes, no = opts  # noul items list yes first, no second
    return {"type": "noul", "criteria": {"true": yes, "false": no}}, {codes[0]: "true", codes[1]: "false"}


# ---------------------------------------------------------------- respondent attributes

# the attributes personas are drawn from (PLAN-respondent-sampling.md, step 2): ISSP variable -> persona line
PERSONA_KEY = {"SEX": "Sex", "AGE": "Age", "EDUCATION": "Education", "RELIGGRP": "Religion",
               "ATTEND": "Attendance of religious services", "PRTY": "Party voted for in the last general election",
               "PARTY_LR": "Voted party's left-right position", "VOTE_LE": "Voted in the last general election",
               "INC": "Household income", "REG": "Region", "URBRURAL": "Place of living",
               "MARITAL": "Marital status", "MAINSTAT": "Main activity", "UNION": "Trade union membership",
               "TOPBOT": "Self-placed social position (1 = bottom, 10 = top)", "ETHN1": "Ancestry"}
COUNTRY_SPECIFIC = {"PRTY", "REG", "INC", "ETHN1"}      # stored per country as <CC>_<name> (US_PRTY, DE_PRTY, ...)


def attribute_columns(raw, name):
    """{country: data variable} holding attribute `name` (PERSONA_KEY) for each country of the year."""
    if name in COUNTRY_SPECIFIC:                         # each country's own variable, e.g. US_PRTY for the US
        return {c[:2]: c for c in raw.columns if re.fullmatch(rf"[A-Z]{{2}}_{name}", c)}
    column = ("EDULEVEL" if "EDULEVEL" in raw else "DEGREE") if name == "EDUCATION" else name
    return {code: column for code in raw.c_alphan.unique()} if column in raw else {}


def attributes(raw, value_labels, missing, notes=None):
    """(codes, reference) for every attribute the year has.

    codes: {persona line: each respondent's data code (Int32; <NA> = missing: negative, or listed in *_missing.txt)}.
    reference: {persona line: {"variables": {country: data variable}, "values": {country: {code: text}}}}, the text of
    every code that occurs, per country: the respondent's raw value (age in years; the value label otherwise), as that
    country's questionnaire had it. A country note in the value label ('2 or 3 times a month; DE: 1-3 times a month',
    '89 years; US: 89 years or more') gives that country's own text (country_options); `notes` {(variable, code,
    country): (text, source)} gives it by hand where the note is abbreviated."""
    notes = notes or {}
    codes, reference = {}, {}
    for name, key in PERSONA_KEY.items():
        columns = attribute_columns(raw, name)
        series = pd.Series(pd.NA, index=raw.index, dtype="Int32")
        values = {}
        for country, column in columns.items():
            rows = raw.c_alphan == country
            data = raw.loc[rows, column]
            data = data.where(data.notna() & (data >= 0) & ~data.isin(missing.get(column, set())))
            if data.isna().all():
                continue
            series[rows] = data.astype("Int32")
            labels = value_labels[column]
            values[country] = {}
            for code in sorted(int(k) for k in data.dropna().unique()):
                label = labels.get(code)
                own = country_options(label).get(note_country(country)) if label is not None else None
                default = str(code) if name == "AGE" else clean(label if label is not None else str(code))
                by_hand = notes.get((column, code, country)) or notes.get((column, code, note_country(country)))
                values[country][str(code)] = (by_hand or (own,))[0] or default
        if values:
            codes[key] = series
            reference[key] = {"variables": {c: columns[c] for c in values}, "values": values}
    return codes, reference


# ---------------------------------------------------------------- dataset

def build_dataset(here, year, dta, asked_of=None, country_fields=None, attribute_notes=None):
    """Write the year's training data for the sampler (issp_sampler.py), aggregated only when a record is drawn:

    - issp<year>_respondents.parquet: one row per respondent (`respondent` = row in the .dta), with `country`, every
      attribute's data code (PERSONA_KEY; empty = missing) and every question's raw answer code (empty = not asked).
    - issp<year>_attributes.json: per attribute and country, the text of each code (see attributes()).
    - issp<year>_questions.json: per question its block and, per country, the Kev question (type, criteria, wording
      with placeholders filled, the country's own options) and the map from answer code to option key (codes that are
      not answers there are not in it); per block and country the persona lines saying who was asked (asked_of); the
      country names.

    asked_of: {block: {persona line: value}} for blocks asked of only part of the sample (placeholders such as
    {association} are filled in). country_fields: {country code: {placeholder: text}}, e.g. {"association": ...}.
    attribute_notes: {(variable, code, country): (text, source)}, a country's attribute value written out by hand where
    the data's country note is abbreviated."""
    raw, _, value_labels = read_dta(dta)
    missing = read_missing(next(Path(dta).parent.glob("*_missing.txt"), Path(dta).parent / "none"))
    qb = pd.read_csv(here / "question_bank.csv")
    qb = qb[qb.type != "exclude"]
    qb["block"] = qb.source_q.str.extract(r"^([A-Za-z]+\d+)")[0]
    countries = pd.read_csv(COUNTRIES, keep_default_na=False).set_index("c_alphan").to_dict("index")
    asked_of = asked_of or {}
    for code, fields in (country_fields or {}).items():
        countries[code] = {**countries[code], **fields}
    present = sorted(raw.c_alphan.unique())

    table = pd.DataFrame({"respondent": raw.index, "country": raw.c_alphan.astype(str)})
    codes, reference = attributes(raw, value_labels, missing, attribute_notes)
    for key, values in codes.items():
        table[key] = values
    for v in qb.v:
        table[v] = raw[v].astype("Int32")
    table.to_parquet(here / f"issp{year}_respondents.parquet", index=False)

    questions = {}
    for row in qb.itertuples():
        per_country = {}
        for code in present:
            c = countries[code]
            spec, code_key = question_spec(row, c, code)
            per_country[code] = {"question": {**spec, "instructions": fill(row.question, c)},
                                 "codes": {str(k): key for k, key in code_key.items()}}
        questions[row.v] = {"block": row.block, "countries": per_country}
    blocks = {block: {code: {fill(k, countries[code]): fill(v, countries[code]) for k, v in lines.items()}
                      for code in present} for block, lines in asked_of.items()}
    out = {"year": year, "src": f"issp{year}",
           "countries": {code: countries[code]["country"].removeprefix("the ") for code in present},
           "questions": questions, "asked_of": blocks}
    (here / f"issp{year}_questions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (here / f"issp{year}_attributes.json").write_text(json.dumps(reference, ensure_ascii=False, indent=1))
    attrs = [k for k in table.columns if k in PERSONA_KEY.values()]
    print(f"{len(table)} respondents, {len(attrs)} attributes, {len(questions)} questions in "
          f"{qb.block.nunique()} blocks, {len(present)} countries -> {here}/issp{year}_respondents.parquet, "
          f"issp{year}_questions.json, issp{year}_attributes.json")
