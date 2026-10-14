"""Turn a stored ISSP record into a request the model reads.

A stored record (data/issp<year>/issp<year>.jsonl) holds the persona as fields and each question in a fixed order:
persona fields such as Country, Sex, Age, Education; `score` levels low -> high; `choice` options in questionnaire order.

- `augment(record, rng)` gives a fresh training view: the persona written in a random presentation (`persona_text`:
  layout, attribute names, value wording, line order; see below), each `score` scale reversed with probability
  `p_reverse`, `choice` options in random order. `noul` is left as is (Kev always shows no, yes).
- `canonical(record)` gives the fixed view used for calibration and evaluation: 'Key: value' lines in stored order, so
  every evaluation sees exactly the same input. Scales must be low -> high at inference, because Kev's `score` output
  is the expected level index.

Every view keeps each answer's probability attached to its answer text: when options move, `target` and `label`
move with them. Every persona presentation describes exactly the same people: only the wording changes (wording
variation is training signal, PLAN-respondent-sampling.md step 3b). Attribute values keep the data's own text except
where an alternative means exactly the same (sex, age, yes / no).
"""


import json
import re

# ---------------------------------------------------------------- persona presentation

# other names for an attribute, each meaning the same; the stored name first. Names not listed are always used as stored
# (e.g. the filter lines "Lives with a spouse or partner").
KEY_NAMES = {
    "Country": ["Country", "Country of residence"],
    "Sex": ["Sex", "Gender"],
    "Education": ["Education", "Education level", "Highest level of education", "Highest completed education"],
    "Religion": ["Religion", "Religious affiliation", "Religious denomination"],
    "Attendance of religious services": ["Attendance of religious services", "Religious service attendance",
                                         "How often attends religious services"],
    "Party voted for in the last general election": ["Party voted for in the last general election",
                                                     "Party voted for in the last national election",
                                                     "Vote in the last general election"],
    "Voted party's left-right position": ["Voted party's left-right position",
                                          "Left-right position of the party voted for"],
    "Voted in the last general election": ["Voted in the last general election", "Voted in the last national election"],
    "Household income": ["Household income", "Income of the household"],
    "Region": ["Region", "Region of residence"],
    "Place of living": ["Place of living", "Type of place of residence", "Area of residence"],
    "Marital status": ["Marital status", "Legal marital status", "Legal partnership status"],
    "Main activity": ["Main activity", "Main status", "Current activity"],
    "Trade union membership": ["Trade union membership", "Trade union member", "Union membership"],
    "Self-placed social position (1 = bottom, 10 = top)": [
        "Self-placed social position (1 = bottom, 10 = top)",
        "Social position on a scale from 1 (bottom) to 10 (top)",
        "Own position in society, from 1 (lowest) to 10 (highest)"],
    "Ancestry": ["Ancestry", "Ethnic group", "Ethnic background"],
}
SEX_WORDS = {"Male": ["Male", "male", "Man", "man"], "Female": ["Female", "female", "Woman", "woman"]}
LAYOUTS = ["lines", "bullets", "one line", "json", "prose"]
# prose: (subject, "to be", "to live", possessive)
PERSONS = [("This person", "is", "lives", "Their"), ("The respondent", "is", "lives", "Their"),
           ("I", "am", "live", "My"), ("You", "are", "live", "Your")]
# attributes whose names read as a noun phrase in "Their <name> is <value>." (the others are written "<name>: <value>.")
NOUN_KEYS = {"Education", "Religion", "Household income", "Region", "Place of living", "Marital status",
             "Main activity", "Ancestry"}


AGE_TOP = re.compile(r"^(\d+)( years)? or (older|more)$")   # a top-coded age: '75 or older', '89 years or more'


def is_band(age):
    """'45-59' or '89 years or more' (an age range) rather than an age in years."""
    return not age.isdigit()


def key_name(key, value, rng):
    """One name for the attribute."""
    if key == "Age":
        return rng.choice(["Age", "Age group", "Age band"] if is_band(value) else ["Age", "Age in years"])
    return rng.choice(KEY_NAMES.get(key, [key]))


def value_text(key, value, rng):
    """One wording of the value that means exactly the same."""
    if key == "Sex" and value in SEX_WORDS:
        return rng.choice(SEX_WORDS[value])
    if key == "Age" and value.isdigit():
        return rng.choice([value, f"{value} years", f"{value} years old"])
    if key == "Age" and (top := AGE_TOP.match(value)):
        return rng.choice([value, f"{top.group(1)} years or {top.group(3)}", f"{top.group(1)} or {top.group(3)}",
                           f"{top.group(1)}+"])
    if key == "Age" and re.fullmatch(r"\d+-\d+", value):
        low, high = value.split("-")
        return rng.choice([value, f"{low} to {high}", f"{low}-{high} years", f"{low}\u2013{high}"])
    if value in ("Yes", "No"):
        return rng.choice([value, value.lower()])
    return value


def country_phrase(country):
    """'United States' -> 'the United States' (in prose)."""
    needs_the = country.startswith("United ") or country.endswith(("Philippines", "Netherlands", "Republic"))
    return ("the " if needs_the else "") + country


def sentence(key, name, value, raw_value, person):
    """One prose sentence for one attribute."""
    subject, be, live, possessive = person
    if key == "Country":
        return f"{subject} {live} in {country_phrase(raw_value)}."
    if key == "Age" and raw_value.isdigit():
        return f"{subject} {be} {raw_value} years old."
    if key == "Age" and (top := AGE_TOP.match(raw_value)):
        return f"{subject} {be} aged {top.group(1)} or {top.group(3)}."
    if key == "Age":
        return f"{subject} {be} aged {value}."
    if key == "Sex" and raw_value in SEX_WORDS:
        return f"{subject} {be} {sex_phrase(value)}."
    if key in NOUN_KEYS:
        return f"{possessive} {name[0].lower() + name[1:]} is {value}."
    return f"{name}: {value}."


def sex_phrase(value):
    """'a woman' / 'female' in prose, following the value wording already drawn."""
    return f"a {value.lower()}" if value.lower() in ("man", "woman") else value.lower()


def persona_text(persona, rng):
    """The persona in a random presentation: layout, attribute names, value wording and order drawn with `rng`."""
    items = list(persona.items())
    rng.shuffle(items)
    pairs = [(key, key_name(key, value, rng), value_text(key, value, rng), value) for key, value in items]
    layout = rng.choice(LAYOUTS)
    if layout == "prose":
        person = rng.choice(PERSONS)
        body = " ".join(sentence(key, name, value, raw, person) for key, name, value, raw in pairs)
    elif layout == "json":
        body = json.dumps({name: value for _, name, value, _ in pairs}, ensure_ascii=False, indent=rng.choice([None, 2]))
    elif layout == "one line":
        body = rng.choice(["; ", ", ", " | "]).join(f"{name}: {value}" for _, name, value, _ in pairs)
    else:
        bullet = rng.choice(["- ", "* ", "\u2022 "]) if layout == "bullets" else ""
        separator = rng.choice([": ", ": ", " - ", " = "])
        body = "\n".join(f"{bullet}{name}{separator}{value}" for _, name, value, _ in pairs)
    return body


def render_state(persona, rng=None):
    """Persona fields as 'Key: value' lines in stored order (canonical view), or a random presentation when an rng is
    given (training view)."""
    if rng is not None:
        return persona_text(persona, rng)
    return "\n".join(f"{key}: {value}" for key, value in persona.items())


def reverse_score(question):
    """The same `score` question with its levels in the opposite order.

    Level i becomes level K-1-i, so target keys and the label are renumbered the same way."""
    n_levels = len(question["criteria"])
    return {**question,
            "criteria": question["criteria"][::-1],
            "label": n_levels - 1 - question["label"],
            "target": {str(n_levels - 1 - int(level)): share for level, share in question["target"].items()}}


def shuffle_choice(question, rng):
    """The same `choice` question with its options in random order. Target and label are keyed by option name,
    so they need no change."""
    names = list(question["criteria"])
    rng.shuffle(names)
    return {**question, "criteria": {name: question["criteria"][name] for name in names}}


def augment_question(question, rng, p_reverse):
    """A fresh view of one question."""
    if question["type"] == "score" and rng.random() < p_reverse:
        return reverse_score(question)
    if question["type"] == "choice":
        return shuffle_choice(question, rng)
    return question


def augment(record, rng, p_reverse=0.5):
    """A fresh training view of one stored record: {"state": text, "questions": {...}}."""
    return {"state": render_state(record["persona"], rng),
            "questions": {v: augment_question(q, rng, p_reverse) for v, q in record["questions"].items()}}


def canonical(record):
    """The fixed view of one stored record, used for calibration and evaluation."""
    return {"state": render_state(record["persona"]), "questions": record["questions"]}
