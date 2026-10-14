"""Build the SubPOP sets (US subpopulations' answer distributions) in the same record format as the ISSP data.

- data/subpop/subpop100.jsonl  from SubPOP's training split (Pew American Trends Panel): the calibration set.
- data/subpop/subpop_eval.jsonl from SubPOP's evaluation split (US General Social Survey): the test set.
(train_issp.py: CALIBRATION_DATASET, TEST_DATASET; the two can be swapped there.)

Steps:
  1. Drop non-answers: keep the options that have a response share, remove "Refused" (and, in the evaluation split,
     "Don't know/can't remember") and renormalise the shares. Questions left with fewer than 2 answers are dropped.
  2. Likert vs choice.
     Training split: it carries no answer-order information, so the 100 most common answer-option sets were
     classified by hand into LIKERT (ordered; listed from low to high -> `score` questions with levels in that order),
     CHOICE (two answers, or a scale with an answer outside it such as "Not sure" -> `choice` questions in SubPOP's
     order) and NOT_ANSWERS (dropped). Only questions whose answer set was classified are kept.
     Evaluation split: every question has an `ordinal` value per answer. Three or more answers with positive, distinct
     values -> `score`, levels sorted by that value; an answer valued -1 lies outside the scale ("Depends", "Other")
     -> `choice`; two answers -> `choice`. NOMINAL lists the questions whose values only number unordered answers
     (marital status, ...) and FIXED_ORDER the one with miscoded values.
  3. Questions that are also asked in the ISSP training data (ISSP_OVERLAP) are left out of both sets, so calibration
     and test only contain questions the model has not been trained on. See data/subpop/README.md.
  4. One record per subpopulation and up to QUESTIONS_PER_RECORD of its questions.
     Persona = "Country: United States" + the subpopulation's one attribute.
"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent                      # data/subpop/
OUT_TRAIN = HERE / "subpop100.jsonl"               # SubPOP training split
OUT_EVAL = HERE / "subpop_eval.jsonl"              # SubPOP evaluation split
QUESTIONS_PER_RECORD = 7

ATTRIBUTE = {"CREGION": "Region", "EDUCATION": "Education", "INCOME": "Income", "POLIDEOLOGY": "Political ideology",
             "POLPARTY": "Political party", "RACE": "Race", "RELIG": "Religion", "SEX": "Sex"}

NOT_ANSWERS = [{"Not asked", "Did not refuse"}]          # survey administration, not an opinion

# SubPOP questions that ask the same thing as a question in the ISSP training data (same question, even if worded
# or scaled differently): left out of calibration and test. Found by comparing every SubPOP question's words with
# every ISSP item, then reviewing the closest matches by hand; README.md also lists related questions that were kept.
ISSP_OVERLAP = {
    # evaluation split (GSS)
    "fechld": "ISSP 2022 v1 (a working mother can establish just as warm a relationship)",
    "fepresch": "ISSP 2022 v2 (a pre-school child is likely to suffer if the mother works)",
    "letin1a": "ISSP 2023 v31 (number of immigrants should be increased/reduced)",
    "xmarsex": "ISSP 2018 v3 (sexual relations with someone other than the spouse)",
    "homosex": "ISSP 2018 v4 (sexual relations between two adults of the same sex)",
    "fair": "ISSP 2017 v34 (people would try to take advantage of you or be fair)",
    "trust": "ISSP 2017 v35, 2018 v78, 2019 v56, 2020 v10 (most people can be trusted)",
    "pray": "ISSP 2018 v43 (how often do you pray)",
    "postlife": "ISSP 2018 v22 (belief in life after death)",
    "bible": "ISSP 2018 v65 (feelings about the Bible)",
    "reborn": "ISSP 2018 v64 (born-again experience)",
    "relpersn": "ISSP 2018 v48 (describe yourself as religious)",
    "health": "ISSP 2015 v97, 2017 v54, 2018 v63, 2021 v51 (self-rated health)",
    "happy": "ISSP 2018 v1, 2021 v1, 2022 v56 (how happy are you)",
    "class": "ISSP 2019 v61 (which social class you belong to)",
    "satjob": "ISSP 2015 v44 (how satisfied are you in your job)",
    "finalter": "ISSP 2015 v93 (financial situation better or worse than some years ago)",
    "conlegis": "ISSP 2018 v7 (confidence in parliament)",
    "conbus": "ISSP 2018 v8 (confidence in business and industry)",
    "conclerg": "ISSP 2018 v9 (confidence in churches and religious organizations)",
    "conjudge": "ISSP 2018 v10 (confidence in courts and the legal system)",
    "coneduc": "ISSP 2018 v11 (confidence in schools and the educational system)",
    "natenvir": "ISSP 2016 v13 (government spending on the environment)",
    "natheal": "ISSP 2016 v14 (government spending on health)",
    "nateduc": "ISSP 2016 v16 (government spending on education)",
    "natarms": "ISSP 2016 v17 (government spending on the military and defence)",
    "natsoc": "ISSP 2016 v18 (government spending on old age pensions / Social Security)",
    # training split (Pew)
    "NATIONALITY_BORN_W124": "ISSP 2023 v1 (being born in the country is important for being truly American)",
    "NATIONALITY_RELIG_W124": "ISSP 2023 v3 (being of the dominant religion is important for being truly American)",
}

# evaluation split: answers that are not answers to the question
EVAL_NOT_ANSWERS = {"Refused", "Don't know/can't remember"}
# evaluation split: questions whose `ordinal` values only number unordered answers -> `choice`
NOMINAL = {"marital", "dwelown16", "dwelown", "union1", "hunt1", "getahead", "bible", "vote16"}
# evaluation split: a question whose `ordinal` values are miscoded -> its answers low to high
FIXED_ORDER = {"nataid": ["Too little", "About right", "Too much"]}

# Answer sets among the 100 most common that are not ordered scales: two answers, or an answer outside the scale.
CHOICE = [
    ['Yes', 'No'],
    ['Favor', 'Oppose'],
    ['Very favorable', 'Mostly favorable', 'Mostly unfavorable', 'Very unfavorable', 'Never heard of this person'],
    ['Not selected', 'Selected'],
    ['Strongly agree with the Republican Party', 'Somewhat agree with the Republican Party',
     'Somewhat agree with the Democratic Party', 'Strongly agree with the Democratic Party', 'Don’t agree with either party'],
    ['Yes, use this', 'No, do not use this'],
    ['Approve', 'Disapprove'],
    ['Comfortable doing this', 'Uncomfortable doing this'],
    ['Favor', 'Oppose', 'Not sure'],
    ['Satisfied', 'Dissatisfied'],
    ['Yes, I have done this in the past year', 'No, I have not done this in the past year'],
    ['Yes, I have', 'No, I have not'],
    ['Yes, this is part of the mainstream news media', 'No, this is not part of the mainstream news media',
     'Do not know enough about this news source to decide'],
    ['Yes, use this', 'No, don’t use this'],
    ['Almost all of the time', 'Most of the time', 'Some of the time', 'Hardly ever', 'Not sure'],
    ['A lot better', 'Somewhat better', 'Neither better nor worse', 'Somewhat worse', 'A lot worse',
     'The president’s gender doesn’t matter'],
    ['Yes, have done this', 'No, have not done this'],
    ['Yes, has happened to me since the outbreak', 'No, has not happened to me since the outbreak'],
    ['Very strongly', 'Not so strongly'],
    ['Yes, have done this since February', 'No, have not done this since February'],
    ['Yes, have done this to cover lost wages or salary', 'No, have not done this to cover lost wages or salary'],
    ['Positive effect', 'Negative effect'],
    ['Definitely would happen', 'Probably would happen', 'Probably would not happen', 'Definitely would not happen', 'Not sure'],
    ['Yes, makes me feel this way', 'No, does not make me feel this way'],
    ['Strongly approve', 'Somewhat approve', 'Somewhat disapprove', 'Strongly disapprove', 'Not sure'],
    ['Yes, a responsibility of the federal government to provide for all Americans',
     'No, not the responsibility of the federal government to provide'],
    ['Yes, have heard of this site or app', 'No, have not heard of this site or app'],
    ['Very favorable', 'Somewhat favorable', 'Neither favorable nor unfavorable', 'Somewhat unfavorable', 'Very unfavorable',
     'Don’t know enough to say'],
    ['Yes, know someone', 'No, do not know someone'],
]
CHOICE_SETS = {frozenset(options) for options in CHOICE}

# Ordered answer sets among the 100 most common, each listed from low to high.
LIKERT = [
    ["Not at all important", "Not too important", "Somewhat important", "Very important"],
    ["Not at all important", "Not too important", "Somewhat important", "Very important", "Extremely important"],
    ["Not important", "Somewhat important", "Very important but not the most important", "One of the most important things"],
    ["Not at all well", "Not too well", "Somewhat well", "Very well"],
    ["Not at all well", "Not too well", "Fairly well", "Very well"],
    ["Not at all well", "Not too well", "Somewhat well", "Very well", "Extremely well"],
    ["Nothing at all", "A little", "A lot"],
    ["Nothing at all", "Not much", "Some", "A lot"],
    ["None at all", "Not too much", "Some", "A lot"],
    ["None at all", "Only a little", "Some", "A lot"],
    ["None at all", "Not much", "Some", "A lot"],
    ["Not at all", "Not too much", "Some", "A lot"],
    ["Not at all", "Not too much", "A fair amount", "A great deal"],
    ["Not at all", "Not too much", "Some", "A great deal"],
    ["Not at all", "Not much", "Some", "A great deal"],
    ["None", "A little", "Some", "Quite a bit", "A great deal"],
    ["No confidence at all", "Not too much confidence", "A fair amount of confidence", "A great deal of confidence"],
    ["No confidence at all", "Not too much", "A fair amount", "A great deal"],
    ["Not at all confident", "Not too confident", "Somewhat confident", "Very confident"],
    ["Not at all concerned", "Not too concerned", "Somewhat concerned", "Very concerned"],
    ["Not at all concerned", "Not too concerned", "Somewhat concerned", "Very concerned", "Extremely concerned"],
    ["Not at all worried", "Not too worried", "Somewhat worried", "Very worried", "Extremely worried"],
    ["Not at all effective", "Not too effective", "Somewhat effective", "Very effective"],
    ["Not at all effective", "Not too effective", "Somewhat effective", "Very effective", "Extremely effective"],
    ["Very ineffective", "Somewhat ineffective", "Somewhat effective", "Very effective"],
    ["Not at all likely", "Not too likely", "Somewhat likely", "Very likely", "Extremely likely"],
    ["Not at all closely", "Not too closely", "Fairly closely", "Very closely"],
    ["Not at all closely", "Not too closely", "Somewhat closely", "Very closely"],
    ["Not a problem at all", "A small problem", "A moderately big problem", "A very big problem"],
    ["Not a problem", "A small problem", "A moderate problem", "A big problem"],
    ["Not a problem", "A minor problem", "A major problem"],
    ["Not a problem in my local community", "Minor problem in my local community", "Major problem in my local community"],
    ["Not a reason", "Minor reason", "Major reason"],
    ["No role at all", "Minor role", "Major role"],
    ["Not a source", "Minor source", "Major source"],
    ["Not a threat", "A minor threat", "A major threat"],
    ["Not a factor", "Minor factor", "Major factor"],
    ["Not much of an impact at all", "A minor impact", "A major impact"],
    ["Should not be done", "Not too important", "Important but lower priority", "Top priority"],
    ["Should not be done", "Should not be too important", "Should be an important but lower priority", "Should be a top priority"],
    ["Strongly oppose", "Somewhat oppose", "Somewhat favor", "Strongly favor"],
    ["Strongly oppose", "Oppose", "Neither favor nor oppose", "Favor", "Strongly favor"],
    ["Very unfavorable", "Mostly unfavorable", "Mostly favorable", "Very favorable"],
    ["Very unfavorable", "Somewhat unfavorable", "Somewhat favorable", "Very favorable"],
    ["Poor", "Only fair", "Good", "Excellent"],
    ["Very bad", "Somewhat bad", "Somewhat good", "Very good"],
    ["Very bad job", "Somewhat bad job", "Somewhat good job", "Very good job"],
    ["Very bad description", "Somewhat bad description", "Somewhat good description", "Very good description"],
    ["Very negative", "Somewhat negative", "Neither positive nor negative", "Somewhat positive", "Very positive"],
    ["Mostly negative", "About an even mix of positive and negative", "Mostly positive"],
    ["Hurts a lot", "Hurts a little", "Neither helps nor hurts", "Helps a little", "Helps a lot"],
    ["Dislike a lot", "Dislike a little", "Neither like nor dislike", "Like a little", "Like a lot"],
    ["Completely unacceptable", "Somewhat unacceptable", "Neither acceptable nor unacceptable", "Somewhat acceptable",
     "Completely acceptable"],
    ["A lot less likely to support", "Somewhat less likely to support", "Neither more nor less likely to support",
     "Somewhat more likely to support", "A lot more likely to support"],
    ["Very concerned", "Somewhat concerned", "Equal excitement and concern", "Somewhat excited", "Very excited"],
    ["Worse", "About the same as now", "Better"],
    ["Easier for young adults today", "About the same as it was for their parents", "Harder for young adults today"],
    ["White people are treated less fairly than Black people", "Both are treated about equally",
     "Black people are treated less fairly than White people"],
    ["Too little", "About the right amount", "Too much"],
    ["Too little attention", "Right amount of attention", "Too much attention"],
    ["Too little attention", "About the right amount of attention", "Too much attention"],
    ["Not enough focus", "About the right amount of focus", "Too much focus"],
    ["Does too little", "Does about the right amount", "Does too much"],
    ["Never", "Rarely", "Sometimes", "Often"],
    ["Never", "Rarely", "Some of the time", "Most of the time", "All of the time"],
    ["Never", "Rarely", "Sometimes", "Almost every day", "Every day"],
    ["Never", "Less than monthly", "Monthly", "Weekly", "Daily"],
    ["Rarely or none of the time (less than 1 day)", "Some or a little of the time (1-2 days)",
     "Occasionally or a moderate amount of time (3-4 days)", "Most or all of the time (5-7 days)"],
]
LIKERT_ORDER = {frozenset(levels): levels for levels in LIKERT}   # answer set (any order) -> answers low to high


def answers(row, not_answers=("Refused",)):
    """The question's answers and their shares, without non-answers: options with a share, minus `not_answers`."""
    shares = dict(zip(row.options[:len(row.responses)], row.responses))
    for option in not_answers:
        shares.pop(option, None)
    total = sum(shares.values())
    return {option: share / total for option, share in shares.items()}


def score_question(row, shares, levels):
    """A `score` question with the given levels, low -> high."""
    return {"type": "score", "instructions": row.question, "criteria": levels,
            "target": {str(i): round(shares[level], 4) for i, level in enumerate(levels)},
            "label": max(range(len(levels)), key=lambda i: shares[levels[i]]), "src": "subpop"}


def choice_question(row, shares):
    """A `choice` question with the answers in SubPOP's order."""
    return {"type": "choice", "instructions": row.question, "criteria": {option: None for option in shares},
            "target": {option: round(share, 4) for option, share in shares.items()},
            "label": max(shares, key=shares.get), "src": "subpop"}


def eval_question(row, shares):
    """Evaluation split: `score` when the answers are ordered (see step 2 above), otherwise `choice`."""
    if row.qkey in FIXED_ORDER:
        return score_question(row, shares, FIXED_ORDER[row.qkey])
    value = dict(zip(row.options, row.ordinal))
    values = [value[option] for option in shares]
    ordered = len(shares) >= 3 and min(values) > 0 and len(set(values)) == len(values) and row.qkey not in NOMINAL
    return score_question(row, shares, sorted(shares, key=value.get)) if ordered else choice_question(row, shares)


def write_records(questions_by_group, out):
    """One record per subpopulation and up to QUESTIONS_PER_RECORD of its questions."""
    records = []
    for (attribute, group), questions in questions_by_group.items():
        items = list(questions.items())
        for start in range(0, len(items), QUESTIONS_PER_RECORD):
            records.append({"group": f"SUBPOP/{attribute}/{group}", "country_code": "US",
                            "block": f"S{start // QUESTIONS_PER_RECORD}",
                            "persona": {"Country": "United States", ATTRIBUTE[attribute]: group},
                            "questions": dict(items[start:start + QUESTIONS_PER_RECORD])})
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    types = pd.Series([q["type"] for r in records for q in r["questions"].values()]).value_counts()
    print(f"{len(records)} records, targets by type: {types.to_dict()} -> {out}")


def build_train(rows):
    """subpop100.jsonl from the training split (hand-classified answer sets)."""
    # the hand classification must match the data exactly and cover the 100 most common answer sets
    # (the top 100 are counted by answer list in SubPOP's order, as they were reviewed)
    answer_lists = pd.Series([tuple(answers(row)) for row in rows.drop_duplicates("qkey").itertuples()])
    classified = set(LIKERT_ORDER) | CHOICE_SETS | {frozenset(s) for s in NOT_ANSWERS}
    not_in_data = [sorted(s) for s in classified if s not in set(answer_lists.map(frozenset))]
    assert not not_in_data, f"hand-classified answer sets not found in the data (typo?): {not_in_data}"
    unclassified_top = [list(a) for a in answer_lists.value_counts().index[:100] if frozenset(a) not in classified]
    assert not unclassified_top, f"top-100 answer sets missing from the classification: {unclassified_top}"

    questions_by_group, kept, left_out, overlap = {}, set(), set(), set()
    for row in rows.itertuples():
        if row.qkey in ISSP_OVERLAP:
            overlap.add(row.qkey)
            continue
        shares = answers(row)
        if frozenset(shares) in LIKERT_ORDER:
            question = score_question(row, shares, LIKERT_ORDER[frozenset(shares)])
        elif frozenset(shares) in CHOICE_SETS:
            question = choice_question(row, shares)
        else:
            left_out.add(row.qkey)                       # not hand-classified, a non-opinion, or < 2 answers
            continue
        kept.add(row.qkey)
        questions_by_group.setdefault((row.attribute, row.group), {})[row.qkey] = question
    print(f"training split: {len(kept)} questions kept, {len(left_out)} left out (answer set not hand-classified, "
          f"not an opinion, or fewer than 2 answers), {len(overlap)} left out as ISSP overlap")
    write_records(questions_by_group, OUT_TRAIN)


def build_eval(rows):
    """subpop_eval.jsonl from the evaluation split (answer order from its `ordinal` values)."""
    unknown = (set(NOMINAL) | set(FIXED_ORDER) | {k for k in ISSP_OVERLAP if k.islower()}) - set(rows.qkey)
    assert not unknown, f"question keys not in the evaluation split (typo?): {sorted(unknown)}"
    questions_by_group, kept, overlap = {}, set(), set()
    for row in rows.itertuples():
        if row.qkey in ISSP_OVERLAP:
            overlap.add(row.qkey)
            continue
        shares = answers(row, EVAL_NOT_ANSWERS)
        if len(shares) < 2:
            continue
        kept.add(row.qkey)
        questions_by_group.setdefault((row.attribute, row.group), {})[row.qkey] = eval_question(row, shares)
    print(f"evaluation split: {len(kept)} questions kept, {len(overlap)} left out as ISSP overlap")
    write_records(questions_by_group, OUT_EVAL)


def main():
    load = lambda split: pd.read_json(hf_hub_download("jjssuh/subpop", f"subpop_{split}.jsonl", repo_type="dataset"),
                                      lines=True)                                   # gated: needs HF access
    build_train(load("train"))
    build_eval(load("eval"))


if __name__ == "__main__":
    main()
