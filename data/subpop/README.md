# SubPOP: calibration and test sets

US subpopulations' answer distributions, used **only for calibration and testing** (never trained on):

| Dataset | From | Role in `train_issp.py` | Questions | Targets |
|---|---|---|---|---|
| `subpop100` | SubPOP training split (Pew American Trends Panel, waves W68–W131) | calibration (`CALIBRATION_DATASET`): picks the best checkpoint | 1,873 | 41,200 |
| `subpop_eval` | SubPOP evaluation split (US General Social Survey) | test (`TEST_DATASET`) | 106 | 2,332 |

Swapping the two roles means swapping the two names in `train_issp.py`. Training uses the ISSP data only, so both
sets test the model on a different survey, different questions and different persona attributes (political party,
race, religion, income, region) than it was trained on.

## Source
- Hugging Face dataset **`jjssuh/subpop`**, files `subpop_train.jsonl` and `subpop_eval.jsonl` (gated: needs Hugging
  Face access). Downloaded by the build script; nothing is stored in this folder.
- 22 US subpopulations, each defined by **one** attribute: region (Northeast, South), education (college graduate,
  less than high school), income ($100,000 or more, less than $30,000), political ideology (liberal, moderate,
  conservative), political party (Democrat, Republican), race (Asian, Black, Hispanic, White), religion (Atheist, Hindu,
  Jewish, Muslim, Protestant), sex. There are no crossed groups (no "Republican women"). Only answer shares are given,
  no respondent counts, so no sampling-noise benchmark can be computed.

## Rebuild
From the project root:
```bash
kev/.venv/bin/python data/subpop/build_subpop.py
```
It writes both `subpop100.jsonl` and `subpop_eval.jsonl`.

## Record format
Same as the ISSP data, without `n`: one record per subpopulation and up to 7 of its questions.
- `persona`: `Country: United States` plus the subpopulation's one attribute, e.g. `Political party: Republican`.
- `group`: e.g. `SUBPOP/POLPARTY/Republican` (an id, not a path).
- `questions`: per item `type`, `criteria`, `instructions`, `target` (share of each answer), `label` (most common answer), `src`.

## Decisions
1. **Non-answers dropped**: "Refused" (both splits) and "Don't know/can't remember" (evaluation split) are removed and
   the shares renormalised. Training-split answer sets that are survey administration, not opinions (`NOT_ANSWERS`:
   "Not asked", "Did not refuse"), are dropped. Questions left with fewer than 2 answers are dropped.
2. **Likert vs choice, training split (`subpop100`), by hand**: the training split has no answer-order information
   (its ordinal field is the same for every question). The 100 most common answer-option sets (counted by ordered
   answer list) were classified by hand in `build_subpop.py`: `LIKERT` (ordered, listed low → high → `score`),
   `CHOICE` (two answers, or a scale with an answer outside it such as "Not sure" → `choice`, SubPOP's order),
   `NOT_ANSWERS` (dropped). **Only hand-classified questions are kept**: 1,354 questions with other answer sets are
   left out. The script checks that every classified set exists in the data and that the 100 most common sets are
   all classified.
3. **Likert vs choice, evaluation split (`subpop_eval`), from the data**: every answer has an `ordinal` value. Three or
   more answers with positive, distinct values → `score`, levels sorted by that value; an answer valued -1 is outside
   the scale ("Depends", "Other", "Lived in institution") → `choice`; two answers → `choice`. Hand corrections:
   `NOMINAL` (values that only number unordered answers: marital status, home ownership, union membership, hunting,
   getting ahead, the Bible, voting in 2016 → `choice`) and `FIXED_ORDER` (`nataid`, whose values are miscoded).
   Result: 44 `score`, 62 `choice` questions.
4. **Questions also asked in the ISSP training data are left out** of both sets (`ISSP_OVERLAP` in
   `build_subpop.py`), so calibration and test contain only questions the model was not trained on. "The same
   question" means the same thing is asked, even if the wording or answer scale differs. How they were found: every
   SubPOP question's words were compared with every ISSP item (2015–2024), and the closest matches were reviewed by
   hand. This is a judgement call; the lists below make it checkable.

   **Left out: evaluation split (27 of 133 questions)**

   | SubPOP question | Same question in ISSP |
   |---|---|
   | `fechld` | 2022 v1: a working mother can establish just as warm a relationship |
   | `fepresch` | 2022 v2: a pre-school child is likely to suffer if the mother works |
   | `letin1a` | 2023 v31: number of immigrants should be increased/reduced |
   | `xmarsex` | 2018 v3: sexual relations with someone other than the spouse |
   | `homosex` | 2018 v4: sexual relations between two adults of the same sex |
   | `fair` | 2017 v34: people would try to take advantage of you or be fair |
   | `trust` | 2017 v35, 2018 v78, 2019 v56, 2020 v10: most people can be trusted |
   | `pray` | 2018 v43: how often do you pray |
   | `postlife` | 2018 v22: belief in life after death |
   | `bible` | 2018 v65: feelings about the Bible |
   | `reborn` | 2018 v64: born-again experience |
   | `relpersn` | 2018 v48: describe yourself as religious |
   | `health` | 2015 v97, 2017 v54, 2018 v63, 2021 v51: self-rated health |
   | `happy` | 2018 v1, 2021 v1, 2022 v56: how happy are you |
   | `class` | 2019 v61: which social class you belong to |
   | `satjob` | 2015 v44: how satisfied are you in your job |
   | `finalter` | 2015 v93: financial situation better or worse than some years ago |
   | `conlegis`, `conbus`, `conclerg`, `conjudge`, `coneduc` | 2018 v7–v11: confidence in parliament, business and industry, churches, courts, schools |
   | `natenvir`, `natheal`, `nateduc`, `natarms`, `natsoc` | 2016 v13, v14, v16, v17, v18: government spending on the environment, health, education, the military, pensions |

   **Left out: training split (2 of 1,875 questions)**

   | SubPOP question | Same question in ISSP |
   |---|---|
   | `NATIONALITY_BORN_W124` | 2023 v1: being born in the country is important for being truly [nationality] |
   | `NATIONALITY_RELIG_W124` | 2023 v3: being of the dominant religion is important for being truly [nationality] |

   **Related but kept** (a similar topic, but a different question):
   - `abpoor` and the other abortion items (`abdefect`, `abnomore`, …): GSS asks whether abortion should be *legal*;
     ISSP 2018 v5 asks whether it is *wrong* (low income only).
   - `attend`: GSS asks how often you attend religious services now; ISSP 2018 v40–v42 ask about childhood.
   - `tax`: GSS asks about your own federal income tax; ISSP 2016 v51–v53 about taxes for high, middle, low incomes.
   - `fepol` (men better suited emotionally for politics): ISSP 2022 v57 asks who is better suited as a cabinet minister.
   - The other GSS spending items (`natspac`, `natcity`, `natdrug`, …) and confidence items (`conarmy`, `confed`,
     `confinan`, `conlabor`, `conmedic`, `conpress`, `consci`, `contv`): areas ISSP does not ask about.
   - The civil-liberties items (`spkath`, `colrac`, …): ISSP 2016 v4–v5 ask about revolutionaries, a different group.
   - `GOD_W96`, `GOD_W106` (believe in God, yes/no): ISSP 2018 v20 asks which of six statements about God fits.
   - `NATIONALITY_CUSTOM_W124`, `NATIONALITY_LANG_W124` (sharing American values, speaking English): not in ISSP 2023.
5. **No held-out part**: both files are used whole for their role.
