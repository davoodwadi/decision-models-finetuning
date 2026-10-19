# Data

Each source dataset sits in its own folder under `data/`. Raw survey downloads from GESIS sit in `data/<source>/raw/`, while manually verified metadata and processed outputs sit directly in each dataset folder. The `data_check_<year>.ipynb` notebooks in `code/` read the raw Stata files alongside the manually curated `question_bank.csv` and `label_fixes.csv` files to produce `questions_verified.csv` and `issp<year>_US_long.parquet`.

| Folder | Dataset | Topic | Unique respondents | Train questions | Valid questions | Test questions | Total questions | Train responses | Valid responses | Test responses | Total responses |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `issp2015/` | `issp2015` | Work orientations IV | 1,474 | 71 | 5 | 6 | 82 | 63,598 | 3,942 | 4,876 | 72,416 |
| `issp2016/` | `issp2016` | Role of government V | 1,384 | 49 | 4 | 7 | 60 | 66,101 | 5,446 | 9,496 | 81,043 |
| `issp2017/` | `issp2017` | Social networks and social resources | 1,173 | 49 | 4 | 7 | 60 | 56,702 | 4,644 | 8,154 | 69,500 |
| `issp2018/` | `issp2018` | Religion IV | 1,175 | 53 | 4 | 5 | 62 | 60,432 | 4,570 | 5,717 | 70,719 |
| `issp2019/` | `issp2019` | Social inequality V | 1,852 | 39 | 4 | 6 | 49 | 67,420 | 7,000 | 10,543 | 84,963 |
| `issp2020/` | `issp2020` | Environment IV | 1,847 | 41 | 6 | 8 | 55 | 71,950 | 10,526 | 14,122 | 96,598 |
| `issp2021/` | `issp2021` | Health and health care II | 1,146 | 44 | 4 | 10 | 58 | 47,583 | 4,365 | 10,845 | 62,793 |
| `issp2022/` | `issp2022` | Family and changing gender roles V | 1,126 | 42 | 6 | 7 | 55 | 38,968 | 5,655 | 6,680 | 51,303 |
| `issp2023/` | `issp2023` | National identity and citizenship | 1,589 | 44 | 8 | 8 | 60 | 65,256 | 12,111 | 12,024 | 89,391 |
| `issp2024/` | `issp2024` | Digital societies I | 1,590 | 62 | 5 | 8 | 75 | 72,281 | 5,942 | 10,535 | 88,758 |
| **Total** | **ISSP US** | **All 10 waves (2015 to 2024)** | **14,356** | **494** | **50** | **72** | **616** | **610,291** | **64,201** | **92,992** | **767,484** |

## Manually curated files

Each ISSP folder contains a manually verified `question_bank.csv` (and the exported `questions_verified.csv`) defining question wording, question types (`score`, `choice`, `noul`, or `exclude`), and scale direction (`reversed`). For waves where the Stata `.dta` file truncates value labels due to character limits (e.g., 2015, 2017, 2020, and 2021), `label_fixes.csv` provides manually verified label completions from the GESIS codebooks. Country names, nationality terms, and parliament names used to fill question placeholders are stored in `issp_countries.csv`.

## Train, validation, and test split (`code/splits/split.json`)

Across the 10 manually verified US waves (`issp2015_US_long.parquet` through `issp2024_US_long.parquet`, totalling 14,356 US respondents and 767,484 respondent-question observations), all 616 answered US questions are partitioned in `code/splits/split.json` into 494 `train` questions (80.2%), 50 `valid` questions (8.1%), and 72 `test` questions (11.7%). Standalone questions, two-item blocks, and cross-year recurring questions are assigned strictly to `train`. For multi-item batteries ($\ge 3$ questions sharing a stem, question type, and answer scale), one non-repeating item is held out for `test` and, for batteries of $\ge 4$ items, one additional non-repeating item is held out for `valid`, leaving at least two sibling items per battery in `train`.
