"""Dataset-agnostic loaders and Kev view builders for standardized `persona_spec.yaml` parquets."""

from pathlib import Path

import numpy as np
import pandas as pd

from data.persona_spec import PersonaAttribute, load_persona_spec

DATA_DIR = Path(__file__).resolve().parent
ISSP_YEARS = ("2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024")
REQUIRED_TASK_COLS = ("CASEID", "v", "question", "type", "option_codes", "option_texts", "answer_code", "answer_text")


def prompt_map_from_spec(spec: dict[str, PersonaAttribute]) -> dict[str, str]:
    """Return `{attribute_name: prompt_label}` from a loaded `persona_spec.yaml` dict."""
    return {name: attr.prompt for name, attr in spec.items()}


def add_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Attach validated one-hot response vectors `one_hot` to a long-format DataFrame in place."""
    one_hots: list[np.ndarray] = []
    for qid, codes, texts, ans in zip(
        df["qid"], df["option_codes"], df["option_texts"], df["answer_code"], strict=True
    ):
        codes_list = [int(c) for c in codes]
        if len(codes_list) != len(texts):
            raise ValueError(f"{qid}: option_codes ({len(codes_list)}) != option_texts ({len(texts)})")
        ans_int = int(ans)
        if ans_int not in codes_list:
            raise ValueError(f"{qid}: answer_code {ans_int!r} not in option_codes {codes_list!r}")
        vec = np.zeros(len(codes_list), dtype=np.float64)
        vec[codes_list.index(ans_int)] = 1.0
        one_hots.append(vec)
    df["one_hot"] = one_hots
    return df


def validate_persona_columns(df: pd.DataFrame, spec: dict[str, PersonaAttribute]) -> list[str]:
    """Verify every persona column between `CASEID` and `v` exists in `spec` and contains only canonical levels."""
    if df.columns[0] != "CASEID":
        raise ValueError(f"Expected first column to be 'CASEID', got {df.columns[0]!r}")
    v_idx = df.columns.get_loc("v")
    persona_cols = list(df.columns[1:v_idx])
    unknown = [c for c in persona_cols if c not in spec]
    if unknown:
        raise ValueError(f"Unknown persona columns not in persona_spec.yaml: {unknown}")

    for col in persona_cols:
        attr_spec = spec[col]
        allowed = set(attr_spec.levels)
        observed = set(df[col].dropna().unique())
        if attr_spec.multi_select:
            observed = {tok for val in observed for tok in val.split("; ")}
        invalid = observed - allowed
        if invalid:
            raise ValueError(f"{col}: values outside persona_spec levels: {sorted(invalid)}")
    return persona_cols


def load_parquet_df(
    path: str | Path,
    *,
    dataset: str,
    spec: dict[str, PersonaAttribute] | None = None,
    qids=None,
) -> pd.DataFrame:
    """Load and validate a standardized `*_long.parquet` file, attaching `dataset`, `qid`, and `one_hot`."""
    pq_path = Path(path)
    if not pq_path.exists():
        raise FileNotFoundError(f"Parquet not found: {pq_path}")
    active_spec = spec if spec is not None else load_persona_spec()

    df = pd.read_parquet(pq_path)
    missing = [c for c in REQUIRED_TASK_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{pq_path}: missing required columns {missing}")

    validate_persona_columns(df, active_spec)
    for col in ("option_codes", "option_texts"):
        df[col] = df[col].map(list)

    df["dataset"] = dataset
    df["qid"] = f"{dataset}:" + df["v"].astype(str)
    add_one_hot(df)

    if qids is not None:
        req = set(qids)
        present = set(df["qid"].unique())
        missing_qids = sorted(req - present)
        if missing_qids:
            raise KeyError(f"{dataset}: requested qids not found: {missing_qids[:10]}")
        df = df[df["qid"].isin(req)].reset_index(drop=True)
    return df


def load_issp_df(
    years=ISSP_YEARS,
    *,
    spec: dict[str, PersonaAttribute] | None = None,
    qids=None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Load one or more standardized ISSP US parquets into a single DataFrame."""
    active_spec = spec if spec is not None else load_persona_spec(data_dir / "persona_spec.yaml")
    year_list = [str(y) for y in ([years] if isinstance(years, (str, int)) else years)]
    if not year_list:
        raise ValueError("years must not be empty")

    frames = [
        load_parquet_df(
            data_dir / f"issp{y}" / f"issp{y}_US_long.parquet",
            dataset=f"issp{y}",
            spec=active_spec,
        )
        for y in year_list
    ]
    df = pd.concat(frames, ignore_index=True)

    if qids is not None:
        req = set(qids)
        present = set(df["qid"].unique())
        missing_qids = sorted(req - present)
        if missing_qids:
            raise KeyError(f"ISSP: requested qids not found: {missing_qids[:10]}")
        df = df[df["qid"].isin(req)].reset_index(drop=True)
    return df


def _option_descs_or_none(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    raise TypeError(f"option_descriptions must be a list, None, or NaN, got {type(value).__name__}: {value!r}")


def build_records(
    df: pd.DataFrame,
    *,
    spec: dict[str, PersonaAttribute],
    include_country: bool = False,
) -> list[dict]:
    """Convert rows of a standardized long DataFrame into training/evaluation record dicts."""
    if df.empty:
        raise ValueError("df must not be empty")
    for req_col in ("CASEID", "qid", "dataset", "v", "type", "question", "option_texts", "one_hot"):
        if req_col not in df.columns:
            raise KeyError(f"Missing required column {req_col!r} in df")
    if include_country and "country" not in df.columns:
        raise KeyError("Expected 'country' column when include_country=True")

    persona_cols = [
        c for c in spec if c in df.columns and df[c].notna().any() and (include_country or c != "country")
    ]
    if not persona_cols:
        raise ValueError("No non-empty persona columns from persona_spec.yaml found in df")

    has_descs = "option_descriptions" in df.columns
    records: list[dict] = []
    for row in df.to_dict("records"):
        personas = {c: row[c] for c in persona_cols if pd.notna(row[c])}
        if not personas:
            raise ValueError(
                f"Respondent CASEID={row['CASEID']!r} (qid={row['qid']!r}) has no non-null persona attributes"
            )
        rec = {
            "qid": row["qid"],
            "dataset": row["dataset"],
            "v": row["v"],
            "CASEID": row["CASEID"],
            "type": row["type"],
            "question": row["question"],
            "option_texts": list(row["option_texts"]),
            "option_descriptions": _option_descs_or_none(row["option_descriptions"]) if has_descs else None,
            "personas": personas,
            "n": 1,
            "human_dist": np.asarray(row["one_hot"], dtype=np.float64),
        }
        records.append(rec)
    return records


def record_chars(rec: dict) -> int:
    """Deterministic character-length estimate for a record's question stem and options."""
    total = len(rec["question"]) + sum(len(o) for o in rec["option_texts"])
    descs = rec["option_descriptions"]
    if descs is not None:
        total += sum(len(d) for d in descs if d is not None)
    return total


def sample_persona_state(
    personas: dict[str, str],
    *,
    prompt_map: dict[str, str],
    rng=None,
) -> str:
    """Format `personas` into a Kev `state` string, optionally shuffling attribute order once via `rng`."""
    if not isinstance(personas, dict) or not personas:
        raise ValueError(f"personas must be a non-empty dict, got {personas!r}")
    items = list(personas.items())
    if rng is not None and len(items) > 1:
        rng.shuffle(items)
    return "\n".join(f"{prompt_map[c]}: {v}" for c, v in items)


def build_question_spec(rec: dict) -> dict:
    """Build the Kev `questions[key]` dict for a single record (`score` or `choice`)."""
    qid = rec["qid"]
    q_type = rec["type"]
    option_texts = rec["option_texts"]
    option_descs = rec["option_descriptions"]
    human_dist = rec["human_dist"]

    if len(option_texts) != len(human_dist):
        raise ValueError(f"{qid}: option_texts ({len(option_texts)}) != human_dist ({len(human_dist)})")

    if q_type == "score":
        criteria = option_texts
        target = {str(i): float(p) for i, p in enumerate(human_dist)}
    elif q_type == "choice":
        if option_descs is not None:
            criteria = dict(zip(option_texts, option_descs, strict=True))
        else:
            criteria = {text: None for text in option_texts}
        if len(criteria) != len(option_texts):
            raise ValueError(f"{qid}: duplicate option_texts in choice question: {option_texts!r}")
        target = {text: float(p) for text, p in zip(option_texts, human_dist, strict=True)}
    else:
        raise ValueError(f"{qid}: unexpected question type {q_type!r}")

    return {
        "type": q_type,
        "instructions": rec["question"],
        "criteria": criteria,
        "target": target,
    }


def make_view(rec: dict, *, prompt_map: dict[str, str], rng=None) -> dict:
    """Build a single-question Kev view from a record dict."""
    state = sample_persona_state(rec["personas"], prompt_map=prompt_map, rng=rng)
    return {
        "state": state,
        "questions": {rec["qid"]: build_question_spec(rec)},
    }


def make_state_chunk_view(state: str, chunk_records: list[dict]) -> dict:
    """Build a multi-question Kev view sharing a pre-sampled `state` string."""
    if not isinstance(state, str) or not state:
        raise ValueError(f"state must be a non-empty string, got {state!r}")
    if not chunk_records:
        raise ValueError("chunk_records must not be empty")

    first_personas = chunk_records[0]["personas"]
    questions = {}
    for idx, rec in enumerate(chunk_records):
        if rec["personas"] != first_personas:
            raise ValueError("All records in chunk_records must share the exact same personas dict")
        questions[f"{idx}:{rec['qid']}"] = build_question_spec(rec)
    return {"state": state, "questions": questions}
