"""Kev collection runner for the COVID conjoint on the audited covid_US_long.parquet prompts, with progressive JSONL persistence.

The persona design (state columns), subsample and checkpoints are set by the calling notebook."""

import gc
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "kev") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "kev"))

from data.covid import load_covid_df  # noqa: E402
from data.persona import PERSONA_NAMES  # noqa: E402
from kev.model import rows_of  # noqa: E402
from models.kev_pointer import Checkpoint, KevEncoder, KevPointerModel, LoadOptions  # noqa: E402

# Question text, option labels and option descriptions come from the audited covid_US_long.parquet (data.covid);
# which persona columns enter the state is passed in by the notebook. Saved probabilities follow option order A, B, C.
CHOICE_LETTERS = ("A", "B", "C")


def enable_qwen3_5_cpu_kernels() -> None:
    """Unwrap Hub/FLA Triton kernels on Qwen3.5 so CPU tensors use the pure-PyTorch reference implementation."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    for fn_name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        fn = getattr(modeling_qwen3_5, fn_name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(modeling_qwen3_5, fn_name, fn.__wrapped__)


def load_audited_choice_df(df_a: pd.DataFrame) -> pd.DataFrame:
    """Load the audited `vax_choice` rows from covid_US_long.parquet and attach `pair_id` from the paired .dta rows.

    The parquet keeps the (respondent, pair_id) order of `extract_choice_pairs_df` but not `pair_id` itself,
    so `df_a` (Vaccine A rows from `wang_mnl.pair_choice_sets`) supplies it by position.
    """
    choice_df = load_covid_df()
    choice_df = choice_df[choice_df["v"] == "vax_choice"].reset_index(drop=True)
    if len(choice_df) != len(df_a) or not np.array_equal(
        choice_df["respondent"].to_numpy(), df_a["respondent"].to_numpy()
    ):
        raise ValueError("Parquet vax_choice rows are not aligned with the .dta (respondent, pair_id) order")
    choice_df["pair_id"] = df_a["pair_id"].to_numpy()
    return choice_df


def build_respondent_states(choice_df: pd.DataFrame, *, persona_cols: Sequence[str]) -> dict[int, str]:
    """Map each respondent in `choice_df` to a 'Label: value' state over `persona_cols` (PERSONA_NAMES columns, in order).

    Missing values are left out of a respondent's state; a respondent with no values at all raises.
    """
    persona_cols = list(persona_cols)
    unknown = [c for c in persona_cols if c not in PERSONA_NAMES]
    if unknown:
        raise ValueError(f"Unknown persona columns {unknown}; expected columns of PERSONA_NAMES")
    if len(set(persona_cols)) != len(persona_cols):
        raise ValueError(f"Duplicate persona columns in {persona_cols}")
    resp_state_by_id: dict[int, str] = {}
    for row in choice_df.drop_duplicates("respondent").to_dict("records"):
        lines = [f"{PERSONA_NAMES[c]}: {row[c]}" for c in persona_cols if pd.notna(row[c])]
        if persona_cols and not lines:
            raise ValueError(f"Empty persona state for respondent {row['respondent']!r}")
        resp_state_by_id[int(row["respondent"])] = "\n".join(lines)
    return resp_state_by_id


def build_kev_prompt_items(
    choice_df: pd.DataFrame,
    *,
    design_name: str,
    persona_cols: Sequence[str],
) -> list[dict[str, object]]:
    """Build per-pair Kev view items from audited parquet rows for one persona design.

    `design_name` labels the design in every item and saved record (e.g. "random", "persona");
    `persona_cols` are rendered in the state in this order, and an empty sequence gives an empty state.
    """
    persona_cols = list(persona_cols)
    resp_state_by_id = build_respondent_states(choice_df, persona_cols=persona_cols)

    items: list[dict[str, object]] = []
    for row in choice_df.to_dict("records"):
        resp_id = int(row["respondent"])
        pair_id = int(row["pair_id"])
        state_str = resp_state_by_id[resp_id]

        option_texts = list(row["option_texts"])
        if len(option_texts) != len(CHOICE_LETTERS):
            raise ValueError(f"{row['qid']}: expected 3 options (A, B, neither), got {option_texts!r}")
        q_spec = {
            "type": "choice",
            "instructions": row["question"],
            "criteria": dict(zip(option_texts, row["option_descriptions"], strict=True)),
            "target": {t: float(p) for t, p in zip(option_texts, row["one_hot"], strict=True)},
        }
        items.append({
            "respondent": resp_id,
            "pair_id": pair_id,
            "persona_mode": design_name,
            "persona_cols": persona_cols,
            "state": state_str,
            "question_spec": q_spec,
            "view": {"state": state_str, "questions": {f"pair_{pair_id}": q_spec}},
        })
    return items


def verify_matches_gemini_subsample(
    prompt_items: list[dict[str, object]],
    *,
    gemini_jsonl_path: str | Path,
) -> None:
    """Fail fast if the (respondent, pair_id) pairs in prompt_items do not match the Gemini subsample JSONL."""
    g_path = Path(gemini_jsonl_path)
    if not g_path.exists():
        raise FileNotFoundError(f"Gemini reference JSONL not found at {g_path.resolve()}")
    gemini_keys: set[tuple[int, int]] = set()
    with g_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gemini_keys.add((int(rec["respondent"]), int(rec["pair_id"])))

    kev_keys = [(int(it["respondent"]), int(it["pair_id"])) for it in prompt_items]
    if len(kev_keys) != len(gemini_keys) or set(kev_keys) != gemini_keys:
        diff_kev = sorted(set(kev_keys) - gemini_keys)[:5]
        diff_gem = sorted(gemini_keys - set(kev_keys))[:5]
        raise ValueError(
            f"Subsample mismatch vs. {g_path.name}: kev has {len(kev_keys)} pairs, gemini has {len(gemini_keys)} pairs. "
            f"Only in kev: {diff_kev}, only in gemini: {diff_gem}"
        )


def preview_kev_item(item: dict[str, object], *, checkpoint: str | Path) -> str:
    """Decode and format a single Kev prompt item using KevEncoder.preview (without loading backbone weights)."""
    enc = KevEncoder.for_base(str(checkpoint))
    header, text = enc.preview(item["view"])
    return f"{header}\n{text}"


def free_kev_memory() -> None:
    """Clear Python exception frames, run garbage collection, and empty CUDA cache if CUDA is initialized."""
    if hasattr(sys, "last_traceback") and sys.last_traceback is not None:
        traceback.clear_frames(sys.last_traceback)
        sys.last_traceback = None
    for attr in ("last_value", "last_type", "last_exc"):
        if hasattr(sys, attr):
            setattr(sys, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def load_kev_model(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    temperature: float | None = None,
) -> KevPointerModel:
    """Free memory/GPU cache and load a Kev checkpoint onto `device` (default 'cpu') using its checkpoint default temperature when temperature=None."""
    free_kev_memory()
    if str(device) == "cpu":
        enable_qwen3_5_cpu_kernels()
    ck = Checkpoint(str(checkpoint))
    tok, net = ck.load(device, LoadOptions(dtype=dtype, temperature=temperature))
    model = KevPointerModel(net, tok, ck.meta, "loaded")
    model.eval()
    return model


def load_kev_jsonl(
    output_path: str | Path,
    *,
    expected_model: str | None = None,
    expected_persona_mode: str | None = None,
    expected_temperature: float | None = None,
    expected_persona_cols: Sequence[str] | None = None,
) -> dict[tuple[int, int], dict[str, object]]:
    """Load previously collected (respondent, pair_id) Kev records from JSONL and verify metadata.

    `expected_persona_cols` requires every record to carry the same `persona_cols` (records written before
    persona_cols were saved have no such field and fail this check).
    """
    path = Path(output_path)
    collected: dict[tuple[int, int], dict[str, object]] = {}
    if not path.exists():
        return collected
    required_keys = {
        "respondent",
        "pair_id",
        "model",
        "persona_mode",
        "device",
        "temperature",
        "meta_temperature",
        "logits",
        "probs",
        "choice",
        "state_tokens",
        "total_tokens",
    }
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = required_keys - rec.keys()
            if missing:
                raise ValueError(f"{path}:{line_num} missing required keys: {sorted(missing)}")
            if expected_model is not None and rec["model"] != expected_model:
                raise ValueError(f"{path}:{line_num} model mismatch: {rec['model']!r} != {expected_model!r}")
            if expected_persona_mode is not None and rec["persona_mode"] != expected_persona_mode:
                raise ValueError(
                    f"{path}:{line_num} persona_mode mismatch: {rec['persona_mode']!r} != {expected_persona_mode!r}"
                )
            if expected_temperature is not None and not np.isclose(float(rec["temperature"]), float(expected_temperature), atol=1e-4):
                raise ValueError(
                    f"{path}:{line_num} temperature mismatch: {rec['temperature']} != {expected_temperature}"
                )
            if expected_persona_cols is not None and rec["persona_cols"] != list(expected_persona_cols):
                raise ValueError(
                    f"{path}:{line_num} persona_cols mismatch: {rec['persona_cols']!r} != {list(expected_persona_cols)!r}"
                )
            collected[(int(rec["respondent"]), int(rec["pair_id"]))] = rec
    return collected


@torch.no_grad()
def _score_packed_enc_with_cache(
    model: KevPointerModel,
    enc: dict[str, object],
    prefix_cache: dict[str, tuple[int, object, torch.Tensor | None]],
    temperature: float,
) -> list[tuple[list[float], list[float]]]:
    """Return per-question (raw_logits, probs_at_temperature) for a packed respondent view, reusing state prefixes."""
    state_ids_key = json.dumps(enc["ids"][: enc["seg"].count(0)])
    net = model.net
    if state_ids_key not in prefix_cache:
        # Keep at most 4 cached state prefixes in RAM
        if len(prefix_cache) >= 4:
            prefix_cache.clear()
        prefix_cache[state_ids_key] = net.prefix(enc)

    prefix_len, cache, _ = prefix_cache[state_ids_key]
    _, _, rows = rows_of(enc)
    hs = net._rows_hidden([(r["ids"], r["pos"]) for r in rows], cache=cache, prefix_len=prefix_len)

    # Compute raw pointer logits (with head.temperature temporarily 1.0) and softmax probabilities at `temperature`
    orig_temp = net.head.temperature
    net.head.temperature = 1.0
    try:
        results: list[tuple[list[float], list[float]]] = []
        for h, r in zip(hs, rows, strict=True):
            opt_indices = torch.tensor(r["opts"], device=net.device)
            raw_z = net.head(h[r["decide"]], h[opt_indices]).float().cpu()
            probs = F.softmax(raw_z / float(temperature), dim=-1)
            results.append(([float(x) for x in raw_z.tolist()], [float(x) for x in probs.tolist()]))
        return results
    finally:
        net.head.temperature = orig_temp


def collect_kev_prompts(
    model: KevPointerModel,
    prompt_items: list[dict[str, object]],
    output_path: str | Path,
    *,
    model_name: str,
    n_personas: int | None = None,
    temperature: float | None = None,
) -> list[dict[str, object]]:
    """Incrementally run Kev inference (grouped by respondent) and progressively flush each choice set to JSONL.

    - Uses the checkpoint's default `model.meta.temperature` when `temperature=None` (e.g. 2.19 for Base Kev 9B, 1.0 for FT).
    - Groups the 5 choice sets of each respondent into one packed view so the respondent's state is encoded once.
    - Reuses the state KV/recurrent prefix whenever consecutive respondents share the same state (e.g. Random Persona).
    - Immediately flushes each respondent's 5 records to `output_path` so stopping mid-run loses 0 completed respondents.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not prompt_items:
        raise ValueError("prompt_items must not be empty")
    eff_temp = float(model.meta.temperature) if temperature is None else float(temperature)

    persona_mode = str(prompt_items[0]["persona_mode"])
    persona_cols = list(prompt_items[0]["persona_cols"])
    if any(it["persona_mode"] != persona_mode or list(it["persona_cols"]) != persona_cols for it in prompt_items):
        raise ValueError("prompt_items mix persona designs; collect one design per output file")
    if n_personas is not None:
        seen_resp: list[int] = []
        seen_set: set[int] = set()
        for item in prompt_items:
            r_id = int(item["respondent"])
            if r_id not in seen_set:
                seen_set.add(r_id)
                seen_resp.append(r_id)
        target_resp = set(seen_resp[:n_personas])
        active_items = [item for item in prompt_items if int(item["respondent"]) in target_resp]
    else:
        active_items = list(prompt_items)

    collected = load_kev_jsonl(
        output_path,
        expected_model=model_name,
        expected_persona_mode=persona_mode,
        expected_temperature=eff_temp,
        expected_persona_cols=persona_cols,
    )
    pending_items = [
        item for item in active_items if (int(item["respondent"]), int(item["pair_id"])) not in collected
    ]
    already_done = len(active_items) - len(pending_items)
    print(
        f"[{output_path.name}] model={model_name}, persona_mode={persona_mode}, temp={eff_temp:.2f}, device={model.net.device} | "
        f"Target: {len(active_items)} prompts ({already_done} already collected, {len(pending_items)} remaining)"
    )

    if pending_items:
        # Group pending items by respondent so all pending choice sets for a respondent run in one packed pass
        by_resp: dict[int, list[dict[str, object]]] = {}
        for item in pending_items:
            by_resp.setdefault(int(item["respondent"]), []).append(item)

        prefix_cache: dict[str, tuple[int, object, torch.Tensor | None]] = {}
        pbar = tqdm(total=len(pending_items), desc=output_path.stem, unit="pair")
        meta_temp = float(model.meta.temperature)
        device_str = str(model.net.device)

        with output_path.open("a", encoding="utf-8") as f_out:
            try:
                for resp_id, resp_items in by_resp.items():
                    state_str = str(resp_items[0]["state"])
                    packed_questions = {
                        f"pair_{int(it['pair_id'])}": it["question_spec"] for it in resp_items
                    }
                    packed_enc, _ = model.prepare({"state": state_str, "questions": packed_questions})
                    scored = _score_packed_enc_with_cache(model, packed_enc, prefix_cache, temperature=eff_temp)

                    for it, (raw_logits, probs) in zip(resp_items, scored, strict=True):
                        pair_id = int(it["pair_id"])
                        single_enc, _ = model.prepare(it["view"])
                        top_choice = CHOICE_LETTERS[int(np.argmax(probs))]
                        rec = {
                            "respondent": resp_id,
                            "pair_id": pair_id,
                            "model": model_name,
                            "persona_mode": persona_mode,
                            "persona_cols": persona_cols,
                            "device": device_str,
                            "temperature": eff_temp,
                            "meta_temperature": meta_temp,
                            "logits": raw_logits,
                            "probs": probs,
                            "choice": top_choice,
                            "state_tokens": int(single_enc["state_tokens"]),
                            "total_tokens": len(single_enc["ids"]),
                        }
                        collected[(resp_id, pair_id)] = rec
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                    pbar.update(len(resp_items))
            finally:
                prefix_cache.clear()
                pbar.close()

    return [collected[(int(it["respondent"]), int(it["pair_id"]))] for it in active_items]


def extract_kev_prob_targets(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract soft probability arrays (p_A, p_B, p_C) from collected Kev JSONL records."""
    probs = np.array([rec["probs"] for rec in records], dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) probability array, got {probs.shape}")
    return probs[:, 0], probs[:, 1], probs[:, 2]


def extract_kev_hard_targets(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract one-hot argmax choice arrays (y_A, y_B, y_C) from collected Kev JSONL records."""
    choices = np.array([str(rec["choice"]).strip() for rec in records])
    return (
        (choices == "A").astype(np.float64),
        (choices == "B").astype(np.float64),
        (choices == "C").astype(np.float64),
    )
