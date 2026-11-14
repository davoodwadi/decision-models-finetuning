"""Kev CPU evaluation and progressive JSONL dataset collection utilities for the Grummon et al. (2025) FOP nutrition label RCT."""

import gc
import json
import sys
import time
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

from data.grummon2025 import (  # noqa: E402
    GRUMMON2025_PERSONA_NAMES,
    PRIMARY_CHOICE_STEMS,
)
from kev.model import rows_of  # noqa: E402
from models.kev_pointer import Checkpoint, KevEncoder, KevPointerModel, LoadOptions  # noqa: E402


def enable_qwen3_5_cpu_kernels() -> None:
    """Replace @use_kernel_func_from_hub_with_fallback Triton wrappers on Qwen3_5GatedDeltaNet with pure-PyTorch fallbacks for CPU."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as q35_mod

    for fn_name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        fn = getattr(q35_mod, fn_name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(q35_mod, fn_name, fn.__wrapped__)


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


def build_kev_prompt_items(
    sub_df: pd.DataFrame,
    *,
    design_name: str,
    persona_cols: Sequence[str],
    skip_missing_state: bool,
) -> list[dict[str, object]]:
    """Build ordered task-level Kev prompt items for one persona design.

    - `design_name` labels the design in every item and saved record (e.g. "random", "persona", "single:SNAP").
    - `persona_cols` are the GRUMMON2025_PERSONA_NAMES columns rendered as "Label: value" lines in the state, in this
      order; an empty sequence gives an empty state (no persona). Missing values are left out of a participant's state.
    - `skip_missing_state=True` drops participants whose state would be empty (e.g. single-attribute designs);
      `False` raises instead.
    """
    persona_cols = list(persona_cols)
    unknown = [c for c in persona_cols if c not in GRUMMON2025_PERSONA_NAMES]
    if unknown:
        raise ValueError(f"Unknown persona columns {unknown}; expected columns of GRUMMON2025_PERSONA_NAMES")
    if len(set(persona_cols)) != len(persona_cols):
        raise ValueError(f"Duplicate persona columns in {persona_cols}")

    items: list[dict[str, object]] = []
    for row in sub_df.to_dict("records"):
        lines = [f"{GRUMMON2025_PERSONA_NAMES[c]}: {row[c]}" for c in persona_cols if pd.notna(row[c])]
        if persona_cols and not lines:
            if skip_missing_state:
                continue
            raise ValueError(f"Empty persona state for CASEID={row['CASEID']!r} under design {design_name!r}")
        state = "\n".join(lines)

        q_type = str(row["type"])
        opt_texts = [str(t) for t in row["option_texts"]]
        raw_descs = row["option_descriptions"]
        if q_type == "choice":
            if raw_descs is not None:
                opt_descs = [None if d is None or pd.isna(d) else str(d) for d in raw_descs]
                if len(opt_texts) != len(opt_descs):
                    raise ValueError(f"Mismatched option_descriptions length for {row['qid']}")
                criteria: object = dict(zip(opt_texts, opt_descs, strict=True))
            else:
                criteria = {t: None for t in opt_texts}
            target = {str(t): 1.0 / len(opt_texts) for t in opt_texts}
        elif q_type == "score":
            criteria = list(opt_texts)
            target = {str(idx): 1.0 / len(opt_texts) for idx in range(len(opt_texts))}
        else:
            raise ValueError(f"Unsupported question type {q_type!r} for {row['qid']}")

        qid = str(row["qid"])
        view = {
            "state": state,
            "questions": {
                qid: {
                    "type": q_type,
                    "instructions": str(row["question"]),
                    "criteria": criteria,
                    "target": target,
                }
            },
        }

        items.append({
            "CASEID": str(row["CASEID"]),
            "v": str(row["v"]),
            "qid": qid,
            "arm": str(row["arm"]),
            "type": q_type,
            "persona_mode": design_name,
            "persona_cols": persona_cols,
            "state": state,
            "instructions": str(row["question"]),
            "criteria": criteria,
            "option_texts": opt_texts,
            "view": view,
        })
    return items


def preview_kev_item(item: dict[str, object], checkpoint: str | Path = "jaredpalmer/kev-9b") -> str:
    """Decode and format a single Grummon 2025 Kev prompt item using KevEncoder.preview (without loading backbone weights)."""
    enc = KevEncoder.for_base(str(checkpoint))
    header, text = enc.preview(item["view"])
    return f"{header}\n{text}"


def load_kev_jsonl(
    jsonl_path: str | Path,
    *,
    expected_model: str | None = None,
    expected_persona_mode: str | None = None,
    expected_temperature: float | None = None,
    expected_persona_cols: Sequence[str] | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    """Load saved Grummon 2025 Kev JSONL records keyed by (CASEID, v) and validate metadata.

    `expected_persona_cols` requires every record to carry the same `persona_cols` (records written before
    persona_cols were saved have no such field and fail this check).
    """
    path = Path(jsonl_path)
    if not path.exists():
        return {}

    records: dict[tuple[str, str], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (str(rec["CASEID"]), str(rec["v"]))
            if key in records:
                raise ValueError(f"Duplicate (CASEID, v) key {key} at line {line_num} in {path}")
            if expected_model is not None and rec["model"] != expected_model:
                raise ValueError(f"Line {line_num} in {path}: model={rec['model']!r} != {expected_model!r}")
            if expected_persona_mode is not None and rec["persona_mode"] != expected_persona_mode:
                raise ValueError(f"Line {line_num} in {path}: persona_mode={rec['persona_mode']!r} != {expected_persona_mode!r}")
            if expected_temperature is not None and abs(float(rec["temperature"]) - float(expected_temperature)) > 1e-6:
                raise ValueError(f"Line {line_num} in {path}: temperature={rec['temperature']!r} != {expected_temperature!r}")
            if expected_persona_cols is not None and rec["persona_cols"] != list(expected_persona_cols):
                raise ValueError(f"Line {line_num} in {path}: persona_cols={rec['persona_cols']!r} != {list(expected_persona_cols)!r}")
            records[key] = rec
    return records


@torch.no_grad()
def _score_packed_enc_with_cache(
    model: KevPointerModel,
    enc: dict[str, object],
    prefix_cache: dict[str, tuple[int, object, torch.Tensor | None]],
    temperature: float,
) -> list[tuple[list[float], list[float]]]:
    """Return per-question (raw_logits, probs_at_temperature) for a packed participant view, reusing state prefixes."""
    state_ids_key = json.dumps(enc["ids"][: enc["seg"].count(0)])
    net = model.net
    if state_ids_key not in prefix_cache:
        if len(prefix_cache) >= 4:
            prefix_cache.clear()
        prefix_cache[state_ids_key] = net.prefix(enc)

    prefix_len, cache, _ = prefix_cache[state_ids_key]
    _, _, rows = rows_of(enc)
    hs = net._rows_hidden([(r["ids"], r["pos"]) for r in rows], cache=cache, prefix_len=prefix_len)

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
    """Progressively evaluate Kev on Grummon 2025 prompt_items using (state, arm) shared-prefix deduplication."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not prompt_items:
        raise ValueError("prompt_items must not be empty")

    persona_mode = str(prompt_items[0]["persona_mode"])
    persona_cols = list(prompt_items[0]["persona_cols"])
    if any(it["persona_mode"] != persona_mode or list(it["persona_cols"]) != persona_cols for it in prompt_items):
        raise ValueError("prompt_items mix persona designs; collect one design per output file")
    effective_temp = float(model.meta.temperature) if temperature is None else float(temperature)

    existing = load_kev_jsonl(
        path,
        expected_model=model_name,
        expected_persona_mode=persona_mode,
        expected_temperature=effective_temp,
        expected_persona_cols=persona_cols,
    )
    if n_personas is not None:
        seen_cids: list[str] = []
        seen_set: set[str] = set()
        for it in prompt_items:
            cid = str(it["CASEID"])
            if cid not in seen_set:
                seen_set.add(cid)
                seen_cids.append(cid)
        target_cids = set(seen_cids[:n_personas])
        target_items = [it for it in prompt_items if str(it["CASEID"]) in target_cids]
    else:
        target_items = list(prompt_items)

    missing_items = [
        item for item in target_items
        if (str(item["CASEID"]), str(item["v"])) not in existing
    ]

    if not missing_items:
        print(
            f"[{path.name}] Complete: all {len(target_items):,}/{len(prompt_items):,} requested items "
            f"already saved (total in file: {len(existing):,})."
        )
        return [existing[(str(it["CASEID"]), str(it["v"]))] for it in target_items]

    # Deduplicate by (state, arm) so identical persona+arm combinations share a single 6-question forward pass
    unique_groups: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    group_items: dict[tuple[str, str], list[dict[str, object]]] = {}
    for item in missing_items:
        g_key = (str(item["state"]), str(item["arm"]))
        v_code = str(item["v"])
        if g_key not in unique_groups:
            unique_groups[g_key] = {}
            group_items[g_key] = []
        opt_texts = list(item["option_texts"])
        q_type = str(item["type"])
        if q_type == "score":
            criteria_val: object = list(item["criteria"])
            target_val = {str(idx): 1.0 / len(opt_texts) for idx in range(len(opt_texts))}
        else:
            criteria_val = dict(item["criteria"])
            target_val = {str(t): 1.0 / len(opt_texts) for t in opt_texts}
        unique_groups[g_key][v_code] = {
            "type": q_type,
            "instructions": str(item["instructions"]),
            "criteria": criteria_val,
            "target": target_val,
        }
        group_items[g_key].append(item)

    print(
        f"[{path.name}] Collecting {len(missing_items):,} missing items across {len(unique_groups):,} unique "
        f"(state, arm) groups on device={model.net.device} (model={model_name!r}, T={effective_temp:.4f})..."
    )

    prefix_cache: dict[str, tuple[int, object, torch.Tensor | None]] = {}
    pbar = tqdm(unique_groups.items(), total=len(unique_groups), desc=f"{path.stem} ({model.net.device})")
    with path.open("a", encoding="utf-8") as f:
        for (state_str, arm_str), q_dict in pbar:
            t0 = time.perf_counter()
            v_codes = list(q_dict.keys())
            grouped_view = {
                "state": state_str,
                "questions": {v_code: q_dict[v_code] for v_code in v_codes},
            }
            enc, _ = model.prepare(grouped_view)
            res_list = _score_packed_enc_with_cache(model, enc, prefix_cache, effective_temp)
            dt = (time.perf_counter() - t0) / len(group_items[(state_str, arm_str)])

            v_to_res = dict(zip(v_codes, res_list, strict=True))
            for item in group_items[(state_str, arm_str)]:
                v_code = str(item["v"])
                q_type = str(item["type"])
                logits, probs = v_to_res[v_code]
                pred_idx = int(np.argmax(probs))
                rec: dict[str, object] = {
                    "CASEID": str(item["CASEID"]),
                    "v": v_code,
                    "qid": str(item["qid"]),
                    "arm": arm_str,
                    "type": q_type,
                    "model": model_name,
                    "persona_mode": persona_mode,
                    "persona_cols": persona_cols,
                    "temperature": effective_temp,
                    "probs": [float(x) for x in probs],
                    "logits": [float(x) for x in logits],
                    "pred_option_idx": pred_idx,
                    "elapsed_sec": round(dt, 4),
                }
                if v_code in PRIMARY_CHOICE_STEMS:
                    rec["prob_correct"] = float(probs[0])
                    rec["prob_wrong"] = float(probs[1])
                    rec["prob_same"] = float(probs[2])
                    rec["logit_correct"] = float(logits[0])
                    rec["logit_wrong"] = float(logits[1])
                    rec["logit_same"] = float(logits[2])
                f.write(json.dumps(rec) + "\n")
                existing[(str(item["CASEID"]), v_code)] = rec
            f.flush()

    return [existing[(str(it["CASEID"]), str(it["v"]))] for it in target_items]


def load_single_attr_jsonl(
    jsonl_path: str | Path,
    *,
    expected_model: str | None = None,
    expected_temperature: float | None = None,
) -> dict[tuple[str, str, str], dict[str, object]]:
    """Load saved single-attribute Grummon 2025 Kev JSONL records keyed by (attribute, CASEID, v)."""
    path = Path(jsonl_path)
    if not path.exists():
        return {}

    records: dict[tuple[str, str, str], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (str(rec["attribute"]), str(rec["CASEID"]), str(rec["v"]))
            if key in records:
                raise ValueError(f"Duplicate (attribute, CASEID, v) key {key} at line {line_num} in {path}")
            if expected_model is not None and rec["model"] != expected_model:
                raise ValueError(f"Line {line_num} in {path}: model={rec['model']!r} != {expected_model!r}")
            if expected_temperature is not None and abs(float(rec["temperature"]) - float(expected_temperature)) > 1e-6:
                raise ValueError(f"Line {line_num} in {path}: temperature={rec['temperature']!r} != {expected_temperature!r}")
            records[key] = rec
    return records


def collect_kev_single_attr_prompts(
    model: KevPointerModel,
    sub_df: pd.DataFrame,
    output_path: str | Path,
    *,
    model_name: str,
    attr_cols: Sequence[str],
    temperature: float | None = None,
) -> list[dict[str, object]]:
    """Evaluate Kev conditioned on each attribute in `attr_cols` alone (participants missing that attribute are skipped)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    effective_temp = float(model.meta.temperature) if temperature is None else float(temperature)

    existing = load_single_attr_jsonl(
        path,
        expected_model=model_name,
        expected_temperature=effective_temp,
    )

    attr_cols = list(attr_cols)
    extra = sorted({key[0] for key in existing} - set(attr_cols))
    if extra:
        raise ValueError(f"{path} already holds attributes outside attr_cols: {extra}")

    all_items: list[tuple[str, dict[str, object]]] = []
    for col in attr_cols:
        col_items = build_kev_prompt_items(
            sub_df, design_name=f"single:{col}", persona_cols=[col], skip_missing_state=True
        )
        for it in col_items:
            all_items.append((col, it))

    missing_items = [
        (col, it) for col, it in all_items
        if (col, str(it["CASEID"]), str(it["v"])) not in existing
    ]

    if not missing_items:
        print(
            f"[{path.name}] Complete: all {len(all_items):,} single-attribute items already saved."
        )
        return [existing[(col, str(it["CASEID"]), str(it["v"]))] for col, it in all_items]

    unique_groups: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    group_items: dict[tuple[str, str], list[tuple[str, dict[str, object]]]] = {}
    for col, item in missing_items:
        g_key = (str(item["state"]), str(item["arm"]))
        v_code = str(item["v"])
        if g_key not in unique_groups:
            unique_groups[g_key] = {}
            group_items[g_key] = []
        opt_texts = list(item["option_texts"])
        unique_groups[g_key][v_code] = {
            "type": "choice",
            "instructions": str(item["instructions"]),
            "criteria": dict(item["criteria"]),
            "target": {str(t): 1.0 / len(opt_texts) for t in opt_texts},
        }
        group_items[g_key].append((col, item))

    print(
        f"[{path.name}] Collecting {len(missing_items):,} single-attribute items across {len(unique_groups):,} unique "
        f"(state, arm) groups on device={model.net.device} (model={model_name!r}, T={effective_temp:.4f})..."
    )

    prefix_cache: dict[str, tuple[int, object, torch.Tensor | None]] = {}
    pbar = tqdm(unique_groups.items(), total=len(unique_groups), desc=f"{path.stem} ({model.net.device})")
    with path.open("a", encoding="utf-8") as f:
        for (state_str, arm_str), q_dict in pbar:
            t0 = time.perf_counter()
            v_codes = list(q_dict.keys())
            grouped_view = {
                "state": state_str,
                "questions": {v_code: q_dict[v_code] for v_code in v_codes},
            }
            enc, _ = model.prepare(grouped_view)
            res_list = _score_packed_enc_with_cache(model, enc, prefix_cache, effective_temp)
            dt = (time.perf_counter() - t0) / len(group_items[(state_str, arm_str)])

            v_to_res = dict(zip(v_codes, res_list, strict=True))
            for col, item in group_items[(state_str, arm_str)]:
                v_code = str(item["v"])
                logits, probs = v_to_res[v_code]
                p_correct, p_wrong, p_same = float(probs[0]), float(probs[1]), float(probs[2])
                pred_idx = int(np.argmax(probs))
                rec = {
                    "attribute": col,
                    "attribute_name": GRUMMON2025_PERSONA_NAMES[col],
                    "CASEID": str(item["CASEID"]),
                    "v": v_code,
                    "qid": str(item["qid"]),
                    "arm": arm_str,
                    "model": model_name,
                    "persona_mode": f"single:{col}",
                    "temperature": effective_temp,
                    "prob_correct": p_correct,
                    "prob_wrong": p_wrong,
                    "prob_same": p_same,
                    "logit_correct": float(logits[0]),
                    "logit_wrong": float(logits[1]),
                    "logit_same": float(logits[2]),
                    "pred_option_idx": pred_idx,
                    "elapsed_sec": round(dt, 4),
                }
                f.write(json.dumps(rec) + "\n")
                existing[(col, str(item["CASEID"]), v_code)] = rec
            f.flush()

    return [existing[(col, str(it["CASEID"]), str(it["v"]))] for col, it in all_items]
