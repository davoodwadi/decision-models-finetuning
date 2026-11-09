"""Shared dataset-agnostic abstractions for building persona records, Kev prompt views, state-grouped evaluation bundles, and model evaluation."""

from collections import defaultdict
import contextlib
import gc
import hashlib
import itertools
import json
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data.persona import PERSONA_MAPS, PERSONA_NAMES

CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = CODE_DIR / ".cache"

if str(CODE_DIR / "kev") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "kev"))
from kev.cuda_graphs import CudaGraphs
from kev.fused_qwen35 import (
    _fix_nb,
    attention_forward,
    decoder_forward,
    deltanet_forward,
    mlp_forward,
)

EXTRA_PERSONA_NAMES = {
    "HISPANIC": "Hispanic or Latino",
    "SNAP": "SNAP / Food Stamps recipient",
    "NFPUSE": "Nutrition Facts label usage",
    "ONLINEAMT": "Share of groceries bought online",
    "HEALTHLIT": "Nutrition literacy score",
}
ALL_PERSONA_NAMES = {**PERSONA_NAMES, **EXTRA_PERSONA_NAMES}

SINGLE_PERSONA_ATTRS = [c for c in PERSONA_NAMES.keys() if c != "c_alphan"]
PAIR_PERSONA_ATTRS = [list(c) for c in itertools.combinations(SINGLE_PERSONA_ATTRS, 2)]


def add_one_hot(df):
    """Attach one-hot human response vectors `one_hot` to a long-format survey DataFrame."""
    one_hots = []
    for qid, codes, texts, ans in zip(
        df["qid"], df["option_codes"], df["option_texts"], df["answer_code"], strict=True
    ):
        codes_list = [int(c) for c in codes]
        if len(codes_list) != len(texts):
            raise ValueError(f"{qid}: option_codes ({len(codes_list)}) != option_texts ({len(texts)})")
        vec = np.zeros(len(codes_list), dtype=np.float64)
        vec[codes_list.index(int(ans))] = 1.0
        one_hots.append(vec)
    df["one_hot"] = one_hots


def make_record(sub_df, cols, persona_names=ALL_PERSONA_NAMES):
    """Build a single persona-question record dict from a respondent row or subgroup DataFrame."""
    if isinstance(sub_df, dict):
        row = sub_df
        human_dist = row["one_hot"]
        n = 1
    else:
        for c in cols:
            if sub_df[c].nunique(dropna=False) != 1:
                raise ValueError(f"sub_df must have a single value for {c!r}, got {sub_df[c].unique().tolist()}")
        row = sub_df.iloc[0]
        human_dist = np.stack(sub_df["one_hot"].values).mean(axis=0)
        n = len(sub_df)

    personas = {c: row[c] for c in cols}
    return {
        "qid": row["qid"],
        "year": row["year"],
        "v": row["v"],
        "type": row["type"],
        "question": row["question"],
        "option_texts": row["option_texts"],
        "personas": personas,
        "attribute": "+".join(personas.keys()) if personas else "none",
        "persona": ", ".join(f"{persona_names[c]}={v}" for c, v in personas.items()) if personas else "All",
        "n": n,
        "human_dist": human_dist,
    }


def load_split_qids(split_spec):
    """Load a split specification into `{split_name: [qid1, qid2, ...]}`.

    `split_spec` can be:
    - a `str` or `Path` pointing to a JSON file (e.g. `splits/split.json`),
    - a nested dict `{prefix: {split_name: [v1, v2, ...]}}` (produces `"prefix:v"` qids), or
    - a flat dict `{split_name: [qid1, qid2, ...]}`.
    """
    if isinstance(split_spec, (str, Path)):
        split_spec = json.loads(Path(split_spec).read_text())
    if not isinstance(split_spec, dict) or not split_spec:
        raise ValueError("split_spec must be a non-empty dict or path to a JSON file")

    first_val = next(iter(split_spec.values()))
    if isinstance(first_val, dict):
        split_names = tuple(first_val.keys())
        out = {s: [] for s in split_names}
        for prefix, prefix_splits in split_spec.items():
            for s in split_names:
                out[s].extend(f"{prefix}:{v}" for v in prefix_splits[s])
        return out

    return {s: list(qids) for s, qids in split_spec.items()}


def filter_by_qids(df, qids=None):
    """Filter `df` to the requested `qids` (or return `df` unchanged when `qids is None`)."""
    if qids is None:
        return df
    qid_set = set(qids)
    if not qid_set:
        raise ValueError("qids must not be empty when provided")
    missing = qid_set - set(df["qid"].unique())
    if missing:
        raise KeyError(f"{len(missing)} requested qids not found in DataFrame (e.g. {sorted(missing)[:5]})")
    return df[df["qid"].isin(qid_set)].copy().reset_index(drop=True)


def split_by_questions(df, split_spec):
    """Partition `df` into split DataFrames according to `split_spec`."""
    qids_by_split = load_split_qids(split_spec)
    return {s: filter_by_qids(df, qids) for s, qids in qids_by_split.items()}


def build_persona_dataset(
    df,
    *,
    attr_sets,
    include_country,
    min_n=None,
    desc="build_persona_dataset",
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build (or load from pickle cache) persona records (`attr_sets=None` for respondent-level)."""
    cache_file = None
    if cache_dir is not None:
        unique_qids = tuple(sorted(df["qid"].unique()))
        key = repr((len(df), unique_qids, attr_sets, include_country, min_n, PERSONA_MAPS))
        digest = hashlib.md5(key.encode()).hexdigest()[:8]
        cache_file = Path(cache_dir) / f"{desc}_min{min_n}_{digest}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                records = pickle.load(f)
            print(f"Loaded {len(records):,} records from {cache_file.name}")
            return records

    candidate_cols = [c for c in PERSONA_NAMES if include_country or c != "c_alphan"]
    records = []
    q_groups = list(df.groupby("qid", sort=False))
    for _, q_df in tqdm(q_groups, desc=desc, unit="question"):
        active_cols = [c for c in candidate_cols if c in q_df.columns and q_df[c].notna().any()]
        active_set = set(active_cols)
        if attr_sets is None:
            for row in q_df.to_dict("records"):
                cols = [c for c in active_cols if pd.notna(row[c])]
                records.append(make_record(row, cols))
            continue

        sets = [[attr_sets]] if isinstance(attr_sets, str) else attr_sets
        for cols in sets:
            cols = [cols] if isinstance(cols, str) else list(cols)
            if include_country and "c_alphan" not in cols:
                cols = ["c_alphan", *cols]
            elif not include_country:
                cols = [c for c in cols if c != "c_alphan"]
            if not cols:
                if min_n is None or len(q_df) >= min_n:
                    records.append(make_record(q_df, cols))
                continue
            if not active_set.issuperset(cols):
                continue
            for _, sub_df in q_df.groupby(cols, sort=False, observed=True):
                if min_n is None or len(sub_df) >= min_n:
                    records.append(make_record(sub_df, cols))

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(records):,} records to {cache_file.name}")

    return records


def build_long_records(long_df, persona_cols, *, include_country, year="long", persona_names=ALL_PERSONA_NAMES):
    """Convert each row of a *_long.parquet DataFrame into a record dict compatible with evaluate_records()."""
    if include_country and "c_alphan" not in long_df.columns:
        raise KeyError("Expected 'c_alphan' column in long_df when include_country=True")
    if "one_hot" not in long_df.columns:
        raise KeyError("Expected 'one_hot' column in long_df; call add_one_hot(long_df) first")
    has_descs = "option_descriptions" in long_df.columns
    has_caseid = "CASEID" in long_df.columns
    has_arm = "arm" in long_df.columns
    has_year = "year" in long_df.columns

    if include_country:
        cols = ["c_alphan", *[c for c in persona_cols if c != "c_alphan"]]
    else:
        cols = [c for c in persona_cols if c != "c_alphan"]

    records = []
    for row in long_df.to_dict("records"):
        personas = {c: row[c] for c in cols if c in row and pd.notna(row[c])}
        opt_descs = (
            list(row["option_descriptions"])
            if has_descs and row["option_descriptions"] is not None
            else None
        )
        rec = {
            "qid": row["qid"],
            "year": row["year"] if has_year else year,
            "v": row["v"],
            "type": row["type"],
            "question": row["question"],
            "option_texts": list(row["option_texts"]),
            "option_descriptions": opt_descs,
            "personas": personas,
            "attribute": "+".join(personas.keys()) if personas else "none",
            "persona": ", ".join(f"{persona_names[c]}={v}" for c, v in personas.items()) if personas else "All",
            "n": 1,
            "human_dist": row["one_hot"],
        }
        if has_caseid:
            rec["CASEID"] = row["CASEID"]
        if has_arm:
            rec["arm"] = row["arm"]
        records.append(rec)
    return records


def make_view(rec, rng=None, persona_names=ALL_PERSONA_NAMES):
    """Build a single-question Kev view from a record dict (supports optional option_descriptions)."""
    qid = rec["qid"]
    q_type = rec["type"]
    option_texts = rec["option_texts"]
    option_descs = rec["option_descriptions"] if "option_descriptions" in rec else None
    human_dist = rec["human_dist"]

    items = list(rec["personas"].items())
    if rng is not None and len(items) > 1:
        rng.shuffle(items)

    state_lines = [f"{persona_names[c]}: {v}" for c, v in items]

    if q_type == "score":
        if len(option_texts) != len(human_dist):
            raise ValueError(f"{qid}: option_texts ({len(option_texts)}) != human_dist ({len(human_dist)})")
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
        "state": "\n".join(state_lines),
        "questions": {
            qid: {
                "type": q_type,
                "instructions": rec["question"],
                "criteria": criteria,
                "target": target,
            }
        },
    }


make_long_view = make_view


def make_grouped_view(group_records, rng=None, persona_names=ALL_PERSONA_NAMES):
    """Build a multi-question Kev view from a list of records sharing the same persona state.

    - Shuffles the persona attributes once at the group level via `rng`.
    - Keys each question by `f"{idx}:{qid}"` so all records and targets in the group are preserved.
    """
    if not group_records:
        raise ValueError("group_records must not be empty")
    first_personas = group_records[0]["personas"]
    first_view = make_view(group_records[0], rng=rng, persona_names=persona_names)
    state = first_view["state"]
    questions = {f"0:{group_records[0]['qid']}": first_view["questions"][group_records[0]["qid"]]}

    for idx, rec in enumerate(group_records[1:], start=1):
        if rec["personas"] != first_personas:
            raise ValueError("All records in group_records must share the exact same personas dict")
        v = make_view(rec, rng=None, persona_names=persona_names)
        questions[f"{idx}:{rec['qid']}"] = v["questions"][rec["qid"]]

    return {"state": state, "questions": questions}


def _record_chars(r):
    """Estimated formatted character length including Kev chat-template framing per question and per option."""
    return 32 + len(r["question"]) + sum(24 + len(str(o)) for o in r["option_texts"])


def _question_spec_chars(q_spec):
    """Estimated formatted character length of a single question entry inside a Kev view."""
    crit = q_spec["criteria"]
    if isinstance(crit, dict):
        opt_chars = sum(24 + len(str(o)) + (len(str(d)) + 2 if d is not None else 0) for o, d in crit.items())
    else:
        opt_chars = sum(24 + len(str(o)) for o in crit)
    return 32 + len(q_spec["instructions"]) + opt_chars


def group_eval_views(packed_views, *, max_questions=512, max_chars=180_000):
    """Group state-packed evaluation views into length-sorted batches bounded by `max_questions` and padded `max_chars`."""
    if max_questions < 1:
        raise ValueError(f"max_questions must be >= 1, got {max_questions}")
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")

    stats = []
    for i, v in enumerate(packed_views):
        qs = v["questions"]
        if not qs:
            raise ValueError(f"packed_views[{i}] has no questions")
        s_chars = 16 + len(v["state"])
        max_q_chars = max(_question_spec_chars(q) for q in qs.values())
        stats.append((i, len(qs), s_chars, max_q_chars))

    stats.sort(key=lambda x: (x[3], x[2]))
    batches = []
    cur = []
    cur_q = 0
    cur_max_s = 0
    cur_max_q = 0
    for idx, n_q, s_chars, max_q_chars in stats:
        next_q = cur_q + n_q
        next_max_s = max(cur_max_s, s_chars)
        next_max_q = max(cur_max_q, max_q_chars)
        next_chars = (len(cur) + 1) * next_max_s + next_q * next_max_q
        if cur and (next_q > max_questions or next_chars > max_chars):
            batches.append(cur)
            cur = []
            cur_q = 0
            cur_max_s = 0
            cur_max_q = 0
            next_q = n_q
            next_max_s = s_chars
            next_max_q = max_q_chars
        cur.append(idx)
        cur_q = next_q
        cur_max_s = next_max_s
        cur_max_q = next_max_q
    if cur:
        batches.append(cur)
    return batches


def group_train_records(records, *, max_questions=8, max_chars=2400, rng=None):
    """Group training records by persona state into length-matched multi-question chunks.

    - Groups records sharing `tuple(sorted(r["personas"].items()))`.
    - Shuffles within each persona state (if `rng` is given), then sorts by formatted question+options length
      so questions packed in the same group have minimal branch padding.
    - Splits each persona's questions into chunks bounded by `max_questions` and padded `max_chars` (bounding peak VRAM).
    """
    if max_questions < 1:
        raise ValueError(f"max_questions must be >= 1, got {max_questions}")
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")

    by_state = defaultdict(list)
    for r in records:
        key = tuple(sorted(r["personas"].items()))
        by_state[key].append(r)

    groups = []
    for state_recs in by_state.values():
        state_recs = list(state_recs)
        if rng is not None and len(state_recs) > 1:
            rng.shuffle(state_recs)
        state_recs.sort(key=_record_chars)
        cur = []
        for r in state_recs:
            q_chars = _record_chars(r)
            # Since state_recs is sorted ascending by q_chars, adding r pads all (len(cur) + 1) branches to q_chars
            if cur and (len(cur) >= max_questions or (len(cur) + 1) * q_chars > max_chars):
                groups.append(cur)
                cur = []
            cur.append(r)
        if cur:
            groups.append(cur)

    if rng is not None:
        rng.shuffle(groups)
    return groups


def pack_views_by_state(views):
    """Deduplicate (state, qid) pairs and pack all questions sharing a state into one multi-question view."""
    state_to_idx = {}
    packed_views = []
    qid_to_pos = []
    unpack_idx = []

    for v in views:
        state = v["state"]
        if state not in state_to_idx:
            state_to_idx[state] = len(packed_views)
            packed_views.append({"state": state, "questions": {}})
            qid_to_pos.append({})
        s_idx = state_to_idx[state]
        s_questions = packed_views[s_idx]["questions"]
        s_qpos = qid_to_pos[s_idx]

        row_map = []
        for qid, q_spec in v["questions"].items():
            if qid not in s_qpos:
                s_qpos[qid] = len(s_questions)
                s_questions[qid] = q_spec
            else:
                prev = s_questions[qid]
                if (prev["type"], prev["instructions"], prev["criteria"]) != (
                    q_spec["type"], q_spec["instructions"], q_spec["criteria"]
                ):
                    raise ValueError(f"Conflicting question definition for qid={qid!r}")
            row_map.append((s_idx, s_qpos[qid]))
        unpack_idx.append(row_map)

    return packed_views, unpack_idx


def group_records_by_state(records, view_fn=make_view, eval_fn=None, name=None, valid=False):
    """Create the state-grouped evaluation bundle ONCE so it can be reused across validation steps."""
    views = [view_fn(r) for r in records]
    packed_views, unpack_idx = pack_views_by_state(views)
    return {
        "name": name,
        "valid": bool(valid),
        "records": records,
        "packed_views": packed_views,
        "unpack_idx": unpack_idx,
        "encs": None,
        "encs_key": None,
        "last_stats": None,
        "eval_fn": eval_fn,
        "last_extra_df": None,
    }


def build_eval_set(
    df,
    *,
    attr_sets,
    include_country,
    min_n=None,
    grouped=True,
    eval_fn=None,
    name=None,
    valid=False,
    persona_cols=None,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """Build an evaluation set from a standardized long DataFrame `df` (`attr_sets` and `include_country` required)."""
    if persona_cols is not None and attr_sets is None:
        records = build_long_records(df, persona_cols=persona_cols, include_country=include_country)
    else:
        desc = f"{name}_records" if name else "eval_records"
        records = build_persona_dataset(
            df,
            attr_sets=attr_sets,
            include_country=include_country,
            min_n=min_n,
            desc=desc,
            cache_dir=cache_dir,
        )

    if grouped:
        return group_records_by_state(
            records, view_fn=make_view, eval_fn=eval_fn, name=name, valid=valid
        )
    return records


@torch.no_grad()
def predict_probs(model, records, batch_size=64, show_progress=True, view_fn=make_view, return_stats=False):
    """Regular inference without state grouping (1 question per view via model.forward).

    Deduplicates identical (state, qid) views so repeated respondent rows are only evaluated once,
    then maps probability vectors back to the flat `records` order.
    """
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()

    t_prep0 = time.perf_counter()
    unique_views = []
    rec_to_unique = []
    seen = {}
    for r in records:
        v = view_fn(r)
        qid = next(iter(v["questions"]))
        key = (v["state"], qid)
        if key not in seen:
            seen[key] = len(unique_views)
            unique_views.append(v)
        rec_to_unique.append(seen[key])

    encs = [model.prepare(v)[0] for v in unique_views]
    t_prep = time.perf_counter() - t_prep0

    torch.cuda.synchronize()
    t_gpu0 = time.perf_counter()
    unique_preds = []
    iterator = range(0, len(encs), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="predict (single-state)", unit="batch", leave=False)
    for start in iterator:
        b_encs = encs[start:start + batch_size]
        logits_batch = model.forward(b_encs)
        for question_logits in logits_batch:
            if len(question_logits) != 1:
                raise ValueError(f"Expected 1 question per record, got {len(question_logits)}")
            unique_preds.append(torch.softmax(question_logits[0].float(), dim=-1).cpu().numpy())
    torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t_gpu0

    if was_training:
        model.train()

    preds = [unique_preds[u] for u in rec_to_unique]
    stats = {
        "mode": "single-state",
        "n_records": len(records),
        "n_views_run": len(unique_views),
        "prep_sec": t_prep,
        "gpu_sec": t_gpu,
        "total_sec": time.perf_counter() - t0,
    }
    return (preds, stats) if return_stats else preds


def release_cuda_graphs(model):
    """Break closure reference cycles inside `CudaGraphs` and release `model.net.graphs`."""
    graphs = model.net.graphs
    if graphs is not None:
        graphs.pending.clear()
        graphs.graphs.clear()
        graphs.failed.clear()
        graphs.eager_runs.clear()
        model.net.graphs = None


def _effective_weight(mod, dtype):
    if hasattr(mod, "base_layer"):
        w = mod.base_layer.weight.to(dtype)
        for adapter in mod.active_adapters:
            if adapter in mod.lora_A:
                w = w + mod.get_delta_weight(adapter).to(dtype)
        return w.contiguous()
    return mod.weight.to(dtype).contiguous()


def _concat_weights(dtype, *mods):
    return torch.cat([_effective_weight(m, dtype) for m in mods], 0).contiguous()


@contextlib.contextmanager
def fused_eval_context(model, eval_dtype=None):
    """Temporarily bind `kev.fused_qwen35` kernels + `CudaGraphs` for fast evaluation, offloading original weights to CPU RAM."""
    lm = model.net.lm
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    dev = base.embed_tokens.weight.device
    orig_emb_dtype = base.embed_tokens.weight.dtype
    target_dtype = eval_dtype if eval_dtype is not None else orig_emb_dtype
    can_fuse = all(hasattr(layer.mlp, "gate_proj") for layer in base.layers)

    saved_dicts = []
    saved_weights = []
    offloaded_params = []
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(dev)

    def _swap_weight_dtype(mod):
        if mod.weight.dtype != target_dtype:
            orig_w = mod.weight
            cpu_t = orig_w.data.cpu()
            new_w = torch.nn.Parameter(orig_w.data.to(target_dtype), requires_grad=False)
            orig_w.data = torch.empty(0, dtype=orig_w.dtype, device=dev)
            saved_weights.append((mod, orig_w, cpu_t))
            mod.weight = new_w

    def _offload_proj(proj):
        wp = proj.base_layer.weight if hasattr(proj, "base_layer") else proj.weight
        offloaded_params.append((wp, wp.data.cpu()))
        wp.data = torch.empty(0, dtype=wp.dtype, device=dev)

    try:
        _swap_weight_dtype(base.embed_tokens)
        _swap_weight_dtype(base.norm)
        if can_fuse:
            _fix_nb()
            for layer in base.layers:
                mlp = layer.mlp
                down_mod = mlp.down_proj
                if layer.block_type == "linear_attention":
                    m = layer.linear_attn
                    out_mod = m.out_proj
                    for obj in (layer, m, mlp, out_mod, down_mod):
                        saved_dicts.append((obj, dict(obj.__dict__)))
                    _swap_weight_dtype(m.norm)
                    m.in_proj = _concat_weights(target_dtype, m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a)
                    m.splits = [m.conv_dim, m.value_dim, m.num_v_heads, m.num_v_heads]
                    m.conv_weight = m.conv1d.weight.squeeze(1).to(target_dtype).contiguous()
                    out_w = _effective_weight(out_mod, target_dtype)
                    out_mod.forward = lambda x, w=out_w: F.linear(x, w)
                    m.forward = types.MethodType(deltanet_forward, m)
                    for proj in (m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a, out_mod):
                        _offload_proj(proj)
                else:
                    m = layer.self_attn
                    out_mod = m.o_proj
                    for obj in (layer, m, mlp, out_mod, down_mod):
                        saved_dicts.append((obj, dict(obj.__dict__)))
                    q_out = m.q_proj.base_layer.out_features if hasattr(m.q_proj, "base_layer") else m.q_proj.out_features
                    k_out = m.k_proj.base_layer.out_features if hasattr(m.k_proj, "base_layer") else m.k_proj.out_features
                    v_out = m.v_proj.base_layer.out_features if hasattr(m.v_proj, "base_layer") else m.v_proj.out_features
                    m.qkv = _concat_weights(target_dtype, m.q_proj, m.k_proj, m.v_proj)
                    m.splits = [q_out, k_out, v_out]
                    m.q_norm_weight = 1.0 + m.q_norm.weight.float()
                    m.k_norm_weight = 1.0 + m.k_norm.weight.float()
                    out_w = _effective_weight(out_mod, target_dtype)
                    out_mod.forward = lambda x, w=out_w: F.linear(x, w)
                    m.forward = types.MethodType(attention_forward, m)
                    for proj in (m.q_proj, m.k_proj, m.v_proj, out_mod):
                        _offload_proj(proj)
                mlp.gate_up = _concat_weights(target_dtype, mlp.gate_proj, mlp.up_proj)
                down_w = _effective_weight(down_mod, target_dtype)
                down_mod.forward = lambda x, w=down_w: F.linear(x, w)
                mlp.forward = types.MethodType(mlp_forward, mlp)
                for proj in (mlp.gate_proj, mlp.up_proj, down_mod):
                    _offload_proj(proj)
                layer.input_norm_weight = 1.0 + layer.input_layernorm.weight.float()
                layer.post_norm_weight = 1.0 + layer.post_attention_layernorm.weight.float()
                layer.forward = types.MethodType(decoder_forward, layer)
        model.net.graphs = CudaGraphs(model.net.lm, model.net.pad_id)
        yield
    finally:
        release_cuda_graphs(model)
        for obj, orig_dict in saved_dicts:
            obj.__dict__.clear()
            obj.__dict__.update(orig_dict)
            if "forward" in obj.__dict__ and isinstance(obj.__dict__["forward"], types.MethodType) and obj.__dict__["forward"].__func__ is type(obj).forward:
                del obj.__dict__["forward"]
        for mod, orig_w, cpu_t in saved_weights:
            orig_w.data = cpu_t.to(device=dev)
            mod.weight = orig_w
        for wp, cpu_t in offloaded_params:
            wp.data = cpu_t.to(device=dev)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, dev)
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def predict_probs_grouped(
    model,
    grouped_records,
    *,
    max_questions=512,
    max_chars=180_000,
    show_progress=True,
    return_stats=False,
    capture_graphs=True,
):
    """Shared-state inference using the pre-grouped evaluation bundle from `group_records_by_state()`.

    - Tokenizes `packed_views` once per `(tokenizer, option_isolation)` and caches `grouped_records["encs"]` in memory and `.cache/`.
    - Packs states into length-sorted batches bounded by `max_questions` and padded `max_chars`.
    - Uses `fused_eval_context` + `CudaGraphs` for fast length-bucketed evaluation and frees temporary buffers on exit.
    - Unpacks predictions back into the flat `records` order so `evaluate_records()` works unchanged.
    """
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    label = grouped_records["name"]
    if not label:
        raise ValueError("grouped_records['name'] must be a non-empty string")

    t_prep0 = time.perf_counter()
    packed_views = grouped_records["packed_views"]
    encs_key = (model.tok.name_or_path, model.encoder.option_isolation, len(packed_views))
    if (
        grouped_records["encs"] is None
        or "encs_key" not in grouped_records
        or grouped_records["encs_key"] != encs_key
    ):
        h = hashlib.sha256()
        h.update(f"tok={model.tok.name_or_path}|iso={model.encoder.option_isolation}|n={len(packed_views)}".encode())
        for v in packed_views:
            h.update(json.dumps(v, sort_keys=True).encode())
        cache_path = DEFAULT_CACHE_DIR / f"encs_{label}_{h.hexdigest()[:12]}.pkl"
        if cache_path.exists():
            grouped_records["encs"] = pickle.loads(cache_path.read_bytes())
        else:
            views_iter = tqdm(packed_views, desc=f"Tokenizing {label}", unit="state", leave=False) if show_progress else packed_views
            grouped_records["encs"] = [model.prepare(v)[0] for v in views_iter]
            DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(pickle.dumps(grouped_records["encs"], protocol=pickle.HIGHEST_PROTOCOL))
        grouped_records["encs_key"] = encs_key
    encs = grouped_records["encs"]
    batches = group_eval_views(packed_views, max_questions=max_questions, max_chars=max_chars)
    t_prep = time.perf_counter() - t_prep0

    outer_graphs = model.net.graphs is not None
    ctx = contextlib.nullcontext() if outer_graphs else fused_eval_context(
        model, eval_dtype=torch.bfloat16 if torch.is_autocast_enabled("cuda") else None
    )

    torch.cuda.synchronize()
    t_gpu0 = time.perf_counter()
    try:
        with ctx:
            packed_probs = [None] * len(encs)
            batch_iter = tqdm(batches, desc=f"Evaluating {label} (shared-state)", unit="batch", leave=False) if show_progress else batches
            for batch_indices in batch_iter:
                b_encs = [encs[i] for i in batch_indices]
                probs_batch, _ = model.net.probs_batch(b_encs, [None] * len(b_encs), [False] * len(b_encs))
                if capture_graphs and model.net.graphs.capture_due(False):
                    model.net.graphs.capture_pending()
                for orig_idx, q_list in zip(batch_indices, probs_batch, strict=True):
                    packed_probs[orig_idx] = [p.float().numpy() for p in q_list]
        torch.cuda.synchronize()
    finally:
        if was_training:
            model.train()
    t_gpu = time.perf_counter() - t_gpu0

    preds = []
    for row_map in grouped_records["unpack_idx"]:
        if len(row_map) != 1:
            raise ValueError(f"Expected 1 question per record in unpack_idx, got {len(row_map)}")
        s_idx, q_pos = row_map[0]
        preds.append(packed_probs[s_idx][q_pos])

    stats = {
        "mode": "shared-state",
        "n_records": len(grouped_records["records"]),
        "n_views_run": len(encs),
        "n_batches": len(batches),
        "prep_sec": t_prep,
        "gpu_sec": t_gpu,
        "total_sec": time.perf_counter() - t0,
    }
    grouped_records["last_stats"] = stats
    return (preds, stats) if return_stats else preds


def evaluate_pair(human_dist, model_dist):
    """Compute binary-log cross-entropy and KL divergence between `human_dist` and `model_dist`."""
    if human_dist.shape != model_dist.shape:
        raise ValueError(f"Shape mismatch: {human_dist.shape} != {model_dist.shape}")
    clipped_model = np.clip(model_dist, 1e-12, 1.0)
    nonzero = human_dist > 0
    ce = float(-np.sum(human_dist[nonzero] * np.log2(clipped_model[nonzero])))
    kl = float(np.sum(human_dist[nonzero] * np.log2(human_dist[nonzero] / clipped_model[nonzero])))
    return {"ce": ce, "dist_kl": kl}


def evaluate_records(records, preds, show_progress=False):
    """Evaluate per-record cross-entropy and KL divergence on flat `preds`."""
    if len(records) != len(preds):
        raise ValueError(f"Length mismatch: {len(records)} records vs {len(preds)} predictions")
    rows = []
    pairs = zip(records, preds, strict=True)
    if show_progress:
        pairs = tqdm(pairs, total=len(records), desc="evaluate", unit="record", leave=False)
    for r, model_dist in pairs:
        metrics = evaluate_pair(r["human_dist"], model_dist)
        rows.append({
            "qid": r["qid"],
            "year": r["year"],
            "v": r["v"],
            "type": r["type"],
            "attribute": r["attribute"],
            "persona": r["persona"],
            "n": r["n"],
            **metrics,
        })
    return pd.DataFrame(rows)


def evaluate_model(
    model,
    eval_data,
    *,
    max_questions=512,
    max_chars=180_000,
    batch_size=64,
    show_progress=True,
    view_fn=make_view,
    return_details=False,
):
    """Unified evaluation entry point accepting either flat `records` (list) or `grouped_records` (dict)."""
    if isinstance(eval_data, dict):
        records = eval_data["records"]
        preds, stats = predict_probs_grouped(
            model,
            eval_data,
            max_questions=max_questions,
            max_chars=max_chars,
            show_progress=show_progress,
            return_stats=True,
        )
        eval_fn = eval_data["eval_fn"]
    else:
        records = eval_data
        preds, stats = predict_probs(
            model, records, batch_size=batch_size, show_progress=show_progress, view_fn=view_fn, return_stats=True
        )
        eval_fn = None

    df_res = evaluate_records(records, preds, show_progress=False)
    summary_dict = {
        "ce": float(df_res["ce"].mean()),
        "dist_kl": float(df_res["dist_kl"].mean()),
    }
    if eval_fn is not None:
        extra_metrics, extra_df = eval_fn(records, preds)
        summary_dict.update(extra_metrics)
        if isinstance(eval_data, dict):
            eval_data["last_extra_df"] = extra_df
    summary = pd.Series(summary_dict)

    if return_details:
        return summary, df_res, preds, stats
    return summary, df_res
