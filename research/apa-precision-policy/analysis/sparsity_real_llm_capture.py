"""Real-LLM sparsity trace capture — streaming, zero-tile persistence.

Goal
----
Evaluate the XAttention-style antidiagonal-sum proxy against ground-truth
post-softmax tile mass on a real LLM's attention scores, and cross-tabulate
with the existing precision controller's INT8/FP16 decision. Output is a
small JSON summary; raw tiles are never written to disk.

Methodology (one-pass, online, per-tile aggregates only)
-------------------------------------------------------
For each prompt in a small corpus, for each transformer layer, for each
attention head, for each (q_block, k_block) tile of pre-softmax scores:

  1. Compute S_tile = Q[q_block] @ K[k_block]^T / sqrt(d).
  2. Apply causal mask (drop tiles strictly above the diagonal; for the
     diagonal tile, mask positions where j > i).
  3. Quantize S_tile to int8 with the same per-tile symmetric scheme the
     RTL test-vector generator uses, then run the bit-exact precision
     controller reference model to get the d_fp16 bit.
  4. Compute the proxy statistics on the *quantized int8 tile* — that is
     what the chip's sparsity controller would see in steady state.
     These are scalars only:
        - max(|S|), sum(|S|)
        - antidiag_sum at strides {1, 2, 4, 8, 16}
        - antidiag_max at stride 8
  5. Compute the ground-truth tile importance from the float32 path:
        a_row = softmax(S_row_full)  for each q row in the tile
        tile_mass[q] = sum_{k in k_block} a_row[k]
        tile_importance = tile_mass.mean()  (averaged over q rows in tile)
     This is "what fraction of attention weight this tile actually
     captures." It is what a perfect sparsity oracle would gate on.
  6. Append scalars to per-layer accumulators. Discard the tile.

After all prompts are processed, sweep antidiagonal-sum thresholds against
ground-truth tile-importance thresholds to produce the receiver-operator
curve we actually care about: for each chosen skip rate, what fraction of
true attention mass is preserved?

Storage discipline
------------------
The only persistent on-disk state is the model weight cache, which the
caller controls (see README at top of branch). No tile, attention matrix,
or per-head per-layer score tensor is ever written; everything is reduced
to scalars on the fly. Per-tile records (~16 floats + 1 byte) are
accumulated in RAM during the run.

For Qwen2-0.5B at seq_len=512, block=64:
  layers (24) * heads (14) * (8x8 blocks/256 ≈ 36 lower-triangular tiles)
  ≈ 12,000 tile records per prompt — well under 1 MB in RAM.

Usage
-----
    python analysis/sparsity_real_llm_capture.py
        [--model Qwen/Qwen2-0.5B] [--seq-len 512] [--block 64]
        [--max-prompts 2] [--delete-cache-after]

Defaults match `analysis/validate_real_llm_v2.py` so the precision
controller numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sw" / "reference_model"))

from precision_controller_ref import PrecisionController            # noqa: E402
from sparsity_controller_ref import SparsityControllerInfo          # noqa: E402

# Same constants as validate_real_llm_v2.py
BLOCK_SIZE       = 64
RATIO_THRESHOLD  = 10.0   # precision-controller threshold
STRIDES          = [1, 2, 4, 8, 16]
DEFAULT_MODEL    = "Qwen/Qwen2-0.5B"

# Same prose / code prompts as v2 — keeps the capture comparable.
LONG_PROMPTS = [
    ("prose", " ".join([
        "The development of artificial intelligence has proceeded through several distinct phases.",
        "Early systems relied on explicit rule-based programming where human experts encoded",
        "domain knowledge directly into software. These expert systems achieved impressive results",
        "in narrow domains but struggled to generalize. The shift toward machine learning",
        "represented a fundamental change in approach allowing systems to learn patterns from data",
        "rather than following hand-crafted rules. Deep learning further accelerated this trend",
        "by enabling models to learn hierarchical representations automatically from raw inputs.",
        "The attention mechanism introduced in the transformer architecture allowed models to",
        "selectively focus on relevant parts of the input when producing each output token.",
        "This led to dramatic improvements in language understanding and generation tasks.",
        "Large language models trained on vast corpora of text demonstrated emergent capabilities",
        "that were not explicitly programmed and were not anticipated from smaller models.",
        "The scaling laws observed in these models suggested that performance would continue",
        "to improve with more data more compute and more parameters following predictable trends.",
        "Whether this scaling will continue indefinitely or eventually plateau remains an open",
        "question in the field. Researchers are now exploring ways to make these models more",
        "efficient through techniques like quantization pruning and knowledge distillation.",
    ] * 30)),   # ~6000 words → ~8000 tokens, enough for max_length=4K
    ("code", """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        Q = self.w_q(q); K = self.w_k(k); V = self.w_v(v)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return self.w_o(torch.matmul(attn, V))
""" * 30),  # repeated source → ~8000 tokens, enough for max_length=4K
]


# ---------------------------------------------------------------------------
# Hooks — same shape as validate_real_llm_v2.AttentionCapture
# ---------------------------------------------------------------------------
class AttentionCapture:
    def __init__(self, model):
        self.q_acts: dict[int, torch.Tensor] = {}
        self.k_acts: dict[int, torch.Tensor] = {}
        self.hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn

            def make_hook(d, key):
                def h(_m, _inp, out):
                    d[key] = out.detach().float()
                return h

            self.hooks.append(attn.q_proj.register_forward_hook(make_hook(self.q_acts, layer_idx)))
            self.hooks.append(attn.k_proj.register_forward_hook(make_hook(self.k_acts, layer_idx)))

    def clear(self):
        self.q_acts.clear()
        self.k_acts.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Antidiagonal mask precomputation
# ---------------------------------------------------------------------------
def make_antidiag_masks(block_m: int, block_n: int) -> dict[int, np.ndarray]:
    """Return a {stride -> flat-index array} table for sampling the
    antidiagonal pattern `(i + j) & (stride-1) == 0` on a BxB tile."""
    i, j = np.indices((block_m, block_n))
    masks = {}
    for s in STRIDES:
        m = ((i + j) & (s - 1)) == 0
        masks[s] = np.flatnonzero(m.flatten())
    return masks


# ---------------------------------------------------------------------------
# Per-tile quantization (matches analysis/gen_rtl_testvectors.quantize_int8)
# ---------------------------------------------------------------------------
def quantize_int8_tile(tile_fp: np.ndarray) -> np.ndarray:
    """Symmetric per-tile int8 quantization. Matches the RTL TB generator
    so the precision-controller decision computed here is bit-equivalent
    to the chip's decision on the same data."""
    max_abs = float(np.abs(tile_fp).max())
    if max_abs < 1e-9:
        return np.zeros(tile_fp.shape, dtype=np.int8)
    scale = max_abs / 127.0
    return np.round(np.clip(tile_fp / scale, -127, 127)).astype(np.int8)


# ---------------------------------------------------------------------------
# Streaming aggregator
# ---------------------------------------------------------------------------
class TileAggregator:
    """Online accumulator over per-tile scalars. Stores one record per tile;
    no raw tiles. Final dump fits in a few MB even at large seq_lens."""

    def __init__(self):
        # Each entry: dict[float scalars + ints]. List of dicts is small enough
        # to keep in RAM during a run; we dump a summary JSON, not this list.
        self.records: list[dict] = []
        self.per_layer_fp16: dict[int, list[int]] = {}

    def record(self, layer: int, head: int, qb: int, kb: int,
               int8_tile: np.ndarray, soft_tile: np.ndarray,
               tile_mass: float, fp16_decision: bool,
               masks: dict[int, np.ndarray]) -> None:
        """soft_tile: post-softmax attention A for this tile, flat float32.

        Records four proxy families per stride:
          - a{S}   : antidiag sum on |int8 S|        (XAttention-style on raw)
          - am{S}  : antidiag max on |int8 S|        (spike-aware variant)
          - ap{S}  : antidiag sum on positive int8 S (sign-bit gated accumulator)
          - aA{S}  : antidiag sum on softmax A       (paper-faithful, expensive in HW)
        """
        signed_q = int8_tile.astype(np.int32).flatten()
        abs_q    = np.abs(signed_q)
        soft_f   = soft_tile.astype(np.float32).flatten()
        rec = {
            "L":     int(layer),
            "H":     int(head),
            "qb":    int(qb),
            "kb":    int(kb),
            "fp16":  bool(fp16_decision),
            "mass":  float(tile_mass),               # ground truth: 0..1
            "max":   int(abs_q.max()),
            "sum":   int(abs_q.sum()),
            "pos_sum": int(np.maximum(signed_q, 0).sum()),  # full-tile positive sum
        }
        for s, idx in masks.items():
            sub_signed = signed_q[idx]
            sub_abs    = abs_q[idx]
            rec[f"a{s}"]  = int(sub_abs.sum())                 # |S| sum
            rec[f"am{s}"] = int(sub_abs.max())                 # |S| max
            rec[f"ap{s}"] = int(np.maximum(sub_signed, 0).sum())  # positive-only sum
            rec[f"aA{s}"] = float(soft_f[idx].sum())           # softmax A sum
        self.records.append(rec)
        self.per_layer_fp16.setdefault(layer, []).append(int(fp16_decision))


# ---------------------------------------------------------------------------
# Per-(layer, head) per-tile processor
# ---------------------------------------------------------------------------
def process_attention(Q: torch.Tensor, K: torch.Tensor,
                      head_dim: int, layer: int,
                      agg: TileAggregator, masks: dict[int, np.ndarray],
                      pc: PrecisionController, block: int) -> None:
    """Q, K: [B=1, H, N, d] already in float32 on CPU.

    Builds the per-head N×N scores, causal-masks, softmaxes, then walks
    lower-triangular block tiles. For each tile: quantize, run precision
    controller, accumulate scalars, discard.
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    n_blocks = math.ceil(N / block)

    # Build the causal mask once for this seq, on the same device as Q.
    dev = Q.device
    rows = torch.arange(N, device=dev).unsqueeze(1)   # [N, 1]
    cols = torch.arange(N, device=dev).unsqueeze(0)   # [1, N]
    causal = (cols <= rows).float()                   # 1 = keep, 0 = mask out
    neg_inf_mask = (1.0 - causal) * -1e9              # additive mask in score domain

    for h in range(H):
        S = (Q[0, h] @ K[0, h].T) * scale         # [N, N]
        S = S + neg_inf_mask
        # Numerically stable softmax across last dim.
        S_max = S.max(dim=-1, keepdim=True).values
        A = torch.softmax(S - S_max, dim=-1)      # [N, N]
        S_np = S.cpu().numpy()
        A_np = A.cpu().numpy()

        for qb in range(n_blocks):
            q_lo = qb * block
            q_hi = min(q_lo + block, N)
            for kb in range(qb + 1):              # lower triangular incl. diagonal
                k_lo = kb * block
                k_hi = min(k_lo + block, N)

                # Skip if the tile is smaller than block (right/bottom edge).
                # We could pad-and-mask, but the chip operates on full tiles;
                # leaving edge tiles out keeps the comparison clean.
                if (q_hi - q_lo) != block or (k_hi - k_lo) != block:
                    continue

                tile_S = S_np[q_lo:q_hi, k_lo:k_hi]            # raw scores
                tile_A = A_np[q_lo:q_hi, k_lo:k_hi]            # softmaxed weights

                # Ground truth: tile mass = row-mean of summed softmax weights.
                # tile_A.sum(axis=1) gives, for each q row, the share of attention
                # this tile captured (already normalized since A rows sum to 1).
                tile_mass = float(tile_A.sum(axis=1).mean())

                # int8 path — what the chip sees. Inline numpy reduction
                # equivalent to PrecisionController.process_tile (the streaming
                # ref model loops 4096 times per tile in Python, far too slow
                # at seq_len=4K).  The condition `max*N > 10*sum` on the
                # int8-quantized tile is the exact RTL formula.
                t_i8 = quantize_int8_tile(tile_S)
                abs_q = np.abs(t_i8.astype(np.int32))
                max_q = int(abs_q.max())
                sum_q = int(abs_q.sum())
                d_fp16 = (max_q * (block * block)) > (10 * sum_q)

                agg.record(
                    layer=layer, head=h, qb=qb, kb=kb,
                    int8_tile=t_i8,
                    soft_tile=tile_A,
                    tile_mass=tile_mass,
                    fp16_decision=bool(d_fp16),
                    masks=masks,
                )


# ---------------------------------------------------------------------------
# Analysis post-pass
# ---------------------------------------------------------------------------
def analyze(records: list[dict], block: int) -> dict:
    """Sweep antidiag-sum thresholds against ground-truth tile mass.

    Returns the receiver-operator data and skip-rate / mass-preserved
    curves the methodology section reports.
    """
    if not records:
        return {"error": "no tiles captured"}

    arr = {k: np.array([r[k] for r in records]) for k in records[0].keys()}

    n = arr["L"].size
    fp16_total = int(arr["fp16"].sum())
    summary: dict = {
        "n_tiles":     n,
        "fp16_pct":    100.0 * fp16_total / n,
        "int8_pct":    100.0 * (n - fp16_total) / n,
        "mass":        {
            "mean":   float(arr["mass"].mean()),
            "median": float(np.median(arr["mass"])),
            "p10":    float(np.percentile(arr["mass"], 10)),
            "p90":    float(np.percentile(arr["mass"], 90)),
        },
        "correlation": {},
        "tau_sweep":   [],
        "oracle_skip_curves": [],
    }

    # Correlation of each proxy with ground-truth tile mass.
    m = arr["mass"].astype(np.float64)
    for s in STRIDES:
        for prefix, label in [("a",  "abs_S_sum"),
                              ("am", "abs_S_max"),
                              ("ap", "pos_S_sum"),
                              ("aA", "softmax_A_sum")]:
            a = arr[f"{prefix}{s}"].astype(np.float64)
            if a.std() > 0 and m.std() > 0:
                r = float(np.corrcoef(a, m)[0, 1])
            else:
                r = float("nan")
            summary["correlation"][f"{label}_s{s}"] = r
    # And the obvious baselines: full tile sum/max/pos_sum
    # (= what the precision controller already accumulates, plus the
    # whole-tile equivalent of the cheap positive-only proxy).
    for full in ["sum", "max", "pos_sum"]:
        a = arr[full].astype(np.float64)
        if a.std() > 0 and m.std() > 0:
            r = float(np.corrcoef(a, m)[0, 1])
        else:
            r = float("nan")
        summary["correlation"][f"full_{full}"] = r

    # Per-layer correlation breakdown for the candidate proxies.
    summary["per_layer_correlation_a8"]  = {}
    summary["per_layer_correlation_ap8"] = {}
    summary["per_layer_correlation_aA8"] = {}
    summary["per_layer_fp16_pct"] = {}
    layers = sorted(set(int(L) for L in arr["L"].tolist()))
    for L in layers:
        sel = arr["L"] == L
        if sel.sum() < 4:
            continue
        m_L = arr["mass"][sel].astype(np.float64)
        for proxy_key, dest in [("a8",  "per_layer_correlation_a8"),
                                ("ap8", "per_layer_correlation_ap8"),
                                ("aA8", "per_layer_correlation_aA8")]:
            a_L = arr[proxy_key][sel].astype(np.float64)
            if a_L.std() > 0 and m_L.std() > 0:
                summary[dest][str(L)] = float(np.corrcoef(a_L, m_L)[0, 1])
            else:
                summary[dest][str(L)] = float("nan")
        summary["per_layer_fp16_pct"][str(L)] = 100.0 * float(
            arr["fp16"][sel].astype(np.float64).mean()
        )

    # Threshold sweep on stride=8 for each candidate proxy.
    masses = arr["mass"].astype(np.float64)
    mass_total = float(masses.sum())
    fp16 = arr["fp16"].astype(bool)
    summary["tau_sweep"] = {}
    for proxy_key, label in [
        ("a8",  "abs_S_sum_s8"),
        ("am8", "abs_S_max_s8"),
        ("ap8", "pos_S_sum_s8"),
        ("aA8", "softmax_A_sum_s8"),
    ]:
        proxy = arr[proxy_key].astype(np.float64)
        mean_v = float(proxy.mean())
        sweep_rows = []
        for tau in [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]:
            thr = tau * mean_v
            skip = proxy < thr
            n_skip = int(skip.sum())
            mass_kept = float(masses[~skip].sum())
            sweep_rows.append({
                "tau": tau,
                "threshold": thr,
                "skip_rate":      n_skip / n,
                "n_skip":         n_skip,
                "mass_kept_frac": mass_kept / max(mass_total, 1e-9),
                "skipped_fp16":   int((skip & fp16).sum()),
                "skipped_int8":   int((skip & ~fp16).sum()),
            })
        summary["tau_sweep"][label] = sweep_rows

    # Oracle curve: if we ranked by ground-truth tile_mass and skipped the
    # lowest k%, how much mass would we lose? This is the upper bound on any
    # sparsity controller's quality.
    sorted_mass = np.sort(masses)              # ascending
    total_mass = sorted_mass.sum()
    for k in [10, 25, 50, 75, 90]:
        cut = int(np.ceil(k / 100.0 * n))
        kept = total_mass - sorted_mass[:cut].sum()
        summary["oracle_skip_curves"].append({
            "skip_pct": k,
            "mass_kept_frac": float(kept / max(total_mass, 1e-9)),
        })

    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--block",   type=int, default=BLOCK_SIZE)
    ap.add_argument("--max-prompts", type=int, default=len(LONG_PROMPTS))
    ap.add_argument("--max-layers", type=int, default=None,
                    help="Limit layers for a quick smoke pass (default: all).")
    ap.add_argument("--delete-cache-after", action="store_true",
                    help="rm the HF cache for this model after the run.")
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" /
                                        "sparsity_real_llm_stats.json"))
    args = ap.parse_args(argv)

    print(f"[setup] model={args.model}  seq_len={args.seq_len}  block={args.block}",
          flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="eager",
    )
    model.eval()
    cfg = model.config
    n_layers     = cfg.num_hidden_layers
    num_q_heads  = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim     = cfg.hidden_size // num_q_heads
    layers_to_use = min(args.max_layers or n_layers, n_layers)
    print(f"[setup] layers={n_layers} (using {layers_to_use})  "
          f"Hq={num_q_heads}  Hkv={num_kv_heads}  d={head_dim}", flush=True)

    capture = AttentionCapture(model)
    masks = make_antidiag_masks(args.block, args.block)
    agg   = TileAggregator()
    pc    = PrecisionController()

    seq_lens = []
    for pname, ptext in LONG_PROMPTS[: args.max_prompts]:
        inputs = tokenizer(ptext, return_tensors="pt",
                           truncation=True, max_length=args.seq_len).to(model.device)
        seq = inputs["input_ids"].shape[1]
        seq_lens.append(seq)
        print(f"[run] prompt={pname!r}  tokens={seq}", flush=True)
        with torch.no_grad():
            model(**inputs)

        for layer_idx in range(layers_to_use):
            Q_raw = capture.q_acts[layer_idx]            # [1, N, Hq*d]
            K_raw = capture.k_acts[layer_idx]            # [1, N, Hkv*d]
            B = Q_raw.shape[0]
            Q = Q_raw.view(B, seq, num_q_heads, head_dim).permute(0, 2, 1, 3)
            K = K_raw.view(B, seq, num_kv_heads, head_dim).permute(0, 2, 1, 3)
            if num_kv_heads != num_q_heads:
                K = K.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
            process_attention(Q.float(), K.float(),
                              head_dim=head_dim, layer=layer_idx,
                              agg=agg, masks=masks, pc=pc, block=args.block)
            # Free per-layer activations as soon as we're done.
            del Q, K, Q_raw, K_raw
        capture.clear()
        print(f"[run] layers done: {layers_to_use}  tiles so far: {len(agg.records)}",
              flush=True)

    capture.remove()

    print(f"[analyze] reducing {len(agg.records)} tile records", flush=True)
    summary = analyze(agg.records, block=args.block)
    summary["model"]     = args.model
    summary["seq_lens"]  = seq_lens
    summary["block"]     = args.block
    summary["layers_used"] = layers_to_use

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {args.out}", flush=True)

    # Brief stdout summary
    print()
    print(f"  Total tiles      : {summary['n_tiles']}")
    print(f"  FP16 / INT8 split: {summary['fp16_pct']:.2f}% / {summary['int8_pct']:.2f}%")
    print(f"  Tile mass median : {summary['mass']['median']:.4f}  "
          f"(p10={summary['mass']['p10']:.4f}, p90={summary['mass']['p90']:.4f})")
    print()
    print("  Proxy ↔ ground-truth tile mass correlation (pearson r):")
    for key, r in summary["correlation"].items():
        print(f"    {key:>22}: {r:+.4f}")
    print()
    print("  τ-sweep across proxies (skip if proxy < τ · mean):")
    for proxy_label, sweep_rows in summary["tau_sweep"].items():
        print(f"   --- proxy = {proxy_label} ---")
        print(f"      {'τ':>5}  {'skip%':>6}  {'mass kept':>9}  "
              f"{'skip∩FP16':>9}  {'skip∩INT8':>9}")
        for row in sweep_rows:
            print(f"      {row['tau']:>5.2f}  "
                  f"{100*row['skip_rate']:>6.1f}  "
                  f"{100*row['mass_kept_frac']:>8.2f}%  "
                  f"{row['skipped_fp16']:>9d}  "
                  f"{row['skipped_int8']:>9d}")
    print()
    print("  Oracle curve (skip lowest-mass tiles, mass kept):")
    for row in summary["oracle_skip_curves"]:
        print(f"    skip {row['skip_pct']:>2}%  →  mass kept "
              f"{100*row['mass_kept_frac']:.2f}%")

    if args.delete_cache_after:
        # Resolve cache dir and rm the model's snapshot. Conservative: only
        # remove paths that mention the model id, never the cache root.
        cache_root = Path(
            os.environ.get("HF_HOME") or
            os.environ.get("TRANSFORMERS_CACHE") or
            (Path.home() / ".cache" / "huggingface")
        )
        slug = "models--" + args.model.replace("/", "--")
        target = cache_root / "hub" / slug
        if target.exists():
            print(f"[cleanup] removing {target}", flush=True)
            shutil.rmtree(target, ignore_errors=True)
        else:
            print(f"[cleanup] no cache dir at {target}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
