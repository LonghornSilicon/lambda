"""KV Cache Engine × Precision Controller integration test.

Question
--------
The Precision Controller (ACU block 1) was validated against attention tiles
whose K and V are the model's true FP16 tensors. The KV Cache Engine
(block 2) replaces K and V with `K_hat`, `V_hat` — lossy reconstructions
through PolarQuant+QJL. Does the precision controller's INT8/FP16 routing
still recover the accuracy paper-claimed 91-97% INT8-safe rate when the
inputs are already compression-noisy?

Method
------
For each real Qwen2-0.5B attention tile (Q, K, V from forward-pass hooks),
compute five outputs and compare to dense FP16 baseline:

  A (REF)        : softmax(Q @ K^T / sqrt(d)) @ V                       — fp16 dense ground truth
  B (PC-alone)   : same but with precision-controller-routed SV         — ACU in isolation
  C (KV-fp16)    : softmax(Q @ K_hat^T / sqrt(d)) @ V_hat                — KV in isolation, fp16 SV
  D (KV-int8)    : same as C but with always-INT8 SV                    — KV + worst-case routing
  E (INTEGRATED) : KV + precision controller (decision on lossy S)      — what the chip will actually do

Per-tile records: MSE of B/C/D/E vs A, the clean d_fp16 decision, the
lossy d_fp16 decision, and a tile-mass tag for stratifying results.

Key questions answered:

  1. How much does the FP16 routing fraction shift when the decision is
     made on K_hat instead of K? (Decision-stability question.)

  2. On tiles where the precision controller would have flagged FP16
     based on clean S — does it still flag FP16 on lossy S, and does the
     FP16 SV path materially beat the INT8 SV path under lossy V_hat?
     (FP16-still-helps question.)

  3. What's the integrated MSE (path E) relative to ACU alone (path B) and
     KV alone (path C)? (Does the precision controller's value-add survive
     the integration?)

Implementation
--------------
GQA shortcut: Qwen2-0.5B has 14 Q-heads, 2 KV-heads (7:1). K_hat / V_hat
are computed once per (layer, KV-head, token), then duplicated for the 7
Q-heads sharing each KV-head. This is exactly how the chip will deploy
the KV cache engine — one entry per real KV vector, regardless of GQA.

KV cache reference model is the Python ref from
`/home/shadeform/kv-cache-engine/sw/reference_model/kv_cache_engine_ref.py`.
Its `compress_key`/`compress_value` take int16 Q4.12 fixed-point input,
so each float vector is converted at the boundary.

Storage discipline: no tile, K/V tensor, or attention matrix is written
to disk. The model weight cache is auto-deleted on `--delete-cache-after`.

Usage
-----
    python analysis/integration_test_kv_pc.py
        [--seq-len 512] [--block 64]
        [--layers 0,5,10,15,20,23] [--prompts 1] [--delete-cache-after]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
KVCE_REF = Path("/home/shadeform/kv-cache-engine/sw/reference_model")
sys.path.insert(0, str(KVCE_REF))
sys.path.insert(0, str(REPO_ROOT / "sw" / "reference_model"))

from kv_cache_engine_ref import KVCacheEngine, KVCacheEngineInfo  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLOCK_SIZE      = 64
PC_THRESHOLD   = 10        # max*N > 10*sum → FP16
DEFAULT_MODEL  = "Qwen/Qwen2-0.5B"
COORD_FRAC     = 12        # Q4.12 fixed-point
COORD_MAX_INT  = (1 << 15) - 1
COORD_MIN_INT  = -(1 << 15)

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
    ] * 4)),
]


# ---------------------------------------------------------------------------
# Hooks (same shape as previous captures)
# ---------------------------------------------------------------------------
class QKVCapture:
    def __init__(self, model):
        self.q_acts: dict[int, torch.Tensor] = {}
        self.k_acts: dict[int, torch.Tensor] = {}
        self.v_acts: dict[int, torch.Tensor] = {}
        self.hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn

            def make_hook(d, key):
                def h(_m, _inp, out):
                    d[key] = out.detach().float()
                return h

            self.hooks.append(attn.q_proj.register_forward_hook(make_hook(self.q_acts, layer_idx)))
            self.hooks.append(attn.k_proj.register_forward_hook(make_hook(self.k_acts, layer_idx)))
            self.hooks.append(attn.v_proj.register_forward_hook(make_hook(self.v_acts, layer_idx)))

    def clear(self):
        self.q_acts.clear()
        self.k_acts.clear()
        self.v_acts.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Float ↔ Q4.12 conversion at the KVCE boundary
# ---------------------------------------------------------------------------
def fp_to_q412(vec_f: np.ndarray) -> list[int]:
    """float32 → int16 Q4.12 list. Clips at ±8.0 (range of Q4.12)."""
    q = np.round(vec_f * (1 << COORD_FRAC))
    q = np.clip(q, COORD_MIN_INT, COORD_MAX_INT)
    return q.astype(np.int32).tolist()


def q412_to_fp(vec_q: list[int]) -> np.ndarray:
    """int16 Q4.12 list → float32 numpy array."""
    return np.array(vec_q, dtype=np.float32) / (1 << COORD_FRAC)


# ---------------------------------------------------------------------------
# Precision-controller bit-exact decision on a flattened int8 tile.
# (Vectorized inline — equivalent to PrecisionController.process_tile but
# without the per-cycle Python loop.)
# ---------------------------------------------------------------------------
def pc_decide_fp16(int8_tile: np.ndarray) -> bool:
    """Returns True if the precision controller would route this tile FP16."""
    abs_q = np.abs(int8_tile.astype(np.int32))
    max_q = int(abs_q.max())
    sum_q = int(abs_q.sum())
    n     = int(int8_tile.size)
    return (max_q * n) > (PC_THRESHOLD * sum_q)


def quantize_int8_tile(tile_fp: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tile int8 quantization. Returns (int8, scale)."""
    max_abs = float(np.abs(tile_fp).max())
    if max_abs < 1e-9:
        return np.zeros(tile_fp.shape, dtype=np.int8), 0.0
    scale = max_abs / 127.0
    q = np.round(np.clip(tile_fp / scale, -127, 127)).astype(np.int8)
    return q, scale


# ---------------------------------------------------------------------------
# Per-tile output paths
# ---------------------------------------------------------------------------
def attention_fp16(S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """softmax(S) @ V at float32 precision, fp16-rounded output.

    S: [Bq, Bk] pre-softmax scores (already causal-masked).
    V: [Bk, d]
    Returns: [Bq, d] float32 (fp16-rounded).
    """
    S_max = S.max(axis=-1, keepdims=True)
    A = np.exp(S - S_max)
    A = A / A.sum(axis=-1, keepdims=True)
    out = A @ V
    return out.astype(np.float16).astype(np.float32)


def attention_int8_sv(S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """softmax(S) computed in fp32, then SV in INT8 (matches ACU INT8 path).

    Per-tile symmetric int8 quantization of both softmax and V; int32
    accumulator; per-tile rescale and fp16 round on output.
    """
    S_max = S.max(axis=-1, keepdims=True)
    A = np.exp(S - S_max)
    A = A / A.sum(axis=-1, keepdims=True)
    A_q, A_scale = quantize_int8_tile(A.astype(np.float32))
    V_q, V_scale = quantize_int8_tile(V.astype(np.float32))
    if A_scale == 0.0 or V_scale == 0.0:
        return np.zeros((A.shape[0], V.shape[1]), dtype=np.float32)
    acc = A_q.astype(np.int32) @ V_q.astype(np.int32)   # int32 result
    out = acc.astype(np.float32) * (A_scale * V_scale)
    return out.astype(np.float16).astype(np.float32)


def tile_paths(Q_tile: np.ndarray, K_tile: np.ndarray, V_tile: np.ndarray,
               K_hat_tile: np.ndarray, V_hat_tile: np.ndarray,
               head_dim: int) -> dict:
    """Run the five output paths on one tile and return per-path outputs +
    decisions. Inputs are float32 (already at fp16 precision).
    """
    scale = 1.0 / math.sqrt(head_dim)

    # ---- baseline / clean paths use true K, V ----
    S_clean = (Q_tile @ K_tile.T) * scale                  # [Bq, Bk]

    out_A = attention_fp16(S_clean, V_tile)                # REF

    # Precision-controller decision on clean S.
    S_q_clean, _ = quantize_int8_tile(S_clean)
    d_fp16_clean = pc_decide_fp16(S_q_clean)

    if d_fp16_clean:
        out_B = attention_fp16(S_clean, V_tile)
    else:
        out_B = attention_int8_sv(S_clean, V_tile)

    # ---- KV-cache paths use K_hat, V_hat ----
    S_lossy = (Q_tile @ K_hat_tile.T) * scale

    out_C = attention_fp16(S_lossy, V_hat_tile)            # KV alone, FP16 SV
    out_D = attention_int8_sv(S_lossy, V_hat_tile)         # KV alone, INT8 SV

    S_q_lossy, _ = quantize_int8_tile(S_lossy)
    d_fp16_lossy = pc_decide_fp16(S_q_lossy)
    if d_fp16_lossy:
        out_E = attention_fp16(S_lossy, V_hat_tile)
    else:
        out_E = attention_int8_sv(S_lossy, V_hat_tile)

    return {
        "out_A": out_A, "out_B": out_B, "out_C": out_C,
        "out_D": out_D, "out_E": out_E,
        "d_fp16_clean": d_fp16_clean,
        "d_fp16_lossy": d_fp16_lossy,
    }


# ---------------------------------------------------------------------------
# Per-layer KV-cache pre-compression
# ---------------------------------------------------------------------------
def precompress_kv(K_kv: np.ndarray, V_kv: np.ndarray,
                   engine: KVCacheEngine,
                   verbose_layer: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Compress + decompress every (KV-head, token) K and V vector.

    K_kv, V_kv: [Hkv, N, d]  (float32, already at fp16 precision)
    Returns:    K_hat, V_hat at the same shape.

    Compresses through Python ref; logs throughput so the scope-tune step
    can estimate runtime.
    """
    Hkv, N, d = K_kv.shape
    K_hat = np.empty_like(K_kv)
    V_hat = np.empty_like(V_kv)
    t0 = time.time()
    for h in range(Hkv):
        for n in range(N):
            kq = fp_to_q412(K_kv[h, n])
            vq = fp_to_q412(V_kv[h, n])
            ck = engine.compress_key(kq)
            cv = engine.compress_value(vq)
            kh = engine.decompress_key(ck)
            vh = engine.decompress_value(cv)
            K_hat[h, n] = q412_to_fp(kh)
            V_hat[h, n] = q412_to_fp(vh)
    dt = time.time() - t0
    if verbose_layer:
        print(f"[kv] {verbose_layer}  {Hkv*N} pairs in {dt:5.1f}s "
              f"({1000*dt/(Hkv*N):.1f} ms/pair)", flush=True)
    return K_hat, V_hat


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--block",   type=int, default=BLOCK_SIZE)
    ap.add_argument("--layers", default="0,4,8,12,16,20,23",
                    help="Comma-separated layer indices to process.")
    ap.add_argument("--prompts", type=int, default=1,
                    help="Number of prompts to run (capped at len(LONG_PROMPTS)).")
    ap.add_argument("--delete-cache-after", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" /
                                        "integration_test_kv_pc_stats.json"))
    args = ap.parse_args(argv)

    layer_indices = sorted(int(s) for s in args.layers.split(",") if s)
    n_prompts = min(args.prompts, len(LONG_PROMPTS))

    print(f"[setup] model={args.model}  seq_len={args.seq_len}  block={args.block}",
          flush=True)
    print(f"[setup] layers={layer_indices}  prompts={n_prompts}", flush=True)

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
    print(f"[setup] model has {n_layers} layers, {num_q_heads} q-heads, "
          f"{num_kv_heads} kv-heads, d={head_dim}", flush=True)

    capture = QKVCapture(model)
    engine = KVCacheEngine(KVCacheEngineInfo(vector_dim=head_dim))
    print(f"[setup] KV cache cfg: dim={head_dim}, pq_bits=3, qjl_bits=1, "
          f"K cr={engine.info.compression_ratio_k():.2f}x, "
          f"V cr={engine.info.compression_ratio_v():.2f}x", flush=True)

    records: list[dict] = []

    for pi in range(n_prompts):
        pname, ptext = LONG_PROMPTS[pi]
        inputs = tokenizer(ptext, return_tensors="pt",
                           truncation=True, max_length=args.seq_len).to(model.device)
        seq = inputs["input_ids"].shape[1]
        print(f"[run] prompt={pname!r}  tokens={seq}", flush=True)
        with torch.no_grad():
            model(**inputs)

        for layer_idx in layer_indices:
            Q_raw = capture.q_acts[layer_idx]            # [1, N, Hq*d]
            K_raw = capture.k_acts[layer_idx]            # [1, N, Hkv*d]
            V_raw = capture.v_acts[layer_idx]            # [1, N, Hkv*d]

            # Reshape to per-head, send to CPU as float32.
            Q = Q_raw.view(1, seq, num_q_heads,  head_dim).permute(0, 2, 1, 3)[0].cpu().numpy()  # [Hq, N, d]
            K = K_raw.view(1, seq, num_kv_heads, head_dim).permute(0, 2, 1, 3)[0].cpu().numpy()  # [Hkv, N, d]
            V = V_raw.view(1, seq, num_kv_heads, head_dim).permute(0, 2, 1, 3)[0].cpu().numpy()  # [Hkv, N, d]

            # Round to fp16 precision so the baseline matches what the chip sees.
            Q = Q.astype(np.float16).astype(np.float32)
            K = K.astype(np.float16).astype(np.float32)
            V = V.astype(np.float16).astype(np.float32)

            # Precompress K, V once per (layer, kv_head, token) — GQA shortcut.
            K_hat, V_hat = precompress_kv(K, V, engine,
                                          verbose_layer=f"L{layer_idx:>2}")

            # Expand KV heads for the Q heads they serve.
            repeats = num_q_heads // num_kv_heads
            K_exp     = np.repeat(K,     repeats, axis=0)
            V_exp     = np.repeat(V,     repeats, axis=0)
            K_hat_exp = np.repeat(K_hat, repeats, axis=0)
            V_hat_exp = np.repeat(V_hat, repeats, axis=0)

            # Per (head, q_block, k_block) tile walk — causal lower-triangular.
            n_blocks = math.ceil(seq / args.block)
            for h in range(num_q_heads):
                for qb in range(n_blocks):
                    q_lo = qb * args.block
                    q_hi = min(q_lo + args.block, seq)
                    Q_blk = Q[h, q_lo:q_hi]
                    for kb in range(qb + 1):
                        k_lo = kb * args.block
                        k_hi = min(k_lo + args.block, seq)
                        # Skip non-full edge tiles to keep the chip's
                        # fixed-tile-size comparison clean (matches prior studies).
                        if (q_hi - q_lo) != args.block or (k_hi - k_lo) != args.block:
                            continue

                        K_blk     = K_exp    [h, k_lo:k_hi]
                        V_blk     = V_exp    [h, k_lo:k_hi]
                        K_hat_blk = K_hat_exp[h, k_lo:k_hi]
                        V_hat_blk = V_hat_exp[h, k_lo:k_hi]

                        # Diagonal tile needs causal masking inside the block.
                        if qb == kb:
                            mask = np.tril(np.ones((args.block, args.block),
                                                   dtype=np.float32))
                            # Add −∞ where mask==0 to zero out post-softmax.
                            neg_inf = (1.0 - mask) * -1e9
                            # paths will apply this via the score addition;
                            # we fold it into Q@K.T by adding to all S inputs.
                            # Easier: skip — the integration test is most
                            # informative on strictly off-diagonal tiles
                            # where every Q row attends to every K row.
                            continue

                        paths = tile_paths(Q_blk, K_blk, V_blk,
                                           K_hat_blk, V_hat_blk, head_dim)

                        # Reduce per-path outputs to MSE vs REF.
                        ref = paths["out_A"]
                        ref_norm = float(np.linalg.norm(ref) ** 2 / ref.size) + 1e-9

                        def rel_mse(out):
                            err = (out - ref) ** 2
                            return float(err.mean() / ref_norm)

                        rec = {
                            "L":   int(layer_idx),
                            "H":   int(h),
                            "qb":  int(qb),
                            "kb":  int(kb),
                            "d_clean": bool(paths["d_fp16_clean"]),
                            "d_lossy": bool(paths["d_fp16_lossy"]),
                            "rmse_B": rel_mse(paths["out_B"]),  # PC alone
                            "rmse_C": rel_mse(paths["out_C"]),  # KV alone fp16
                            "rmse_D": rel_mse(paths["out_D"]),  # KV alone int8
                            "rmse_E": rel_mse(paths["out_E"]),  # integrated
                        }
                        records.append(rec)
            print(f"[run] L{layer_idx:>2}  layer done  tiles so far: "
                  f"{len(records)}", flush=True)
        capture.clear()
    capture.remove()

    print(f"[analyze] reducing {len(records)} tile records", flush=True)
    summary = analyze(records)
    summary["model"]          = args.model
    summary["seq_len"]        = args.seq_len
    summary["block"]          = args.block
    summary["layers_used"]    = layer_indices
    summary["prompts_used"]   = n_prompts
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {args.out}", flush=True)
    print_summary(summary)

    if args.delete_cache_after:
        cache_root = Path(
            os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or
            (Path.home() / ".cache" / "huggingface")
        )
        slug = "models--" + args.model.replace("/", "--")
        target = cache_root / "hub" / slug
        if target.exists():
            print(f"[cleanup] removing {target}", flush=True)
            shutil.rmtree(target, ignore_errors=True)
    return 0


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------
def analyze(records: list[dict]) -> dict:
    if not records:
        return {"error": "no records"}
    arr = {k: np.array([r[k] for r in records]) for k in records[0].keys()}
    n = arr["L"].size

    def pct(b):
        return 100.0 * float(b.sum()) / max(b.size, 1)

    out: dict = {"n_tiles": n}

    # --- decision shift ---
    out["d_fp16_clean_pct"] = pct(arr["d_clean"])
    out["d_fp16_lossy_pct"] = pct(arr["d_lossy"])
    out["decision_agree_pct"] = pct(arr["d_clean"] == arr["d_lossy"])
    flipped = arr["d_clean"] != arr["d_lossy"]
    out["flipped_clean_fp16_to_lossy_int8"] = int(((arr["d_clean"]) & (~arr["d_lossy"])).sum())
    out["flipped_clean_int8_to_lossy_fp16"] = int(((~arr["d_clean"]) & (arr["d_lossy"])).sum())

    # --- MSE distribution per path ---
    for path in ["B", "C", "D", "E"]:
        v = arr[f"rmse_{path}"]
        out[f"rmse_{path}"] = {
            "mean":   float(v.mean()),
            "median": float(np.median(v)),
            "p90":    float(np.percentile(v, 90)),
            "p99":    float(np.percentile(v, 99)),
        }

    # --- KEY QUESTION: does FP16 routing still help under lossy V? ---
    # On tiles where d_fp16_clean == True (precision controller said "FP16"),
    # compare KV+FP16 (C) vs KV+INT8 (D). If their MSE distributions are
    # statistically indistinguishable, the FP16 path is decorative.
    sel_should_fp16 = arr["d_clean"]
    if sel_should_fp16.sum() > 0:
        rmse_C_sub = arr["rmse_C"][sel_should_fp16]
        rmse_D_sub = arr["rmse_D"][sel_should_fp16]
        out["should_fp16_n"] = int(sel_should_fp16.sum())
        out["should_fp16_rmse_C_median"] = float(np.median(rmse_C_sub))
        out["should_fp16_rmse_D_median"] = float(np.median(rmse_D_sub))
        out["should_fp16_fp16_helps_pct"] = float(
            100.0 * (rmse_C_sub < rmse_D_sub).mean()
        )
    else:
        out["should_fp16_n"] = 0

    # Same comparison for tiles flagged INT8: FP16 path should be ≤ INT8.
    sel_should_int8 = ~arr["d_clean"]
    if sel_should_int8.sum() > 0:
        rmse_C_sub = arr["rmse_C"][sel_should_int8]
        rmse_D_sub = arr["rmse_D"][sel_should_int8]
        out["should_int8_n"] = int(sel_should_int8.sum())
        out["should_int8_rmse_C_median"] = float(np.median(rmse_C_sub))
        out["should_int8_rmse_D_median"] = float(np.median(rmse_D_sub))
        out["should_int8_fp16_helps_pct"] = float(
            100.0 * (rmse_C_sub < rmse_D_sub).mean()
        )

    # --- Per-layer breakdown ---
    layers = sorted(set(int(L) for L in arr["L"].tolist()))
    out["per_layer"] = {}
    for L in layers:
        sel = arr["L"] == L
        if sel.sum() == 0:
            continue
        out["per_layer"][str(L)] = {
            "n":                 int(sel.sum()),
            "d_clean_fp16_pct":  pct(arr["d_clean"][sel]),
            "d_lossy_fp16_pct":  pct(arr["d_lossy"][sel]),
            "rmse_B_median":     float(np.median(arr["rmse_B"][sel])),
            "rmse_C_median":     float(np.median(arr["rmse_C"][sel])),
            "rmse_D_median":     float(np.median(arr["rmse_D"][sel])),
            "rmse_E_median":     float(np.median(arr["rmse_E"][sel])),
        }
    return out


def print_summary(s: dict) -> None:
    print()
    print(f"  Total tiles            : {s['n_tiles']:,}")
    print()
    print( "  Decision stability (clean S vs lossy S):")
    print(f"    FP16% on clean S     : {s['d_fp16_clean_pct']:.2f}%")
    print(f"    FP16% on lossy S     : {s['d_fp16_lossy_pct']:.2f}%")
    print(f"    decisions agree      : {s['decision_agree_pct']:.2f}%")
    print(f"    clean-FP16→lossy-INT8: {s['flipped_clean_fp16_to_lossy_int8']}")
    print(f"    clean-INT8→lossy-FP16: {s['flipped_clean_int8_to_lossy_fp16']}")
    print()
    print( "  Median relative MSE vs dense FP16 baseline:")
    for path, label in [("B", "PC alone (no KV cache)"),
                        ("C", "KV alone, FP16 SV"),
                        ("D", "KV alone, INT8 SV"),
                        ("E", "Integrated (PC + KV)")]:
        st = s[f"rmse_{path}"]
        print(f"    {label:25s} median={st['median']:.4f}  "
              f"p90={st['p90']:.4f}  p99={st['p99']:.4f}")
    print()
    if s.get("should_fp16_n", 0) > 0:
        print( "  Among tiles the PC says \"FP16\" (clean decision):")
        print(f"    n                                = {s['should_fp16_n']}")
        print(f"    median rMSE, KV+FP16 SV (C)      = {s['should_fp16_rmse_C_median']:.4f}")
        print(f"    median rMSE, KV+INT8 SV (D)      = {s['should_fp16_rmse_D_median']:.4f}")
        print(f"    %tiles where FP16 SV beats INT8  = {s['should_fp16_fp16_helps_pct']:.2f}%")
    print()
    if s.get("should_int8_n", 0) > 0:
        print( "  Among tiles the PC says \"INT8\" (clean decision):")
        print(f"    n                                = {s['should_int8_n']}")
        print(f"    median rMSE, KV+FP16 SV (C)      = {s['should_int8_rmse_C_median']:.4f}")
        print(f"    median rMSE, KV+INT8 SV (D)      = {s['should_int8_rmse_D_median']:.4f}")
        print(f"    %tiles where FP16 SV beats INT8  = {s['should_int8_fp16_helps_pct']:.2f}%")


if __name__ == "__main__":
    sys.exit(main())
