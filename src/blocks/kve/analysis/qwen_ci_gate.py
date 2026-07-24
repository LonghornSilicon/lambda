#!/usr/bin/env python3
"""CI accuracy gate -- smallest meaningful REAL-Qwen eval that exercises the KV codec.

Runs the EXACT Phase-1 WHT-INT3 value codec (raw-forward WHT -> INT3 srint -> inverse
WHT -> x2^-k; keys held CQ-4+) inside a real Qwen2-0.5B forward on a small, seeded
slice of HellaSwag, and PASSES iff the codec's acc_norm degradation vs the fp16
baseline is within tolerance. This is the CPU/CI-sized sibling of
analysis/wht_ref_accuracy.py. It is a PAIRED comparison (same items, same seed, same
model dtype), so the measured delta isolates the codec effect -- exactly the
regression the analysis scripts used to catch by hand.

Reuse (this file adds NO new model math):
  * The codec AND its in-model application are imported wholesale from
    analysis/wht_ref_accuracy.py. Importing that module runs, at import time,
        AttentionInterface.register("whtref", attn)
    so a model built with attn_implementation="whtref" dispatches every attention
    call to `attn`, which -- when CFG["mode"] != "fp16" -- calls `q_wht3` on V and
    `q_keys_per_channel_hw` (imported there from channelquant_hw.py) on K. We only
    flip that module-level CFG["mode"] between "fp16" (baseline, no KV quant) and
    "q" (codec on) -- exactly the toggle wht_ref_accuracy.main() sweeps.
  * The eval is the same simple_evaluate(tasks=["hellaswag"]) -> acc_norm,none call
    main() uses, only with a small limit on CPU so it is deterministic and cheap.

We deliberately do NOT reuse wht_ref_accuracy.main(): it hardcodes .cuda() + float16
and batch_size=16. On the CPU runner we build in float32 (Qwen2's Linear/MLP addmm
has no half-precision CPU kernel) and pin batch_size=1 (see build_lm). The codec
still quantizes K/V inside the hook regardless of weight dtype, and the gate is a
paired delta, so model dtype cancels; only the absolute acc_norm shifts slightly.

Contract with the CI wrapper (block-ci.yml gate 9):
  * prints EXACTLY the line  ALL TESTS PASSED  on success (plus the measured delta),
  * prints a  FAILED (delta=... > tol=...)  line and sys.exit(1) on regression,
  * never prints FAILED / MISMATCH / OUT OF TOL (case-insensitive) on the pass path.

Tolerance / gate direction (see committed analysis/wht_ref_accuracy_qwen05b.json):
  The gate is ONE-SIDED: it fails iff the codec DEGRADES acc_norm by more than tol
  (delta < -tol). A codec that ties or beats the baseline is never a regression, so
  we do not fail on positive small-n jitter -- at n=50 one flipped HellaSwag item
  moves acc_norm by 0.02, and a +1..2-item lucky swing is pure sampling noise.
  At n=1000 on Qwen2-0.5B this codec degrades acc_norm by 0.007 (0.489->0.482); the
  silicon-faithful cq4+ path by 0.009; the worst random-WHT seed by 0.019. We gate
  at tol=0.05: ~2.5x the worst committed degradation, absorbing ~2 adverse items of
  n=50 noise, while a broken codec (acc collapse toward chance ~0.25, delta ~ -0.2)
  trips it decisively.
"""
import argparse
import os
import random
import sys
import warnings

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # force CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DATASETS_VERBOSITY", "error")
os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

import numpy as np       # noqa: E402
import torch             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wht_ref_accuracy as wht   # noqa: E402  -- side effect: registers "whtref"


def seed_everything(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.set_num_threads(int(os.environ.get("QWEN_CI_THREADS", "4")))


def quiet_logs():
    import logging
    logging.disable(logging.WARNING)
    for name in ("transformers", "datasets", "lm_eval", "huggingface_hub", "accelerate"):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass


def build_lm(model_id):
    """HFLM around the "whtref" attention, CPU/fp32, batch_size=1. The `attn` hook
    applies only a causal mask and ignores the HF padding mask, so a right-padded
    batch would let query rows attend to pad keys. batch_size=1 has no intra-batch
    padding. Keep it pinned unless the hook starts honoring attention_mask."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="whtref"
    ).to("cpu").eval()
    batch = int(os.environ.get("QWEN_CI_BATCH", "1"))
    return HFLM(pretrained=model, tokenizer=tok, batch_size=batch, device="cpu")


def eval_acc_norm(lm, n, seed, mode):
    """acc_norm on the first n HellaSwag docs with CFG["mode"]=mode (deterministic)."""
    import lm_eval
    wht.CFG["mode"] = mode
    seed_everything(seed)
    kw = dict(model=lm, tasks=["hellaswag"], limit=n, bootstrap_iters=0)
    try:  # newer lm_eval accepts explicit seeds; older signatures ignore them
        out = lm_eval.simple_evaluate(random_seed=seed, numpy_random_seed=seed,
                                      torch_random_seed=seed, fewshot_random_seed=seed, **kw)
    except TypeError:
        out = lm_eval.simple_evaluate(**kw)
    return float(out["results"]["hellaswag"]["acc_norm,none"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN_CI_MODEL", "Qwen/Qwen2-0.5B"))
    ap.add_argument("--n", type=int, default=int(os.environ.get("QWEN_CI_N", "50")))
    ap.add_argument("--tol", type=float, default=float(os.environ.get("QWEN_CI_TOL", "0.05")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("QWEN_CI_SEED", "0")))
    args = ap.parse_args()

    quiet_logs()
    seed_everything(args.seed)
    torch.set_grad_enabled(False)

    print(f"[qwen_ci_gate] model={args.model} n={args.n} tol={args.tol} "
          f"seed={args.seed} device=cpu dtype=fp32 codec=WHT-INT3(V)+CQ4+(K)")

    lm = build_lm(args.model)

    base = eval_acc_norm(lm, args.n, args.seed, "fp16")   # no KV quant
    codec = eval_acc_norm(lm, args.n, args.seed, "q")     # WHT-INT3 values + CQ-4+ keys
    delta = codec - base

    print(f"[qwen_ci_gate] fp16 baseline    acc_norm={base:.4f}")
    print(f"[qwen_ci_gate] WHT-INT3 codec   acc_norm={codec:.4f}")
    print(f"[qwen_ci_gate] delta(codec - fp16) = {delta:+.4f}   allowed degradation tol = {args.tol:.4f}")
    print(f"[qwen_ci_gate] reference: committed wht_ref_accuracy_qwen05b.json delta = -0.0070 (n=1000)")

    # One-sided gate: only a degradation worse than tol is a regression.
    if delta >= -args.tol:
        print(f"[qwen_ci_gate] degradation within tol (delta {delta:+.4f} >= -{args.tol:.4f})")
        print("ALL TESTS PASSED")
        return 0

    print(f"FAILED (delta={delta:+.4f} > tol={args.tol:.4f} degradation)")
    return 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:   # any eval/setup error is a hard gate failure, not a hang
        import traceback
        traceback.print_exc()
        print(f"FAILED (exception: {type(e).__name__}: {e})")
        rc = 1
    sys.exit(rc)
