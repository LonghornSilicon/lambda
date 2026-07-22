#!/usr/bin/env python3
"""CI accuracy gate — real Qwen2, TIU eviction path (block 3 + full stack).

Runs the smallest meaningful REAL-Qwen accuracy eval that exercises the TIU
eviction datapath and gates the accuracy degradation against a defensible
tolerance. Deterministic, CPU-only, small-n — designed to run inside the
reusable `qwen-model-test` CI job (`python analysis/qwen_ci_gate.py`).

Contract with the CI wrapper (block-ci.yml gate 9):
  * prints EXACTLY `ALL TESTS PASSED` on success (plus the measured delta),
  * on failure prints a `FAILED (...)` line and `sys.exit(1)`,
  * never prints FAILED / MISMATCH / OUT OF TOL on the passing path.

Reuse: this file adds NO new model math. It imports full_stack_integration.py
and reuses, unchanged, its registered "full_stack" attention (TIU keep/evict +
KVCE ChannelQuant + ACU/APA) and its run_cfg / CFG grid.

Tolerance: committed analysis/full_stack_qwen05b_n1000.json records the ALL-3
config at Δ = -0.033 vs the fp16 baseline (acc 0.456 vs 0.489). We gate the
ALL-3 config at Δ >= -0.06, ~0.027 absolute margin over the worst committed
ALL-3 delta.
"""
import argparse
import os
import random
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # force CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_stack_integration as fs   # registers the "full_stack" attention on import  # noqa: E402


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


def build_lm(model_id):
    """HFLM around the custom full_stack attention, CPU/fp32 (fp16 addmm is not
    implemented on CPU). The gated quantity is a DELTA between two configs under
    the same model dtype, so it is robust to the fp16->fp32 change; only the
    absolute acc_norm shifts slightly from the committed GPU-fp16 numbers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="full_stack"
    ).to("cpu").eval()
    batch = int(os.environ.get("QWEN_CI_BATCH", "8"))
    return HFLM(pretrained=model, tokenizer=tok, batch_size=batch, device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN_CI_MODEL", "Qwen/Qwen2-0.5B"))
    ap.add_argument("--n", type=int, default=int(os.environ.get("QWEN_CI_N", "100")))
    ap.add_argument("--tol", type=float, default=float(os.environ.get("QWEN_CI_TOL", "0.06")))
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--recent_ratio", type=float, default=0.5)
    args = ap.parse_args()

    seed_everything(0)
    fs.CFG["frac"] = args.frac
    fs.CFG["recent_ratio"] = args.recent_ratio

    print(f"[qwen_ci_gate] model={args.model} n={args.n} tol={args.tol} "
          f"frac={args.frac} recent_ratio={args.recent_ratio} device=cpu dtype=fp32")

    lm = build_lm(args.model)

    base = fs.run_cfg(lm, "baseline", args.n)
    all3 = fs.run_cfg(lm, "ALL3", args.n, tiu=True, kvce="cq4+", apa=True)

    base_acc = base["acc_norm"]
    all3_acc = all3["acc_norm"]
    delta = all3_acc - base_acc

    print(f"[qwen_ci_gate] baseline acc_norm={base_acc:.4f}")
    print(f"[qwen_ci_gate] ALL3     acc_norm={all3_acc:.4f}  kept={all3['kept']}  int8={all3['int8']}")
    print(f"[qwen_ci_gate] ALL3 delta vs baseline = {delta:+.4f}  (allowed >= -{args.tol:.3f})")
    print(f"[qwen_ci_gate] reference: committed full_stack_qwen05b_n1000.json ALL3 delta = -0.0330")

    if delta >= -args.tol:
        print(f"measured ALL3 delta={delta:+.4f} within tolerance {args.tol:.3f}")
        print("ALL TESTS PASSED")
        return 0

    print(f"FAILED (delta={delta:+.4f} < -{args.tol:.3f})")
    return 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:   # any eval/setup error is a hard gate failure
        import traceback
        traceback.print_exc()
        print(f"FAILED (exception: {type(e).__name__}: {e})")
        rc = 1
    sys.exit(rc)
