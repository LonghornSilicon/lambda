#!/usr/bin/env python3
"""CI accuracy gate -- smallest meaningful REAL-Qwen eval for the ACU + ChannelQuant-KV path.

Runs a real Qwen2-0.5B forward on a small, seeded slice of HellaSwag twice -- once
with the precision path OFF (fp16-semantics baseline) and once with the combined
ChannelQuant-KV + APA precision-controller path ON -- and PASSES iff the gated
path's acc_norm degradation vs the baseline is within tolerance. This is the
CPU/CI-sized sibling of analysis/cq_apa_e2e.py; it is a PAIRED comparison (same
items, same seed, same model dtype), so the measured delta isolates the codec/ACU
effect -- exactly the regression the analysis scripts used to catch by hand.

Reuse (this file adds NO new model math -- everything below is imported from
analysis/cq_apa_e2e.py):
  * Importing cq_apa_e2e runs, at import time,
        AttentionInterface.register("cq_apa", cq_apa_attention)
    so a model built with attn_implementation="cq_apa" dispatches every attention
    call to the authoritative logic verified at n=1000 (ChannelQuant codec + APA
    INT8/FP16 S.V routing).
  * We toggle the module-global CFG dict that cq_apa_attention reads live -- the
    same knob cq_apa_e2e.main() sweeps -- between the baseline and gated configs.
  * The eval is the same simple_evaluate(tasks=["hellaswag"]) -> acc_norm,none
    call main() uses, only with limit=n on CPU so it is deterministic and cheap.

We deliberately do NOT reuse cq_apa_e2e.build_model(): it hardcodes .cuda() +
float16. On the CPU runner we build in float32 (Qwen2's Linear/MLP addmm has no
half-precision CPU kernel). The codec still quantizes K/V inside the hook
regardless of weight dtype, and the gate is a paired delta, so model dtype
cancels; only the absolute acc_norm shifts slightly from the committed numbers.

Contract with the CI wrapper (block-ci.yml gate 9):
  * prints EXACTLY the line  ALL TESTS PASSED  on success (plus the measured delta),
  * prints a  FAILED (delta=... > tol=...)  line and sys.exit(1) on regression,
  * never prints FAILED / MISMATCH / OUT OF TOL (case-insensitive) on the pass path.

Tolerance (see committed analysis/cq_apa_qwen05b_n1000.json): the gated cq4+apa
config degrades only 0.003 acc_norm at n=1000. On a PAIRED small-n run the healthy
delta sits ~0.00 with +/-1..2-item jitter; we gate at tol=0.06, well above jitter
and an order of magnitude below a real regression (broken codec -> acc toward 0.25).
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
import cq_apa_e2e as e2e   # noqa: E402  -- side effect: registers "cq_apa"


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
    """HFLM around the "cq_apa" attention, on CPU/fp32. batch_size=1: the APA
    router keys off isfinite(scores), and a padded batch would count pad columns
    as valid keys and perturb routing. Keep it pinned."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="cq_apa"
    ).to("cpu").eval()
    batch = int(os.environ.get("QWEN_CI_BATCH", "1"))
    return HFLM(pretrained=model, tokenizer=tok, batch_size=batch, device="cpu")


def eval_acc_norm(lm, n, seed, tier, apa):
    """acc_norm on the first n HellaSwag docs with CFG={tier,apa} (deterministic)."""
    import lm_eval
    e2e.CFG["tier"], e2e.CFG["apa"] = tier, apa
    e2e.STATS["int8_tiles"] = e2e.STATS["fp16_tiles"] = 0
    seed_everything(seed)
    kw = dict(model=lm, tasks=["hellaswag"], limit=n, bootstrap_iters=0)
    try:  # newer lm_eval accepts explicit seeds; older signatures ignore them
        out = lm_eval.simple_evaluate(random_seed=seed, numpy_random_seed=seed,
                                      torch_random_seed=seed, fewshot_random_seed=seed, **kw)
    except TypeError:
        out = lm_eval.simple_evaluate(**kw)
    acc = float(out["results"]["hellaswag"]["acc_norm,none"])
    tot = e2e.STATS["int8_tiles"] + e2e.STATS["fp16_tiles"]
    int8_frac = (e2e.STATS["int8_tiles"] / tot) if tot else None
    return acc, int8_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QWEN_CI_MODEL", "Qwen/Qwen2-0.5B"))
    ap.add_argument("--n", type=int, default=int(os.environ.get("QWEN_CI_N", "50")))
    ap.add_argument("--tol", type=float, default=float(os.environ.get("QWEN_CI_TOL", "0.06")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("QWEN_CI_SEED", "0")))
    args = ap.parse_args()

    quiet_logs()
    seed_everything(args.seed)
    torch.set_grad_enabled(False)

    print(f"[qwen_ci_gate] model={args.model} n={args.n} tol={args.tol} "
          f"seed={args.seed} device=cpu dtype=fp32 path=ChannelQuant-KV(cq4)+APA")

    lm = build_lm(args.model)

    base, _ = eval_acc_norm(lm, args.n, args.seed, tier="off", apa=False)
    gated, int8_frac = eval_acc_norm(lm, args.n, args.seed, tier="cq4", apa=True)

    delta = gated - base            # signed: negative = degraded
    deg = base - gated              # positive magnitude of degradation

    print(f"[qwen_ci_gate] baseline (off)       acc_norm={base:.4f}")
    print(f"[qwen_ci_gate] gated (cq4+apa)      acc_norm={gated:.4f}  int8_frac={int8_frac}")
    print(f"[qwen_ci_gate] delta(cq4+apa - off) = {delta:+.4f}   tol = {args.tol:.4f}")
    print(f"[qwen_ci_gate] reference: committed cq_apa_qwen05b_n1000.json cq4+apa delta = -0.0030")

    if base < 0.35:
        print(f"FAILED (baseline acc_norm={base:.4f} < 0.35 -- model/dataset load looks broken)")
        return 1

    if deg <= args.tol:
        print(f"[qwen_ci_gate] degradation within tol (deg={deg:+.4f} <= {args.tol:.4f})")
        print("ALL TESTS PASSED")
        return 0

    print(f"FAILED (delta={deg:.4f} > tol={args.tol:.4f})")
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
