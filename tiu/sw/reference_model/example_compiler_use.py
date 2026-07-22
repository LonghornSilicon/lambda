"""End-to-end example of using the Token Importance Unit reference model
from a silicon-agnostic compiler backend's perspective.

The TIU silicon core is deliberately small: accumulate attention mass, argmin to
pick an eviction victim, and a per-slot keep/demote comparator. The H2O policy
that decides *when* to load and evict tokens -- the recent-window protection and
the KV-budget accounting -- is a thin control-layer wrapper the compiler emits
around the core. This script shows that wrapper at three levels of sophistication,
all hitting the same Python model that is bit-exact with the RTL (and eventually
the chip):

    Level A -- budget sizing:   pick the cache size C and recent window L from the
               validated gold config, and map them onto the hardware's N_SLOTS.
    Level B -- eviction schedule: stream a token's attention over the cache, drive
               LOAD / ACC / EVICT, and collect the eviction schedule the backend
               would emit to the KV cache engine.
    Level C -- keep/demote tiers: calibrate the tier threshold from accumulated
               mass and read per-token keep(CQ-8)/demote(CQ-4) tiers off the core.

If you are a compiler engineer evaluating an integration, this is the fastest way
to see exactly what surface area you are targeting.

Run:    python3 sw/reference_model/example_compiler_use.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tiu_ref import TokenImportanceUnit, TokenImportanceUnitInfo  # noqa: E402


# ---------------------------------------------------------------------------
# The gold config, straight from docs/findings/h2o-analysis.md &
# h2o-deep-analysis.md: 25% KV budget with a 50/50 split between the recent
# local window and the heavy-hitter set is near-lossless (-0.006 acc_norm /
# +1.0% perplexity on long context). SCORE_WIDTH = 8 is loss-free.
# ---------------------------------------------------------------------------
GOLD_BUDGET_FRAC = 0.25
GOLD_RECENT_RATIO = 0.50


# ===========================================================================
# Level A -- budget sizing
# ===========================================================================
def level_a_size_budget(context_len: int, info: TokenImportanceUnitInfo):
    """Turn a context length into a cache size C and recent window L.

    A compiler sizes the on-die KV budget once per (model, deployment). The gold
    config keeps 25% of the context; half of that budget is a recent local
    window, half is reserved for heavy hitters ranked by the TIU. The hardware
    exposes N_SLOTS physical slots, so the effective C is clamped to N_SLOTS.
    """
    c_ideal = max(1, round(GOLD_BUDGET_FRAC * context_len))
    c = min(c_ideal, info.n_slots)          # clamp to the physical slot count
    recent = max(1, round(GOLD_RECENT_RATIO * c))
    heavy = c - recent
    return {"context_len": context_len, "c_ideal": c_ideal, "c": c,
            "recent_window": recent, "heavy_slots": heavy}


# ===========================================================================
# Level B -- eviction schedule
# ===========================================================================
class H2OScheduler:
    """The control-layer wrapper a compiler emits around the TIU core.

    Streams tokens into a fixed set of N_SLOTS slots. When the cache is full and
    a new token arrives, it asks the TIU for the minimum-mass victim (the core's
    serialized argmin), evicts it, and reuses the freed slot. Every query's
    post-softmax attention over the currently cached tokens is fed back as ACC
    weights -- exactly the op stream the RTL replay testbench consumes.
    """

    def __init__(self, info: TokenImportanceUnitInfo) -> None:
        self.info = info
        self.tiu = TokenImportanceUnit(info)
        self.slot_token = [-1] * info.n_slots      # which token id sits in each slot
        self.schedule = []                         # (query_token, evicted_token, slot)

    def step(self, token_id: int, attn_over_cache: dict) -> None:
        """Admit `token_id`; `attn_over_cache` maps slot -> INT weight for this step."""
        info = self.info
        # 1. find a free slot, else evict the TIU-chosen victim and reuse it.
        free = next((k for k in range(info.n_slots) if not self.tiu.valid[k]), None)
        if free is None:
            victim_slot = self.tiu.evict()
            self.schedule.append((token_id, self.slot_token[victim_slot], victim_slot))
            free = victim_slot
        # 2. install the new token.
        self.tiu.load(free)
        self.slot_token[free] = token_id
        # 3. accumulate this query's attention mass onto every cached token.
        for slot, weight in attn_over_cache.items():
            if self.tiu.valid[slot] and weight > 0:
                self.tiu.acc(slot, weight)


def level_b_eviction_schedule(info: TokenImportanceUnitInfo, num_tokens: int):
    """Stream a synthetic 'heavy-hitter' workload and collect the eviction schedule.

    A few early tokens are heavy hitters (they keep receiving attention); the rest
    are background. The TIU should protect the heavy hitters and evict background
    tokens -- which is exactly what the compiler needs to schedule KV drops.
    """
    rng = random.Random(7)
    sched = H2OScheduler(info)
    heavy_hitters = {0, 3}                       # token ids that stay important
    wmax = info.weight_max

    for t in range(num_tokens):
        # Build this query's attention over currently cached slots.
        attn = {}
        for slot in range(info.n_slots):
            tok = sched.slot_token[slot]
            if tok < 0 or not sched.tiu.valid[slot]:
                continue
            if tok in heavy_hitters:
                attn[slot] = rng.randint(wmax // 2, wmax)      # large, sustained mass
            else:
                attn[slot] = rng.randint(0, wmax // 16)        # trickle
        sched.step(t, attn)

    return sched


# ===========================================================================
# Level C -- keep/demote tiers
# ===========================================================================
def level_c_keep_demote(sched: H2OScheduler):
    """Calibrate the tier threshold and read per-token keep/demote tiers.

    The software layer sets `tier_threshold` from a calibration pass (the docs
    suggest a percentile of accumulated mass). Here we use the median of the
    valid slots' scores. tier() then returns, per slot, whether the KV cache
    engine should keep the value at CQ-8 (heavy hitter) or demote it to CQ-4.
    """
    scores = [s for s, v in zip(sched.tiu.score, sched.tiu.valid) if v]
    if not scores:
        return {"threshold": 0, "keep": [], "demote": []}
    scores_sorted = sorted(scores)
    threshold = scores_sorted[len(scores_sorted) // 2]        # median
    keep_flags = sched.tiu.tier(threshold)

    keep, demote = [], []
    for slot, flag in enumerate(keep_flags):
        if not sched.tiu.valid[slot]:
            continue
        (keep if flag else demote).append(sched.slot_token[slot])
    return {"threshold": threshold, "keep": sorted(keep), "demote": sorted(demote)}


# ===========================================================================
def main() -> int:
    info = TokenImportanceUnitInfo()
    print(f"TIU config: N_SLOTS={info.n_slots}, SCORE_WIDTH={info.score_width}, "
          f"WEIGHT_WIDTH={info.weight_width}\n")

    # Level A ---------------------------------------------------------------
    print("Level A -- KV budget sizing (gold config: 25% budget, 50/50 split):")
    for ctx in (16, 32, 512):
        b = level_a_size_budget(ctx, info)
        print(f"  context={b['context_len']:>4}: C_ideal={b['c_ideal']:>3} "
              f"-> C={b['c']} slots (recent={b['recent_window']}, "
              f"heavy={b['heavy_slots']})")
    print()

    # Level B ---------------------------------------------------------------
    NUM_TOKENS = 40
    sched = level_b_eviction_schedule(info, NUM_TOKENS)
    print(f"Level B -- eviction schedule over {NUM_TOKENS} tokens "
          f"(heavy hitters = tokens 0, 3):")
    print(f"  {len(sched.schedule)} evictions scheduled; "
          f"first 8: {[(q, ev) for q, ev, _ in sched.schedule[:8]]}")
    evicted = {ev for _, ev, _ in sched.schedule}
    protected = [hh for hh in (0, 3) if hh not in evicted]
    print(f"  heavy hitters never evicted: {protected}  "
          f"(evicted token ids: {sorted(evicted)[:10]}...)")
    print()

    # Level C ---------------------------------------------------------------
    tiers = level_c_keep_demote(sched)
    print("Level C -- per-token keep/demote tiers (threshold from median mass):")
    print(f"  tier_threshold = {tiers['threshold']}")
    print(f"  KEEP  (value @ CQ-8, heavy hitters): tokens {tiers['keep']}")
    print(f"  DEMOTE(value @ CQ-4):                tokens {tiers['demote']}")
    print()

    # Consistency check: the protected heavy hitters must be in the KEEP tier.
    if not all(hh in tiers["keep"] for hh in protected):
        print("ERROR: a surviving heavy hitter was not in the KEEP tier!")
        return 1
    print("Surviving heavy hitters are all in the KEEP tier (as required).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
