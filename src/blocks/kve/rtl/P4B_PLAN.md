# P4b — Synthesizable fp16 core lowering (resume plan)

**Goal:** replace the `real`-math behavioral cores in `cq_units.sv` with
synthesizable fp16 fixed-function hardware, **bit-exact vs the golden vectors**,
so the integrated top (branch `feature/cq-top-integration`) synthesizes and the
CI synth / formal-equivalence / OpenLane gates pass. Then merge the branch.

**Why this exists:** the top integration is done + sim-verified on the branch,
but the top instantiates the behavioral cores which use `real` → yosys errors
(`cq_fp_pkg.sv` TOK_REAL). P4b is the unblock. Master stays on the synthesizable
passthrough top until then.

## What must be lowered (all in `cq_units.sv`, all use `cq_fp_pkg` `real`)
- `cq_scale_unit`   : `s = max(amax_fp16 / qmax, EPS)` → fp16      (qmax = 7 or 127)
- `cq_quant_unit`   : `q = clamp(round_half_even(x_fp16 / s_fp16), qmin, qmax)`
- `cq_dequant_unit` : `xhat_fp32 = code * s_fp16`
`cq_pack2` and `amax_unit` are already synthesizable (no `real`).

## Strategy — keep behavioral as oracle, add synthesizable, prove equal
1. New `cq_units_syn.sv`: synthesizable modules, **identical port interfaces**
   (`cq_scale_unit_syn`, `cq_quant_unit_syn`, `cq_dequant_unit_syn`) — no `real`.
2. New `tb/tb_cq_syn.sv` (`make sim_syn`): instantiate BOTH the behavioral and
   the syn core for each op, drive every golden element (all 9 vectors, via the
   hex), assert outputs identical bit-for-bit. Behavioral cores are the oracle.
3. yosys check: `read_verilog cq_units_syn.sv; synth -top ...` → no `real`, 0
   CHECK problems, no inferred latches.
4. Order: **dequant (exact) → scale (÷const) → quant (hardest)**; prove each first.

## fp16 format
`{s[15], e[14:10], m[9:0]}`; normal (e∈[1,30]): `(-1)^s · 1.m · 2^(e-15)`;
subnormal (e=0): `(-1)^s · 0.m · 2^-14`. Golden inputs are finite (no inf/nan).
`EPS = 2^-14 = 0x0400`. Scales are ≥ EPS so the scale output stays normal, but
**inputs x/amax can be subnormal** → handle e=0 significand = `{1'b0, m}` (no
implicit 1), value exponent = -14.

## Algorithms (bit-exact targets)

### dequant  `xhat = code · s`  → fp32   [EXACT, no rounding]
- `code` signed, |code| ≤ 127 (≤8 significant bits). `s` fp16.
- sig_s = 11-bit significand (`{1,m}` normal / `{0,m}` subnormal); exp_s value.
- product significand = `|code| · sig_s` ≤ 8+11 = **19 bits** → fits fp32's 23-bit
  mantissa exactly. Steps: mul → leading-one detect → left-justify to 24 bits →
  drop implicit 1 → fp32 `{sign = code.sign ^ s.sign, exp = 127 + (unbiased),
  mantissa[22:0]}`. `code == 0` → `32'h0`.
- Verify vs `expected_*_hat.f32.hex` (all 9). Should be trivially exact.

### scale  `s = max(amax / qmax, EPS)` → fp16   [÷ constant]
- amax fp16 ≥ 0; qmax ∈ {7 (int4), 127 (int8)} selected by `bits`.
- Compute `amax / qmax` then round-half-even to fp16 (10-bit mantissa), floor EPS.
- Approach: significand division by a small constant. sig_a (11b) / qmax → carry
  ≥ 13 fractional guard bits; round-half-even to the 10-bit fp16 mantissa;
  normalize (÷qmax shifts exponent by −3 for 7, −7..−6 for 127 depending on
  leading one); assemble fp16. If value < EPS → `0x0400`.
- Simpler alt to validate first: multiply by reciprocal with exact correction —
  but the **golden `*_scales.f16.hex` is the arbiter**; iterate until all match.

### quant  `q = clamp(round_half_even(x / s), qmin, qmax)`   [HARDEST]
- Output small int: [−8,7] (int4) or [−128,127] (int8).
- sign_q = sign_x ^ sign_s. magnitude = |x| / |s|.
- Fixed-point divide sig_x (11b) by sig_s (11b), exponent shift `dexp = expv_x −
  expv_s`. Produce the integer quotient + enough fractional bits AND the exact
  remainder. round-half-even: if frac > .5 round up; < .5 down; **== .5 exactly
  (remainder 0 at the half bit) → round to even**.
- Then clamp to [qmin,qmax]; two's-complement if sign_q.
- **Tie correctness is the whole game.** numpy does `np.rint(np.float32(x)/
  np.float32(s))` = round-half-even. Carrying the true remainder makes ties exact.
  The behavioral core uses `double` and matches golden; float32==float64 on all 9
  vectors (verified), so a divider with the full remainder + ≥ a few guard bits
  will match. Iterate guard width / tie logic against `*_payload` codes until
  all 9 vectors are bit-exact.

## Landing sequence (on branch `feature/cq-top-integration`)
1. Land `cq_units_syn.sv` + `tb_cq_syn.sv`, all-9 bit-exact + yosys-clean (master
   or branch — these are additive, synthesizable, so they can even go to master).
2. Point `cq_value_path` / `cq_key_path` at the `_syn` cores (keep behavioral +
   `cq_fp_pkg` for the reference model and the existing parity TBs — they are the
   oracle; do NOT delete them).
3. Re-run `make sim_top / sim_vpath / sim_kpath / sim_cq` — must stay bit-exact.
4. yosys synth the integrated top → clean; read the FF count from the CI artifact
   (apt yosys 0.33 is authoritative — local 0.65 differs) and set
   `.github/workflows/ci.yml expected-ff-count`; add `cq_units_syn.sv` +
   `cq_value_path.sv` + `amax_unit.sv` + `cq_key_path.sv` to `extra-rtl-sources`.
5. Merge `feature/cq-top-integration` → master; confirm CI green (synth/formal/
   OpenLane). Then grouped-key top routing + outlier-mask ROM from vendored
   `masks/`, and the real-data trace.

## Verification gates (every step)
- **G1** `make sim_syn`: behavioral == syn, all 9 golden vectors, bit-exact.
- **G2** yosys: no `real`, 0 CHECK problems, no latches.
- **G3** existing sims stay green after the core swap.

## Traps (learned this session)
- Streaming TBs must drive stimulus on **negedge** (TB/DUT posedge active-region
  race, else the DUT misses the token).
- Local yosys is **0.65** (conda-forge); CI is **apt 0.33** — FF-count numbers
  differ; CI's is the gate arbiter. Toolchain: `. rtl/eda-env.sh`.
- Keep the behavioral cores + `cq_fp_pkg.sv` — they are the golden oracle and the
  basis of the C++/SV 3-way parity; P4b adds the syn cores alongside.
