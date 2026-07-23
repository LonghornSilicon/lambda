# kve/research/ — design rationale, dead ends, experiment ledger

Context for future humans/agents working on KVE: the "why," not just the RTL. Existing exploration
already lives alongside in `../analysis/`, `../findings/`, and `../NOTES.md` — this dir is the home
for new design notes, dead ends, and the experiment ledger (*result · n · artifact · script*).

Key prior findings (pointers, not re-runs):
- **CQ-4 vs CQ-4+ (outlier lane):** n=1000 on real Qwen2-0.5B/1.5B reversed the n=250 screening —
  the FP16 outlier lane only marginally helps at D=128 and slightly hurts at D=64. See `../analysis/`.
- **WHT value rotation (CQ-3-rot):** flat 3.0 b/val, near-lossless; rotation primitive is
  reconfigurable (per-channel sign vector). See `../docs/wht_value_rotation.md` and `DECISIONS.md`.
- **TurboQuant+ retired 2026-06-22:** ~3.5× but −0.10 HellaSwag acc_norm collapse on GQA (rotation
  delocalizes key-channel error). See `README.md`.
