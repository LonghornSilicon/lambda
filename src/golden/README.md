# src/golden/ — chip-level golden reference

Chip-level bit-accurate reference models and the cross-block golden index. **Per-block reference
models are the source of truth** and live with their block (`src/blocks/<block>/sw/reference_model/`);
this directory holds chip-level golden (full decode-loop reference, LSU schedule assembler) and an
index of the per-block models.

Per-block reference models (source of truth):
| Block | Reference model | Parity |
|---|---|---|
| kve | [`../blocks/kve/sw/reference_model/`](../blocks/kve/sw/reference_model/) | Py + C++ (3-way vs RTL) |
| tiu | [`../blocks/tiu/sw/reference_model/`](../blocks/tiu/sw/reference_model/) | Py |
| acu/mate | [`../blocks/acu/mate/sw/reference_model/`](../blocks/acu/mate/sw/reference_model/) | Py + C++ (FP16 tol-based) |
| acu/vecu | [`../blocks/acu/vecu/sw/reference_model/`](../blocks/acu/vecu/sw/reference_model/) | Py |
| acu/precision_controller | [`../blocks/acu/precision_controller/sw/reference_model/`](../blocks/acu/precision_controller/sw/reference_model/) | Py + C++ |

The reference model — not prose — is authoritative for numeric behavior
([`../../docs/documentation_standard.md`](../../docs/documentation_standard.md) §3). Chip-level
golden (LSU assembler, end-to-end decode reference) lands here as integration matures.
