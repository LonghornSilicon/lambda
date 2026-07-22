# tiu/research/ — design rationale, dead ends, experiment ledger

Context for future humans/agents working on TIU: the "why," not just the RTL. Existing exploration
lives in `../analysis/` and `../docs/findings/`; this dir is the home for new design notes, dead
ends, and the experiment ledger (*result · n · artifact · script*).

Key prior findings (pointers, not re-runs):
- **H2O post-softmax mass predicts token importance (r≈0.99);** pre-softmax proxies do not — see the
  ACU sparsity-controller finding referenced in `README.md`.
- **All-three-blocks end-to-end with CQ-3-rot values** — see `../analysis/full_stack_integration.py`
  and `../analysis/full_stack_cq3rot_qwen{05b,15b}.json`.
