<!-- TIU settled calls. Append-only; never delete, mark superseded. what · why · date.
     Seeded from docs/prototypes/DECISIONS.seed.md at monorepo creation 2026-07-22.
     (The DECISIONS.seed.md had no dedicated TIU block; TIU-relevant calls are captured here from
     the tier-handshake work. Add rows as TIU decisions land.) -->

# DECISIONS — TIU (do not re-litigate unless the premise changed)

- **Token importance = H2O accumulated attention mass (post-softmax)** · post-softmax mass predicts
  token importance (r≈0.99); pre-softmax proxies do not (ACU sparsity study) · 2026-07.
- **Tier handshake with KVE: TIU owns keep/evict; the value-precision role of `tier_keep` is
  dropped** once KVE moves to a single flat value tier (CQ-3-rot / per-token INT) · there's no
  per-token value bit-width to select anymore · 2026-07. See `docs/tier_handshake.md`.
