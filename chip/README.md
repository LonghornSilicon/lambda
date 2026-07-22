# chip/ — cross-block integration (belongs to no single block)

Cross-block work that isn't owned by any one block: the integration cosim and the full-chip PDK
assembly. Mirrors (optionally) to `lambda-chip`.

```
chip/
├── verif/       Cross-block cosim: tb_chip_cosim.sv + vendored block RTL (blocks/) + Makefile + real-Qwen vectors.
│               Run: make -C chip/verif cosim
└── pdk/gf180/   Full-chip GF180MCU padring assembly. HELD placeholder (imports from chipathon-lambda-acu).
```

## verif/ — the cross-block cosim
First cross-block integration of the live block RTLs on one shared attention tile, with a real INT8
P·V MAC so the attention output flows all the way through. Curated copy of `architecture`'s
`rtl/` cosim (rtl branch). See `verif/README.md` for what runs.

**Note (drift):** `verif/blocks/` currently **vendors copies** of the KVE/ACU/TIU block RTL (the
copy-drift the monorepo is meant to eliminate). Once `kve/rtl`, `tiu/rtl`, and `acu/rtl` are the
single source of truth, re-point the `verif/Makefile` include paths at them and drop the vendored
copies. Held this round because ACU isn't imported yet.
