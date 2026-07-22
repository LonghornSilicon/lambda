# Cadence chamber walkthrough — TSMC 16FFC sign-off

End-to-end steps for running the KV cache engine on a fresh Cadence
chamber, from "I just opened a terminal" to "I have a GDSII and post-PnR
reports". Assumes you have the TSMC University Program 16FFC PDK
provisioned by your university's CAD admin.

Three TCL files do the work; the only edits you make are PDK paths at the
top of each.

## 0 · Inventory check (~2 min)

Before doing anything else, confirm what the chamber actually has. Run
each of these and note the path / version:

```sh
which genus innovus tempus joules
genus -version
innovus -version

# License server reachable?
echo $CDS_LIC_FILE        # or LM_LICENSE_FILE
lmstat -a 2>/dev/null | head -20
```

If `genus` or `innovus` is missing, tell your admin which tools you need.
If license check fails, you're done — no point continuing until that's
fixed.

## 1 · Locate the TSMC 16FFC PDK (~5 min)

The University Program ships TSMC 16FFC as `tcbn16ffcllbwp7t30p140lvt`
(or similar — the suffix varies by your specific signed enablement). Find
the three roots:

```sh
# Liberty (.lib) files — std cells at all PVT corners
find /cad /opt/pdk /pdk -name "*.lib" -path "*16ffc*" 2>/dev/null | head

# LEF files — std cell + tech LEF
find /cad /opt/pdk /pdk -name "*.lef" -path "*16ffc*" 2>/dev/null | head

# QRC tech files — for parasitic extraction
find /cad /opt/pdk /pdk -name "qrcTechFile*" -path "*16ffc*" 2>/dev/null | head
```

You're looking for three corners of the .lib:
- `*ssg0p72v125c.lib`  → SS, 0.72V, 125°C — sign-off setup corner
- `*tt0p80v25c.lib`    → TT, 0.80V, 25°C — typical / datasheet corner
- `*ffg0p88vm40c.lib`  → FF, 0.88V, -40°C — hold-timing corner

The exact filenames depend on which TSMC kit you're licensed for. If your
kit has different names, just substitute them in step 3.

## 2 · Get the repo onto the chamber (~5 min)

```sh
# On the chamber:
cd ~/work
git clone git@github.com:LonghornSilicon/lambda-kve.git   # kve block mirror (or clone the monorepo: LonghornSilicon/lambda, then cd lambda/kve/rtl)
cd lambda-kve/rtl
ls *.tcl                  # genus.tcl, innovus.tcl, mmmc.tcl
```

If the chamber is air-gapped, scp the repo from a machine that has it:

```sh
# From outside:
git archive --format=tar.gz HEAD -o /tmp/kve.tar.gz
scp /tmp/kve.tar.gz chamber:~/work/
# On chamber:
cd ~/work && tar xzf kve.tar.gz && cd kv-cache-engine/rtl
```

## 3 · Plug in the PDK paths (~3 min)

Edit the three PDK variables at the top of `genus.tcl` and `innovus.tcl`.
You can do them once and use them everywhere:

```sh
# Open genus.tcl, find these lines, set them to your PDK paths from step 1:
set TSMC_LIB_DIR  "/path/to/tsmc16/stdcells"
set TSMC_LEF_DIR  "/path/to/tsmc16/lef"
set TSMC_QRC_DIR  "/path/to/tsmc16/qrc"

set LIB_SS_125C   "$TSMC_LIB_DIR/<ss-corner>.lib"
set LIB_TT_25C    "$TSMC_LIB_DIR/<tt-corner>.lib"
set LIB_FF_M40C   "$TSMC_LIB_DIR/<ff-corner>.lib"
```

Repeat the same paths in `innovus.tcl` (top of file). `mmmc.tcl` reads
the variables that `innovus.tcl` already set — no edits needed there.

**Sanity check before launching anything:**

```sh
ls $TSMC_LIB_DIR/<ss-corner>.lib       # must exist
ls $TSMC_LEF_DIR/tsmc16_tech.lef       # must exist
ls $TSMC_QRC_DIR/qrcTechFile_typ       # must exist
```

If any of those fail, the run will crash 30 seconds in. Catch it now.

## 4 · Run Genus synthesis (~10-30 min)

```sh
cd ~/work/kv-cache-engine/rtl
mkdir -p reports netlist
genus -files genus.tcl -log reports/genus.log
```

While it runs you'll see four phases:
1. `read_hdl` / `elaborate` — fast, ~10 sec
2. `syn_generic` — RTL → generic gates, ~1-3 min
3. `syn_map` — map to TSMC 16FFC cells, ~3-10 min
4. `syn_opt` — incremental timing/area opt, ~5-15 min

The KV cache engine is significantly larger than the precision controller
(~6400 FFs vs ~30 FFs), so synthesis takes longer.

When it finishes, the **single file you read first** is:

```sh
cat reports/qor.rpt
```

This tells you in 20 lines whether the design closed. Look for:
- `WNS` (worst negative slack) — should be ≥ 0 ps
- `TNS` (total negative slack) — should be 0 ps
- `Total area` — expected ~15,000-25,000 µm² (vs ~150 µm² for the precision controller)

### What the slack tells you

| WNS at SS | Action |
|---|---|
| `≥ 0 ps` | Done. Move to step 5 (Innovus). |
| `-1 to -100 ps` | Run Genus once more with `set_db syn_global_effort high`. Often closes. |
| `-100 to -500 ps` | Add pipeline registers in the WHT butterfly or quantizer critical path. |
| `< -500 ps` | Something is wrong with the PDK paths or SDC. Re-check step 3. |

### Useful detail reports

```sh
less reports/timing_ss.rpt    # critical path at SS sign-off corner
less reports/timing_tt.rpt    # critical path at TT typical corner
less reports/area.rpt         # cell-by-cell area
less reports/power.rpt        # leakage + estimated dynamic
```

The critical path will likely run through the WHT butterfly adder tree
or the norm_unit's integer sqrt. Both are pipelined in the RTL, but if
slack is tight, check whether additional pipeline stages are needed.

## 5 · Run Innovus PnR (~1-2 hours)

Only after Genus has positive WNS at SS:

```sh
innovus -files innovus.tcl -log reports/innovus.log
```

Phases (visible in the log):
1. `init_design` — ~1 min
2. `floorPlan` + power rings — ~2 min
3. `place_design` — ~10-20 min
4. `ccopt_design` (clock tree) — ~15-30 min
5. `routeDesign` — ~30-60 min (the long pole — larger design)
6. `optDesign -postRoute` — ~10 min
7. `streamOut` (GDSII) — ~2 min

Final outputs:

```sh
ls -la results/
# kv_cache_engine.gds  — GDSII for tape-out
# kv_cache_engine.def  — DEF for placement re-use

less reports/pnr_summary.rpt        # one-line pass/fail
less reports/timing_pnr_setup.rpt   # post-route SS slack (sign-off)
less reports/timing_pnr_hold.rpt    # post-route FF slack (hold)
less reports/power_pnr.rpt          # final dynamic + leakage power
```

## 6 · Power with realistic activity (optional, ~10 min)

For a paper-grade power number, regenerate the replay testbench's switching
activity and feed it to Voltus / Joules:

```sh
# On any machine with iverilog (or the chamber, if iverilog is installed):
cd rtl
make sim_realdata        # produces switching activity
# Edit tb_realdata.sv to add: $dumpvars(0, dut); $dumpfile("activity.vcd");
# Re-run, then load activity.vcd in Joules.
```

If iverilog isn't on the chamber, run it on your laptop, scp `activity.vcd`
across, and load it from there.

## Troubleshooting

**License server timeout.** First thing the admin checks. Make sure
`CDS_LIC_FILE` points at the right port@host.

**`Library not found` in Genus.** Path typo in `LIB_SS_125C` etc. The
ls in step 3 catches this — never skip it.

**`Cell not found in library` for some primitive.** Genus is using a
.lib that doesn't contain the cell types the netlist needs. Check
that you set up the *full* RVT std-cell library, not just a subset.

**Innovus errors on `init_design`.** Almost always a missing tech LEF.
The tech LEF is separate from the cell LEFs; both need to be in
`init_lef_file`.

**`No timing constraints found`.** Confirm `constraints/timing.sdc` is
in the cwd when you launch Genus, and that the path in `genus.tcl`
matches.

**Slack much worse than projected.** Two likely causes:
1. You're reading the FF (best) corner instead of SS (worst). The
   `WNS` you care about is in `timing_ss.rpt`, not `timing_ff.rpt`.
2. The library you're using is HVT instead of LVT. LVT cells are
   ~30% faster.

**Yosys-incompatible SV features.** The Cadence tools (Genus/Innovus)
handle all SystemVerilog features including unpacked array ports. Unlike
the Yosys/OpenLane flow, you can include all sub-modules (cq_key_path,
cq_value_path, cq_units_syn, amax_unit, residual_buffer, scale_bank,
sram_controller) in the Cadence flow without any workarounds.

## Quick reference — what to share back

When you have results, paste the following into our chat:

```
=== Genus QoR ===
$ tail -20 reports/qor.rpt

=== Genus SS slack ===
$ grep -E "Slack|Path" reports/timing_ss.rpt | head

=== PnR final ===
$ tail -30 reports/pnr_summary.rpt
$ grep -E "WNS|TNS|Total power" reports/timing_pnr_setup.rpt reports/power_pnr.rpt
```

That's enough to tell whether we're shipping the current RTL or adding
pipeline registers.
