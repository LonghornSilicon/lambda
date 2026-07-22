# Cadence chamber walkthrough — TSMC 16FFC sign-off

End-to-end steps for running the precision controller on a fresh Cadence
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
- `*ffg0p88vm40c.lib`  → FF, 0.88V, –40°C — hold-timing corner

The exact filenames depend on which TSMC kit you're licensed for. If your
kit has different names, just substitute them in step 3.

## 2 · Get the repo onto the chamber (~5 min)

```sh
# On the chamber:
cd ~/work
git clone git@github.com:LonghornSilicon/attention-compute-unit.git
cd attention-compute-unit/rtl
ls *.tcl                  # genus.tcl, innovus.tcl, mmmc.tcl
```

If the chamber is air-gapped, scp the repo from a machine that has it:

```sh
# From outside:
git archive --format=tar.gz HEAD -o /tmp/apa.tar.gz
scp /tmp/apa.tar.gz chamber:~/work/
# On chamber:
cd ~/work && tar xzf apa.tar.gz && cd attention-compute-unit/rtl
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

## 4 · Run Genus synthesis (~5–15 min)

```sh
cd ~/work/attention-compute-unit/rtl
mkdir -p reports netlist
genus -files genus.tcl -log reports/genus.log
```

While it runs you'll see four phases:
1. `read_hdl` / `elaborate` — fast, ~5 sec
2. `syn_generic` — RTL → generic gates, ~30 sec
3. `syn_map` — map to TSMC 16FFC cells, ~1–2 min
4. `syn_opt` — incremental timing/area opt, ~2–5 min

When it finishes, the **single file you read first** is:

```sh
cat reports/qor.rpt
```

This tells you in 20 lines whether the design closed. Look for:
- `WNS` (worst negative slack) — should be ≥ 0 ps
- `TNS` (total negative slack) — should be 0 ps
- `Total area` — expected ~150 µm² ± 50

### What the slack tells you

| WNS at SS | Action |
|---|---|
| `≥ 0 ps` | ✓ Done. Move to step 5 (Innovus). |
| `−1 to −50 ps` | Run Genus once more with `set_db syn_global_effort high`. Often closes. |
| `−50 to −200 ps` | Apply the pipeline patch (see "Pipeline patch" below). |
| `< −200 ps` | Something is wrong with the PDK paths or SDC. Re-check step 3. |

### Useful detail reports

```sh
less reports/timing_ss.rpt    # critical path at SS sign-off corner
less reports/timing_tt.rpt    # critical path at TT typical corner
less reports/area.rpt         # cell-by-cell area
less reports/power.rpt        # leakage + estimated dynamic
```

The critical path in `timing_ss.rpt` should run through the
`sum_acc` register → shift-add → comparator → `d_fp16` register chain.
That's the path the pipeline patch breaks if needed.

## 5 · Run Innovus PnR (~30–60 min)

Only after Genus has positive WNS at SS:

```sh
innovus -files innovus.tcl -log reports/innovus.log
```

Phases (visible in the log):
1. `init_design` — ~30 sec
2. `floorPlan` + power rings — ~1 min
3. `place_design` — ~5 min
4. `ccopt_design` (clock tree) — ~10 min
5. `routeDesign` — ~15–30 min (the long pole)
6. `optDesign -postRoute` — ~5 min
7. `streamOut` (GDSII) — ~1 min

Final outputs:

```sh
ls -la results/
# precision_controller.gds  — GDSII for tape-out
# precision_controller.def  — DEF for placement re-use

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

## Pipeline patch — only if SS slack is negative

The 30-line modification to `precision_controller.sv`:

```systemverilog
// Add one register stage between sum_next and the (lhs > rhs) comparator.
// Cuts critical path ~50%, costs +24 FFs, latency now 2 cycles after s_last.

reg [CMP_W-1:0] lhs_q, rhs_q;
reg             s_last_q;

always @(posedge clk) begin
    if (!rst_n) begin
        lhs_q    <= '0;
        rhs_q    <= '0;
        s_last_q <= 1'b0;
    end else if (s_valid) begin
        lhs_q    <= lhs;
        rhs_q    <= rhs;
        s_last_q <= s_last;
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        d_valid <= 1'b0;
        d_fp16  <= 1'b0;
    end else begin
        d_valid <= s_last_q;        // pulses one cycle later
        d_fp16  <= (lhs_q > rhs_q);
    end
end
```

After applying the patch, re-run `make sim` and `make sim_realdata` to
re-verify (both testbenches need to be updated for the +1-cycle latency
— change the `@(posedge clk)` count after `s_last` from 1 to 2).

Then re-run Genus from step 4. WNS should improve by ~600 ps at SS.

## Troubleshooting

**License server timeout.** First thing the admin checks. Make sure
`CDS_LIC_FILE` points at the right port@host.

**`Library not found` in Genus.** Path typo in `LIB_SS_125C` etc. The
ls in step 3 catches this — never skip it.

**`Cell not found in library` for some primitive.** Genus is using a
.lib that doesn't contain the cell types Yosys/the netlist needs. Check
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
   ~30% faster; if your PDK only ships HVT, expect ~1700 ps SS path
   and apply the pipeline patch.

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

That's enough for me to tell you whether we're shipping the current RTL
or applying the pipeline patch.
