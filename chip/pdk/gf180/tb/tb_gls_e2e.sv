// tb_gls_e2e.sv — GF180 GATE-LEVEL cross-block end-to-end verification.
//
// Reproduces the cross-block check from the architecture repo's
// rtl/tb/tb_chip_cosim.sv (KVE reconstruct V̂ -> MatE P·V -> ACU precision gate
// -> TIU keep/evict, on a real Qwen attention tile), but instantiates the
// GF180 LibreLane gate-level netlists (runs/<macro>/final/nl/<macro>.nl.v)
// against the gf180mcu_fd_sc_mcu7t5v0 standard-cell Verilog models. The
// reference computations and tolerance gates are the SAME as the RTL cosim:
// INT paths bit-exact, FP16 path rel_err < 5e-3.
//
// COVERAGE BOUNDARY (be explicit — see docs/gf180_gls_report.md):
//   GATE-LEVEL (GF180 hardened netlists) : mate_pv, mate_pv_fp16,
//                                          precision_controller, token_importance_unit
//   RTL (feeds the gate-level blocks)    : KVE value path (cq_value_path_wht +
//                                          wht_inverse_out) — combinational,
//                                          reconstructs the real V̂ that drives
//                                          the gate-level P·V. Its own bit-exact
//                                          check here is RTL, not gate-level.
//
// The hardened macros bake in proxy parameters (mate_pv/_fp16 N=4, tiu
// N_SLOTS=4, precision_controller BLOCK_M*BLOCK_N=4096). This TB drives each
// gate-level netlist at its baked width with the identical reference formula,
// so the match-vs-expected is a true gate-level result at that width.
//
// Build: see tb/Makefile target `test-gls-e2e`.

`timescale 1ns/1ps

module tb_gls_e2e;
    localparam int D    = 128, DW = 16;  // KVE head dim / fp16 width (RTL)
    localparam int NPV  = 4;             // mate_pv / mate_pv_fp16 hardened lane count (N=4)
    localparam int NS   = 4;             // token_importance_unit hardened N_SLOTS
    localparam int ACU_N = 4096;         // precision_controller hardened tile (64*64)
    localparam int LQK  = 8;             // mate_qkt / vecu_softmax hardened N (keys / row length)
    localparam int DQK  = 64;            // Q·Kᵀ head-dim reduction length (TB stimulus)

    reg clk = 0, rst_n = 0; always #5 clk = ~clk;
    integer errors = 0, e0;

    // ============ KVE value path (RTL — reconstructs the real V̂) ============
    reg  [D*DW-1:0] kve_in;
    wire [D*8-1:0]  kve_codes; wire [DW-1:0] kve_scale;
    reg  [$clog2(D)-1:0] kve_didx; wire [DW-1:0] kve_drot;
    cq_value_path_wht #(.D(D), .DW(DW)) u_kve (
        .in_vec(kve_in), .out_codes(kve_codes), .out_scale(kve_scale),
        .dec_codes(kve_codes), .dec_scale(kve_scale), .dec_idx(kve_didx), .dec_rot_f16(kve_drot));
    reg  [D*DW-1:0] kve_rot; wire [D*32-1:0] kve_vhat;
    wht_inverse_out #(.D(D), .DW(DW)) u_wht (.rot_out(kve_rot), .vhat_out(kve_vhat));

    // ============ MatE INT8 P·V (GF180 gate-level, N=4) ============
    reg               pv_sv, pv_sl;
    reg  signed [7:0] pv_a;
    reg  [NPV*8-1:0]  pv_v;
    wire              pv_cv;
    wire signed [NPV*32-1:0] pv_c;
    mate_pv u_pv (                        // no #(): params baked into the netlist
        .clk(clk), .rst_n(rst_n),
        .s_valid(pv_sv), .a_data(pv_a), .v_data(pv_v), .s_last(pv_sl),
        .c_valid(pv_cv), .c_data(pv_c));

    // ============ MatE FP16 P·V escape (GF180 gate-level, N=4) ============
    reg               pv16_sv, pv16_sl;
    reg  [15:0]       pv16_a;
    reg  [NPV*16-1:0] pv16_v;
    wire              pv16_cv;
    wire [NPV*16-1:0] pv16_c;
    mate_pv_fp16 u_pv16 (
        .clk(clk), .rst_n(rst_n),
        .s_valid(pv16_sv), .a_data(pv16_a), .v_data(pv16_v), .s_last(pv16_sl),
        .c_valid(pv16_cv), .c_data(pv16_c));

    // ============ ACU precision gate (GF180 gate-level, N=4096 tile) ============
    reg acu_sv, acu_sl; reg signed [7:0] acu_s; wire acu_dv, acu_fp16;
    precision_controller u_acu (
        .clk(clk), .rst_n(rst_n), .s_valid(acu_sv), .s_data(acu_s), .s_last(acu_sl),
        .d_valid(acu_dv), .d_fp16(acu_fp16));

    // ============ TIU H2O importance (GF180 gate-level, N_SLOTS=4) ============
    reg tiu_av, tiu_lv, tiu_er; reg [1:0] tiu_as, tiu_ls; reg [7:0] tiu_aw, tiu_thr;
    wire tiu_ev; wire [1:0] tiu_es; wire [NS-1:0] tiu_keep; wire tiu_busy;
    token_importance_unit u_tiu (
        .clk(clk), .rst_n(rst_n), .acc_valid(tiu_av), .acc_slot(tiu_as), .acc_weight(tiu_aw),
        .ld_valid(tiu_lv), .ld_slot(tiu_ls), .evict_req(tiu_er), .evict_valid(tiu_ev),
        .evict_slot(tiu_es), .tier_threshold(tiu_thr), .tier_keep(tiu_keep), .busy(tiu_busy));

    // ============ MatE Q·Kᵀ decode scoring (GF180 gate-level, N=8) ============
    reg               qkt_sv, qkt_sl;
    reg  signed [7:0] qkt_q;
    reg  [LQK*16-1:0] qkt_k;
    wire              qkt_cv;
    wire [LQK*16-1:0] qkt_c;
    mate_qkt u_qkt (                      // no #(): N=8 baked into the netlist
        .clk(clk), .rst_n(rst_n),
        .s_valid(qkt_sv), .a_data(qkt_q), .k_data(qkt_k), .s_last(qkt_sl),
        .c_valid(qkt_cv), .c_data(qkt_c));

    // ============ VecU decode online-softmax (GF180 gate-level, N=8) ============
    reg               sm_sv, sm_sl;
    reg  [15:0]       sm_s;
    wire              sm_wv, sm_wl, sm_busy;
    wire [15:0]       sm_w;
    vecu_softmax u_sm (                   // no #(): N=8 baked into the netlist
        .clk(clk), .rst_n(rst_n),
        .s_valid(sm_sv), .s_data(sm_s), .s_last(sm_sl),
        .w_valid(sm_wv), .w_data(sm_w), .w_last(sm_wl), .busy(sm_busy));

    // ---- shared scenario data (real Qwen tile) ----
    reg [DW-1:0] Vin [0:255][0:127]; reg [31:0] Ghat [0:255][0:127];
    integer Dn, Tn, Bn, fv, fg, code, t, d, k;
    reg [DW-1:0] t16; reg [31:0] g32;

    localparam integer PVM = 8;          // tokens accumulated by the P·V tiles
    localparam real    PV_TOL  = 0.06;   // INT8 rotated-space reconstruction tol
    localparam real    PVF_TOL = 0.005;  // FP16 path rel-err gate (rel_err < 5e-3)
    reg  [DW-1:0] rotv16 [0:PVM*D-1];    // rotated V̂ per (token,channel), fp16, from KVE
    reg  signed [7:0] Vint [0:PVM*NPV-1];
    integer Aint [0:PVM-1];
    integer tbc  [0:NPV-1];
    real scaleA, scaleV, vmax, rr, ortl, oref, gmax, adiff, maxrel;
    integer iv, pd, mx, sm;
    reg exp_fp16, gate_peak, gate_unif;
    real gg, gmax16, rr16, adiff16, maxrel16;
    reg [15:0] Af16 [0:PVM-1];
    reg [7:0] mass [0:NS-1]; integer exp_evict, mn;

    // ---- Q·Kᵀ -> softmax -> P·V decode datapath working state ----
    localparam real QKT_TOL  = 0.005;    // fp16 Q·Kᵀ score rel-err gate (< 5e-3)
    localparam real SM_TOL   = 0.05;     // softmax weights vs exact softmax (absorbs ~2% LUT err)
    localparam real PVSM_TOL = 0.06;     // attention output vs reference (softmax LUT thru P·V)
    reg  signed [7:0] Qi [0:DQK-1];      // INT8 query
    reg  [15:0]  Kf [0:LQK*DQK-1];       // per-channel-dequantized fp16 keys
    reg  [15:0]  ksc; reg signed [7:0] qc;
    integer sc_i8 [0:LQK-1]; integer iq, n;
    real gmaxq, rrq, adq, maxrelq, smaxq, sscaleq;
    reg  [15:0] Wsm [0:LQK-1];           // vecu_softmax attention weights (fp16)
    real smax_r, sumexp, refw_n, wmaxrel;

    task step; begin @(negedge clk); end endtask

    initial begin
        fv = $fopen("vectors/qwen_val.hex", "r"); fg = $fopen("vectors/qwen_vhatwht.hex", "r");
        if (fv==0||fg==0) begin $display("FATAL: missing vectors/"); $finish; end
        code = $fscanf(fv, "%d %d %d\n", Dn, Tn, Bn);
        for (t=0;t<Tn;t=t+1) begin
            for (d=0;d<Dn;d=d+1) begin code=$fscanf(fv,"%h",t16); Vin[t][d]=t16; end
            for (d=0;d<Dn;d=d+1) begin code=$fscanf(fg,"%h",g32); Ghat[t][d]=g32; end
        end
        $fclose(fv); $fclose(fg);
        pv_sv=0; pv_sl=0; pv16_sv=0; pv16_sl=0; acu_sv=0; acu_sl=0;
        tiu_av=0; tiu_lv=0; tiu_er=0;
        qkt_sv=0; qkt_sl=0; sm_sv=0; sm_sl=0;
        rst_n = 0; repeat(6) step; rst_n = 1; step;

        $display("=== GF180 GATE-LEVEL cross-block end-to-end (gf180mcu_fd_sc_mcu7t5v0) ===");

        // ===== KVE (RTL): reconstruct each token's V̂, bit-exact vs reference =====
        if (Tn > PVM) Tn = PVM;
        e0 = errors;
        for (t=0;t<Tn;t=t+1) begin
            for (d=0;d<Dn;d=d+1) kve_in[d*DW +: DW] = Vin[t][d];
            #1;
            for (d=0;d<Dn;d=d+1) begin
                kve_didx = d[$clog2(D)-1:0]; #1;
                kve_rot[d*DW +: DW] = kve_drot;
                rotv16[t*D + d] = kve_drot;         // stash rotated V̂ for the P·V tiles
            end
            #1;
            for (d=0;d<Dn;d=d+1) if (kve_vhat[d*32 +: 32] !== Ghat[t][d]) errors = errors + 1;
        end
        $display("[KVE  RTL] CQ-3-rot V̂ over %0d real-Qwen tokens: %s",
                 Tn, (errors==e0)?"bit-exact vs reference":"MISMATCH");

        // ===== MatE INT8 P·V (GATE-LEVEL): Σ_t A[t]·V̂rot[t], first NPV channels =====
        e0 = errors;
        for (t=0;t<PVM;t=t+1) Aint[t] = 127 - 10*t;
        scaleA = 1.0/127.0;
        vmax = 0.0;
        for (t=0;t<PVM;t=t+1) for (d=0;d<NPV;d=d+1) begin
            rr = cq_fp_pkg::f16_to_real(rotv16[t*D+d]); if (rr<0.0) rr=-rr;
            if (rr>vmax) vmax=rr;
        end
        scaleV = (vmax>0.0) ? (vmax/127.0) : 1.0;
        for (t=0;t<PVM;t=t+1) for (d=0;d<NPV;d=d+1) begin
            rr = cq_fp_pkg::f16_to_real(rotv16[t*D+d]) / scaleV;
            iv = $rtoi(rr + (rr>=0.0 ? 0.5 : -0.5));
            if (iv>127) iv=127; if (iv<-127) iv=-127;
            Vint[t*NPV+d] = iv[7:0];
        end
        for (d=0;d<NPV;d=d+1) begin
            tbc[d] = 0;
            for (t=0;t<PVM;t=t+1) tbc[d] = tbc[d] + Aint[t]*$signed(Vint[t*NPV+d]);
        end
        for (t=0;t<PVM;t=t+1) begin
            step;
            pv_sv = 1; pv_a = Aint[t][7:0]; pv_sl = (t==PVM-1);
            for (d=0;d<NPV;d=d+1) pv_v[d*8 +: 8] = Vint[t*NPV+d];
        end
        step; pv_sv = 0; pv_sl = 0;
        pd = 0; while (pv_cv !== 1'b1 && pd < 8) begin step; pd = pd + 1; end
        if (pv_cv !== 1'b1) begin errors=errors+1; $display("  P·V c_valid never pulsed"); end
        else for (d=0;d<NPV;d=d+1)
            if ($signed(pv_c[d*32 +: 32]) !== tbc[d]) begin
                errors=errors+1;
                $display("  P·V lane %0d: got %0d exp %0d", d, $signed(pv_c[d*32 +: 32]), tbc[d]);
            end
        $display("[MatE  GL] INT8 P·V MAC, %0d tokens x N=%0d, INT32 acc: %s",
                 PVM, NPV, (errors==e0)?"int32 BIT-EXACT vs matmul_int8":"MISMATCH");

        // e2e: dequant the gate-level int32 result and compare to Σ A·V̂rot (fp)
        e0 = errors;
        gmax = 1.0e-9;
        for (d=0;d<NPV;d=d+1) begin
            oref = 0.0;
            for (t=0;t<PVM;t=t+1) oref = oref + ($itor(Aint[t])*scaleA)*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
            if (oref<0.0 ? -oref>gmax : oref>gmax) gmax = (oref<0.0?-oref:oref);
        end
        maxrel = 0.0;
        for (d=0;d<NPV;d=d+1) begin
            ortl = $itor($signed(pv_c[d*32 +: 32])) * scaleA * scaleV;
            oref = 0.0;
            for (t=0;t<PVM;t=t+1) oref = oref + ($itor(Aint[t])*scaleA)*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
            adiff = ortl - oref; if (adiff<0.0) adiff=-adiff;
            if (adiff/gmax > maxrel) maxrel = adiff/gmax;
        end
        if (maxrel >= PV_TOL) errors = errors + 1;
        $display("[MatE  GL] e2e KVE->P·V dequant vs Sigma A*Vhat: max rel err %f (%s, tol %.2f)",
                 maxrel, (maxrel<PV_TOL)?"within tol":"OUT OF TOL", PV_TOL);

        // ===== MatE FP16 P·V escape (GATE-LEVEL): rel_err < 5e-3 vs seq-fp32 golden =====
        e0 = errors;
        Af16[0] = cq_fp_pkg::real_to_f16(0.86);
        for (t=1;t<PVM;t=t+1) Af16[t] = cq_fp_pkg::real_to_f16(0.02);
        for (t=0;t<PVM;t=t+1) begin
            step;
            pv16_sv = 1; pv16_a = Af16[t]; pv16_sl = (t==PVM-1);
            for (d=0;d<NPV;d=d+1) pv16_v[d*16 +: 16] = rotv16[t*D+d];
        end
        step; pv16_sv = 0; pv16_sl = 0;
        pd = 0; while (pv16_cv !== 1'b1 && pd < 8) begin step; pd = pd + 1; end
        if (pv16_cv !== 1'b1) begin errors=errors+1; $display("  FP16 P·V c_valid never pulsed"); end
        else begin
            gmax16 = 1.0e-9;
            for (d=0;d<NPV;d=d+1) begin
                gg = 0.0;
                for (t=0;t<PVM;t=t+1) gg = gg + cq_fp_pkg::f16_to_real(Af16[t])*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
                rr16 = (gg<0.0) ? -gg : gg; if (rr16>gmax16) gmax16 = rr16;
            end
            maxrel16 = 0.0;
            for (d=0;d<NPV;d=d+1) begin
                gg = 0.0;
                for (t=0;t<PVM;t=t+1) gg = gg + cq_fp_pkg::f16_to_real(Af16[t])*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
                rr16 = cq_fp_pkg::f16_to_real(pv16_c[d*16 +: 16]);
                adiff16 = rr16 - gg; if (adiff16<0.0) adiff16=-adiff16;
                if (adiff16/gmax16 > maxrel16) maxrel16 = adiff16/gmax16;
            end
            if (maxrel16 >= PVF_TOL) errors=errors+1;
        end
        $display("[MatE  GL] FP16 P·V escape: tile Sigma A*Vhat max rel err %f vs seq-fp32 golden (%s, tol %.3f)",
                 maxrel16, (errors==e0)?"within tol":"OUT OF TOL", PVF_TOL);

        // ===== ACU precision gate (GATE-LEVEL, N=4096): peaked->FP16, uniform->INT8 =====
        e0 = errors;
        // (1) peaked full tile
        mx = 0; sm = 0;
        for (k=0;k<ACU_N;k=k+1) begin
            step; acu_sv=1; acu_sl=(k==ACU_N-1); acu_s = (k==0) ? 8'sd120 : 8'sd3;
            if (((k==0)?120:3) > mx) mx = (k==0)?120:3; sm = sm + ((k==0)?120:3);
        end
        step; acu_sv=0; acu_sl=0;
        k=0; while (acu_dv !== 1'b1 && k<8) begin step; k=k+1; end
        gate_peak = acu_fp16;
        exp_fp16 = (mx*ACU_N > 10*sm);
        if (acu_dv !== 1'b1) begin errors=errors+1; $display("  ACU d_valid never pulsed (peaked)"); end
        else if (acu_fp16 !== exp_fp16) begin errors=errors+1; $display("  ACU peaked got=%0b exp=%0b", acu_fp16, exp_fp16); end
        // (2) near-uniform full tile
        mx = 0; sm = 0;
        for (k=0;k<ACU_N;k=k+1) begin
            step; acu_sv=1; acu_sl=(k==ACU_N-1); acu_s = 8'sd30;
            if (30 > mx) mx = 30; sm = sm + 30;
        end
        step; acu_sv=0; acu_sl=0;
        k=0; while (acu_dv !== 1'b1 && k<8) begin step; k=k+1; end
        gate_unif = acu_fp16;
        exp_fp16 = (mx*ACU_N > 10*sm);
        if (acu_dv !== 1'b1) begin errors=errors+1; $display("  ACU d_valid never pulsed (uniform)"); end
        else if (acu_fp16 !== exp_fp16) begin errors=errors+1; $display("  ACU uniform got=%0b exp=%0b", acu_fp16, exp_fp16); end
        $display("[ACU   GL] precision gate: FP16=%0b (peaked) / FP16=%0b (uniform) -> %s (match reference decision: %s)",
                 gate_peak, gate_unif,
                 (gate_peak==1'b1 && gate_unif==1'b0)?"discriminates":"BROKEN",
                 (errors==e0)?"YES":"NO");

        // ===== TIU H2O importance (GATE-LEVEL, N_SLOTS=4): keep-tier + evict =====
        e0 = errors;
        for (k=0;k<NS;k=k+1) mass[k] = (Vin[k][0] & 8'hFF);
        for (k=0;k<NS;k=k+1) begin step; tiu_lv=1; tiu_ls=k[1:0]; end
        step; tiu_lv=0;
        for (k=0;k<NS;k=k+1) begin step; tiu_av=1; tiu_as=k[1:0]; tiu_aw=mass[k]; end
        step; tiu_av=0; tiu_thr = 8'd128; step; step;
        exp_evict = 0; mn = mass[0];
        for (k=1;k<NS;k=k+1) if (mass[k] < mn) begin mn = mass[k]; exp_evict = k; end
        for (k=0;k<NS;k=k+1) if (tiu_keep[k] !== (mass[k] >= tiu_thr)) errors = errors + 1;
        tiu_er = 1; step; tiu_er = 0;
        k = 0; while (tiu_ev !== 1'b1 && k < 40) begin step; k = k + 1; end
        if (tiu_ev !== 1'b1) begin errors=errors+1; $display("  TIU evict_valid never pulsed"); end
        else if (tiu_es !== exp_evict[1:0]) begin errors=errors+1; $display("  TIU evict got=%0d exp=%0d", tiu_es, exp_evict); end
        $display("[TIU   GL] keep-tier (thr=%0d) + eviction victim: %s (evict slot %0d, exp %0d)",
                 tiu_thr, (errors==e0)?"match reference":"MISMATCH", tiu_es, exp_evict);

        // ===== Q·Kᵀ decode scoring (GATE-LEVEL mate_qkt, N=8) -> ACU gate =====
        // Score one INT8 query against LQK per-channel-dequantized fp16 keys over
        // DQK head-dim channels, then quantize + gate. (cosim BLOCK 1.)
        e0 = errors;
        for (d=0;d<DQK;d=d+1) Qi[d] = 8'sd1;                   // query: +1 on every channel
        ksc = cq_fp_pkg::real_to_f16(1.0/64.0);                // per-channel fp16 key scale
        for (n=0;n<LQK;n=n+1) begin
            case (n)                                           // score targets (moderate spread)
                0: qc = 8'sd3;  1: qc = 8'sd1;  2: qc = 8'sd0;  3: qc = -8'sd1;
                4: qc = 8'sd2;  5: qc = -8'sd2; 6: qc = 8'sd1;  default: qc = -8'sd3;
            endcase
            for (d=0;d<DQK;d=d+1) Kf[n*DQK+d] = cq_fp_pkg::cq_dequant_f16(qc, ksc);
        end
        for (d=0;d<DQK;d=d+1) begin                            // stream head-dim channels
            step; qkt_sv=1; qkt_q=Qi[d]; qkt_sl=(d==DQK-1);
            for (n=0;n<LQK;n=n+1) qkt_k[n*16 +: 16] = Kf[n*DQK+d];
        end
        step; qkt_sv=0; qkt_sl=0;
        pd=0; while (qkt_cv !== 1'b1 && pd<8) begin step; pd=pd+1; end
        if (qkt_cv !== 1'b1) begin errors=errors+1; $display("  Q·Kᵀ c_valid never pulsed"); end
        else begin
            // (a) mate_qkt scores vs sequential-fp32 golden (rel_err < 5e-3)
            gmaxq = 1.0e-9;
            for (n=0;n<LQK;n=n+1) begin
                gg = 0.0;
                for (d=0;d<DQK;d=d+1) gg = gg + $itor(Qi[d])*cq_fp_pkg::f16_to_real(Kf[n*DQK+d]);
                rrq=(gg<0.0)?-gg:gg; if (rrq>gmaxq) gmaxq=rrq;
            end
            maxrelq = 0.0;
            for (n=0;n<LQK;n=n+1) begin
                gg = 0.0;
                for (d=0;d<DQK;d=d+1) gg = gg + $itor(Qi[d])*cq_fp_pkg::f16_to_real(Kf[n*DQK+d]);
                rrq = cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]);
                adq = rrq-gg; if (adq<0.0) adq=-adq;
                if (adq/gmaxq > maxrelq) maxrelq = adq/gmaxq;
            end
            if (maxrelq >= QKT_TOL) begin errors=errors+1; $display("  Q·Kᵀ scores OUT OF TOL: %f (tol %.3f)", maxrelq, QKT_TOL); end
        end
        // (b) quantize scores to int8 (per-tile symmetric) and gate the tile
        smaxq = 0.0;
        for (n=0;n<LQK;n=n+1) begin rrq=cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]); if (rrq<0.0) rrq=-rrq; if (rrq>smaxq) smaxq=rrq; end
        sscaleq = (smaxq>1.0e-9) ? (smaxq/127.0) : 1.0;
        mx=0; sm=0;
        for (n=0;n<LQK;n=n+1) begin
            rrq = cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]) / sscaleq;
            iq = $rtoi(rrq + (rrq>=0.0?0.5:-0.5)); if (iq>127) iq=127; if (iq<-127) iq=-127;
            sc_i8[n]=iq;
            if ((iq<0?-iq:iq) > mx) mx = (iq<0?-iq:iq);
            sm = sm + (iq<0?-iq:iq);
        end
        for (n=0;n<LQK;n=n+1) begin step; acu_sv=1; acu_sl=(n==LQK-1); acu_s=sc_i8[n][7:0]; end
        step; acu_sv=0; acu_sl=0;
        k=0; while (acu_dv !== 1'b1 && k<8) begin step; k=k+1; end
        exp_fp16 = (mx*ACU_N > 10*sm);
        if (acu_dv !== 1'b1) begin errors=errors+1; $display("  ACU(qkt) d_valid never pulsed"); end
        else if (acu_fp16 !== exp_fp16) begin errors=errors+1; $display("  ACU(qkt) got=%0b exp=%0b", acu_fp16, exp_fp16); end
        $display("[MatE  GL] Q·Kᵀ (mate_qkt) scores: rel-err %f (< %.3f) vs seq-fp32 golden; -> ACU gate fp16=%0b: %s",
                 maxrelq, QKT_TOL, acu_fp16, (errors==e0)?"match reference":"MISMATCH");

        // ===== decode closed loop: Q·Kᵀ -> softmax (GL vecu_softmax) -> P·V (GL) =====
        // (cosim BLOCK 2d.) The attention weights feeding the FP16 P·V are computed
        // by vecu_softmax from the mate_qkt scores — the whole compute datapath is GL.
        e0 = errors;
        for (n=0;n<LQK;n=n+1) begin step; sm_sv=1; sm_s=qkt_c[n*16 +: 16]; sm_sl=(n==LQK-1); end
        step; sm_sv=0; sm_sl=0;
        // vecu_softmax is multi-cycle (micro-sequenced COMPUTE/EMIT, ~8 fp32-op
        // cycles per score + per weight), so weights arrive well after s_last;
        // the collection window is data-independent — just wait on the w_valid
        // handshake for all LQK weights, with a generous bound.
        k=0; pd=0;
        while (k<LQK && pd<600) begin
            step;
            if (sm_wv) begin Wsm[k]=sm_w; k=k+1; end
            pd=pd+1;
        end
        if (k !== LQK) begin errors=errors+1; $display("  softmax: only %0d of %0d weights emitted", k, LQK); end
        else begin
            // reference exact-fp64 softmax of the (fp16) scores
            smax_r = -1.0e30;
            for (n=0;n<LQK;n=n+1) begin rrq=cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]); if (rrq>smax_r) smax_r=rrq; end
            sumexp = 0.0;
            for (n=0;n<LQK;n=n+1) sumexp = sumexp + $exp(cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]) - smax_r);
            wmaxrel = 0.0;
            for (n=0;n<LQK;n=n+1) begin
                refw_n = $exp(cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]) - smax_r) / sumexp;
                adq = cq_fp_pkg::f16_to_real(Wsm[n]) - refw_n; if (adq<0.0) adq=-adq;
                if (adq > wmaxrel) wmaxrel = adq;
            end
            if (wmaxrel >= SM_TOL) begin errors=errors+1; $display("  softmax weights OUT OF TOL: %f (tol %.3f)", wmaxrel, SM_TOL); end
        end
        // feed the softmax weights to the GL FP16 P·V over the KVE's rotated V̂ (first NPV ch)
        for (t=0;t<LQK;t=t+1) begin
            step; pv16_sv=1; pv16_a=Wsm[t]; pv16_sl=(t==LQK-1);
            for (d=0;d<NPV;d=d+1) pv16_v[d*16 +: 16] = rotv16[t*D+d];
        end
        step; pv16_sv=0; pv16_sl=0;
        pd=0; while (pv16_cv !== 1'b1 && pd<8) begin step; pd=pd+1; end
        if (pv16_cv !== 1'b1) begin errors=errors+1; $display("  closed-loop P·V c_valid never pulsed"); end
        else begin
            // reference attention: o_ref[d] = Σ_t softmax_ref[t]·V̂[t][d]
            gmax16 = 1.0e-9;
            for (d=0;d<NPV;d=d+1) begin
                gg=0.0;
                for (t=0;t<LQK;t=t+1) gg = gg + ($exp(cq_fp_pkg::f16_to_real(qkt_c[t*16 +: 16])-smax_r)/sumexp)*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
                rr16=(gg<0.0)?-gg:gg; if (rr16>gmax16) gmax16=rr16;
            end
            maxrel16 = 0.0;
            for (d=0;d<NPV;d=d+1) begin
                gg=0.0;
                for (t=0;t<LQK;t=t+1) gg = gg + ($exp(cq_fp_pkg::f16_to_real(qkt_c[t*16 +: 16])-smax_r)/sumexp)*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
                rr16 = cq_fp_pkg::f16_to_real(pv16_c[d*16 +: 16]);
                adiff16 = rr16-gg; if (adiff16<0.0) adiff16=-adiff16;
                if (adiff16/gmax16 > maxrel16) maxrel16 = adiff16/gmax16;
            end
            if (maxrel16 >= PVSM_TOL) begin errors=errors+1; $display("  closed-loop P·V OUT OF TOL: %f (tol %.3f)", maxrel16, PVSM_TOL); end
        end
        $display("[VecU  GL] decode Q·Kᵀ->softmax->P·V (weights = vecu_softmax GL): softmax err %f (< %.3f), attn-out rel-err %f (< %.3f): %s",
                 wmaxrel, SM_TOL, maxrel16, PVSM_TOL, (errors==e0)?"within tol":"FAIL");

        $display("");
        $display("GF180 GATE-LEVEL E2E (full compute datapath Q·Kᵀ->softmax->P·V + ACU + TIU gate-level; KVE RTL): %s",
                 (errors==0)?"ALL PASS":"FAILED");
        $finish;
    end
endmodule
