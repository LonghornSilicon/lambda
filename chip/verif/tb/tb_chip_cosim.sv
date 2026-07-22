// tb_chip_cosim.sv — cross-block RTL cosim: all three block RTLs instantiated together and
// driven by one shared attention scenario. Proves ACU + KVE + TIU co-simulate on real data.
//   KVE : CQ-3-rot value codec (cq_value_path_wht -> wht_inverse_out) — bit-exact vs reference
//   TIU : H2O importance (token_importance_unit) — keep-tier + eviction victim
//   ACU : precision gate (precision_controller) — INT8/FP16 per tile
// Chip order: a token's V flows through KVE; its attention mass drives TIU; the query's
// score row drives ACU. Each block's RTL output is checked against its reference here.
`timescale 1ns/1ps
module tb_chip_cosim;
    localparam int D = 128, DW = 16;
    reg clk = 0, rst_n = 0; always #5 clk = ~clk;
    integer errors = 0;

    // ================= KVE: CQ-3-rot value path (Path B) =================
    reg  [D*DW-1:0] kve_in;
    wire [D*8-1:0]  kve_codes; wire [DW-1:0] kve_scale;
    reg  [$clog2(D)-1:0] kve_didx; wire [DW-1:0] kve_drot;
    cq_value_path_wht #(.D(D), .DW(DW)) u_kve (
        .in_vec(kve_in), .out_codes(kve_codes), .out_scale(kve_scale),
        .dec_codes(kve_codes), .dec_scale(kve_scale), .dec_idx(kve_didx), .dec_rot_f16(kve_drot));
    reg  [D*DW-1:0] kve_rot; wire [D*32-1:0] kve_vhat;
    wht_inverse_out #(.D(D), .DW(DW)) u_mate (.rot_out(kve_rot), .vhat_out(kve_vhat));

    // ===== MatE: INT8 P·V MAC (mate_pv) — the token-reduction accumulation =====
    reg              pv_sv, pv_sl;
    reg  signed [7:0] pv_a;
    reg  [D*8-1:0]   pv_v;
    wire             pv_cv;
    wire signed [D*32-1:0] pv_c;
    mate_pv #(.N(D)) u_pv (
        .clk(clk), .rst_n(rst_n),
        .s_valid(pv_sv), .a_data(pv_a), .v_data(pv_v), .s_last(pv_sl),
        .c_valid(pv_cv), .c_data(pv_c));

    // ===== MatE: FP16 P·V MAC (mate_pv_fp16) — the controller→FP16 escape datapath =====
    // Same streaming interface as mate_pv, but fp16 operands/result + fp32 accumulator.
    reg              pv16_sv, pv16_sl;
    reg  [15:0]      pv16_a;
    reg  [D*16-1:0]  pv16_v;
    wire             pv16_cv;
    wire [D*16-1:0]  pv16_c;
    mate_pv_fp16 #(.N(D)) u_pv16 (
        .clk(clk), .rst_n(rst_n),
        .s_valid(pv16_sv), .a_data(pv16_a), .v_data(pv16_v), .s_last(pv16_sl),
        .c_valid(pv16_cv), .c_data(pv16_c));

    // ================= ACU: precision gate =================
    reg acu_sv, acu_sl; reg signed [7:0] acu_s; wire acu_dv, acu_fp16;
    precision_controller #(.SCORE_WIDTH(8)) u_acu (
        .clk(clk), .rst_n(rst_n), .s_valid(acu_sv), .s_data(acu_s), .s_last(acu_sl),
        .d_valid(acu_dv), .d_fp16(acu_fp16));

    // Tile-sized precision gate for the FP16 escape: N = BLOCK_M*BLOCK_N = 16, so the
    // (max·N > 10·Σ) decision genuinely discriminates on a 16-position attention tile
    // (the shared u_acu is sized to the full 4096-score chip tile).
    reg acu16_sv, acu16_sl; reg signed [7:0] acu16_s; wire acu16_dv, acu16_fp16;
    precision_controller #(.BLOCK_M(4), .BLOCK_N(4), .SCORE_WIDTH(8)) u_acu16 (
        .clk(clk), .rst_n(rst_n), .s_valid(acu16_sv), .s_data(acu16_s), .s_last(acu16_sl),
        .d_valid(acu16_dv), .d_fp16(acu16_fp16));

    // ===== MatE: Q·Kᵀ decode-scoring engine (mate_qkt) — the real score source =====
    // Scores one INT8 query against LQK per-channel-dequantized fp16 keys, reducing
    // over the head-dim DQK. Replaces the hard-coded score row feeding the ACU gate.
    // LQK also = the number of cached tokens the decode attention pass runs over
    // (Q·Kᵀ scores -> softmax weights -> P·V), so it matches the KVE V̂ token count.
    localparam integer LQK = 8;          // keys/tokens scored (= score row length)
    localparam integer DQK = 64;         // head-dim reduction length
    reg               qkt_sv, qkt_sl;
    reg  signed [7:0] qkt_q;
    reg  [LQK*16-1:0] qkt_k;
    wire              qkt_cv;
    wire [LQK*16-1:0] qkt_c;
    mate_qkt #(.N(LQK)) u_qkt (
        .clk(clk), .rst_n(rst_n),
        .s_valid(qkt_sv), .a_data(qkt_q), .k_data(qkt_k), .s_last(qkt_sl),
        .c_valid(qkt_cv), .c_data(qkt_c));

    // ===== VecU: decode online-softmax slice (vecu_softmax) — real attention weights =====
    // Turns the LQK mate_qkt scores into LQK fp16 attention weights (exp-LUT online
    // softmax), which then feed the FP16 P·V — closing Q·Kᵀ -> softmax -> P·V in RTL.
    reg               sm_sv, sm_sl;
    reg  [15:0]       sm_s;
    wire              sm_wv, sm_wl, sm_busy;
    wire [15:0]       sm_w;
    vecu_softmax #(.N(LQK)) u_sm (
        .clk(clk), .rst_n(rst_n),
        .s_valid(sm_sv), .s_data(sm_s), .s_last(sm_sl),
        .w_valid(sm_wv), .w_data(sm_w), .w_last(sm_wl), .busy(sm_busy));

    // ================= TIU: H2O importance =================
    localparam int NS = 8;
    reg tiu_av, tiu_lv, tiu_er; reg [2:0] tiu_as, tiu_ls; reg [7:0] tiu_aw, tiu_thr;
    wire tiu_ev; wire [2:0] tiu_es; wire [NS-1:0] tiu_keep; wire tiu_busy;
    token_importance_unit #(.N_SLOTS(NS), .SCORE_WIDTH(8), .WEIGHT_WIDTH(8)) u_tiu (
        .clk(clk), .rst_n(rst_n), .acc_valid(tiu_av), .acc_slot(tiu_as), .acc_weight(tiu_aw),
        .ld_valid(tiu_lv), .ld_slot(tiu_ls), .evict_req(tiu_er), .evict_valid(tiu_ev),
        .evict_slot(tiu_es), .tier_threshold(tiu_thr), .tier_keep(tiu_keep), .busy(tiu_busy));

    // ---- shared scenario data ----
    reg [DW-1:0] Vin [0:255][0:127]; reg [31:0] Ghat [0:255][0:127];
    integer Dn, Tn, Bn, fv, fg, code, t, d, k, n;
    reg [DW-1:0] t16; reg [31:0] g32;
    reg [7:0] mass [0:NS-1];
    integer exp_evict, mn; reg exp_fp16; integer mx, sm, e0;

    // ---- P·V (MatE) working state ----
    localparam integer PVM = 8;          // tokens accumulated by the P·V tile
    localparam real    PV_TOL = 0.06;    // e2e reconstruction rel-err gate (INT8 tile)
    reg  [DW-1:0] rotv16 [0:PVM*D-1];    // rotated V̂ per (token,channel), fp16, from KVE
    reg  signed [7:0] Vint [0:PVM*D-1];  // int8-quantized rotated V̂ (shared tile scale)
    integer Aint [0:PVM-1];              // int8 attention weights
    integer tbc  [0:D-1];                // TB int32 reference of the P·V accumulation
    real scaleA, scaleV, vmax, rr, orot_r, ortl, oref, gmax, adiff, maxrel;
    integer iv, pd;

    // ---- FP16 P·V (escape) working state ----
    localparam integer PVF     = 8;        // tokens accumulated by the FP16 P·V tile
    localparam real    PVF_TOL = 0.005;    // FP16 path rel-err gate (rel_err < 5e-3)
    reg  [15:0] Af16 [0:PVF-1];            // peaked fp16 attention weights (mass on token 0)
    reg  gate_peak, gate_unif;             // captured gate decisions (peaked / near-uniform)
    real gg, gmax16, rr16, adiff16, maxrel16;

    // ---- Q·Kᵀ decode-scoring working state ----
    localparam real QKT_TOL = 0.005;       // FP16 Q·Kᵀ score rel-err gate (rel_err < 5e-3)
    reg  signed [7:0] Qi [0:DQK-1];        // INT8 query
    reg  [15:0]  Kf [0:LQK*DQK-1];         // per-channel-dequantized fp16 keys (KVE key path)
    reg  [15:0]  ksc;                      // per-channel fp16 key scale
    integer sc_i8 [0:LQK-1];               // int8-quantized scores fed to the gate
    reg  signed [7:0] qc;                  // per-key query·key code (score-spread control)
    real gmaxq, rrq, adq, maxrelq, smaxq, sscaleq;
    integer iq;

    // ---- decode closed-loop (Q·Kᵀ -> softmax -> P·V) working state ----
    localparam real SM_TOL   = 0.05;       // softmax weights vs exact softmax (absorbs ~2% LUT err)
    localparam real PVSM_TOL = 0.06;       // attention output vs reference (softmax LUT thru P·V)
    reg  [15:0] Wsm [0:LQK-1];             // vecu_softmax attention weights (fp16)
    real smax_r, sumexp, refw_n, wmaxrel;

    task step; begin @(negedge clk); end endtask

    initial begin
        // ---------- load a real Qwen attention tile (V + its reference V̂) ----------
        fv = $fopen("vectors/qwen_val.hex", "r"); fg = $fopen("vectors/qwen_vhatwht.hex", "r");
        if (fv==0||fg==0) begin $display("FATAL: missing vectors/"); $finish; end
        code = $fscanf(fv, "%d %d %d\n", Dn, Tn, Bn);
        for (t=0;t<Tn;t=t+1) begin
            for (d=0;d<Dn;d=d+1) begin code=$fscanf(fv,"%h",t16); Vin[t][d]=t16; end
            for (d=0;d<Dn;d=d+1) begin code=$fscanf(fg,"%h",g32); Ghat[t][d]=g32; end
        end
        $fclose(fv); $fclose(fg);
        pv16_sv = 0; pv16_sl = 0; acu16_sv = 0; acu16_sl = 0;   // FP16-escape drives idle at reset
        qkt_sv = 0; qkt_sl = 0;                                 // Q·Kᵀ scorer idle at reset
        sm_sv = 0; sm_sl = 0;                                   // softmax slice idle at reset
        rst_n = 0; repeat(4) step; rst_n = 1; step;

        // ========== BLOCK 2 (KVE): reconstruct each token's V̂, check bit-exact ==========
        if (Tn > 8) Tn = 8;   // cosim: a few tokens suffice to prove KVE bit-exact in-context
        e0 = errors;
        for (t=0;t<Tn;t=t+1) begin
            for (d=0;d<Dn;d=d+1) kve_in[d*DW +: DW] = Vin[t][d];
            #1;
            for (d=0;d<Dn;d=d+1) begin
                kve_didx = d[$clog2(D)-1:0]; #1;
                kve_rot[d*DW +: DW] = kve_drot;
                if (t < PVM) rotv16[t*D + d] = kve_drot;   // stash rotated V̂ for the P·V tile
            end
            #1;
            for (d=0;d<Dn;d=d+1) if (kve_vhat[d*32 +: 32] !== Ghat[t][d]) errors = errors + 1;
        end
        $display("[KVE ] CQ-3-rot V̂ over %0d real-Qwen tokens: %s", Tn, (errors==e0)?"bit-exact vs reference":"MISMATCH");

        // ===== BLOCK 2b (MatE P·V MAC): true end-to-end KVE -> P·V -> inverse =====
        // Insert the INT8 P·V accumulation Σ_t A[t]·V̂rot[t] between the KVE's rotated V̂
        // and wht_inverse_out, so the cosim runs the whole attention-output datapath —
        // not a straight V̂ copy. Bit-exact int32 gate + an e2e reconstruction check:
        // because the inverse WHT is linear, inverse(Σ A·V̂rot) = Σ A·V̂ = Σ A·Ghat, so the
        // reference is TB-computable from Ghat (the reference values) — no model needed.
        e0 = errors;
        for (t=0;t<PVM;t=t+1) Aint[t] = 127 - 10*t;          // distinct positive int8 weights
        scaleA = 1.0/127.0;
        vmax = 0.0;                                          // shared tile scale for V̂rot -> int8
        for (t=0;t<PVM;t=t+1) for (d=0;d<D;d=d+1) begin
            rr = cq_fp_pkg::f16_to_real(rotv16[t*D+d]); if (rr<0.0) rr=-rr;
            if (rr>vmax) vmax=rr;
        end
        scaleV = (vmax>0.0) ? (vmax/127.0) : 1.0;
        for (t=0;t<PVM;t=t+1) for (d=0;d<D;d=d+1) begin
            rr = cq_fp_pkg::f16_to_real(rotv16[t*D+d]) / scaleV;
            iv = $rtoi(rr + (rr>=0.0 ? 0.5 : -0.5));
            if (iv>127) iv=127; if (iv<-127) iv=-127;
            Vint[t*D+d] = iv[7:0];
        end
        for (d=0;d<D;d=d+1) begin                            // TB int32 reference (matmul_int8)
            tbc[d] = 0;
            for (t=0;t<PVM;t=t+1) tbc[d] = tbc[d] + Aint[t]*$signed(Vint[t*D+d]);
        end
        for (t=0;t<PVM;t=t+1) begin                          // drive the mate_pv RTL
            step;
            pv_sv = 1; pv_a = Aint[t][7:0]; pv_sl = (t==PVM-1);
            for (d=0;d<D;d=d+1) pv_v[d*8 +: 8] = Vint[t*D+d];
        end
        step; pv_sv = 0; pv_sl = 0;
        pd = 0; while (pv_cv !== 1'b1 && pd < 8) begin step; pd = pd + 1; end
        if (pv_cv !== 1'b1) begin errors=errors+1; $display("  P·V c_valid never pulsed"); end
        else for (d=0;d<D;d=d+1)
            if ($signed(pv_c[d*32 +: 32]) !== tbc[d]) begin
                errors=errors+1;
                if (d<3) $display("  P·V lane %0d: got %0d exp %0d", d, $signed(pv_c[d*32 +: 32]), tbc[d]);
            end
        $display("[MatE] INT8 P·V MAC (mate_pv), %0d tokens x D=%0d, INT32 acc: %s",
                 PVM, D, (errors==e0)?"int32 bit-exact vs matmul_int8":"MISMATCH");

        // e2e: dequant the int32 result -> wht_inverse_out -> attention output; compare
        // to Σ_t A[t]·Ghat[t] (the reference values). Gap = INT8 P·V quantization only.
        e0 = errors;
        for (d=0;d<D;d=d+1) begin
            orot_r = $itor($signed(pv_c[d*32 +: 32])) * scaleA * scaleV;
            kve_rot[d*DW +: DW] = cq_fp_pkg::real_to_f16(orot_r);
        end
        #1;
        gmax = 1.0e-9;
        for (d=0;d<D;d=d+1) begin
            oref = 0.0;
            for (t=0;t<PVM;t=t+1) oref = oref + ($itor(Aint[t])*scaleA)*cq_fp_pkg::f32_to_real(Ghat[t][d]);
            if (oref<0.0 ? -oref>gmax : oref>gmax) gmax = (oref<0.0?-oref:oref);
        end
        maxrel = 0.0;
        for (d=0;d<D;d=d+1) begin
            ortl = cq_fp_pkg::f32_to_real(kve_vhat[d*32 +: 32]);
            oref = 0.0;
            for (t=0;t<PVM;t=t+1) oref = oref + ($itor(Aint[t])*scaleA)*cq_fp_pkg::f32_to_real(Ghat[t][d]);
            adiff = ortl - oref; if (adiff<0.0) adiff=-adiff;
            if (adiff/gmax > maxrel) maxrel = adiff/gmax;
        end
        if (maxrel >= PV_TOL) errors = errors + 1;
        $display("[MatE] e2e KVE->P·V->inverse vs Sigma A*Ghat: max rel err %f (%s, tol %.2f)",
                 maxrel, (maxrel<PV_TOL)?"within tol":"OUT OF TOL", PV_TOL);

        // ===== BLOCK 2c (MatE FP16 P·V escape): controller routes a PEAKED tile to FP16,
        // then the FP16 P·V tile (mate_pv_fp16) computes Σ_t A[t]·V̂rot[t] and is checked
        // against the sequential-fp32 golden (the faithful streaming order — NOT numpy
        // BLAS pairwise) within the FP16 path's documented rel_err < 5e-3 tolerance. =====
        e0 = errors;

        // (1) the precision gate must ROUTE this peaked 16-position tile to FP16 (escape
        //     genuinely fires — one spike, rest small → max·N > 10·Σ with N=16).
        for (k=0;k<16;k=k+1) begin
            step; acu16_sv=1; acu16_sl=(k==15); acu16_s = (k==0) ? 8'sd120 : 8'sd3;
        end
        step; acu16_sv=0; acu16_sl=0;
        k=0; while (acu16_dv !== 1'b1 && k<8) begin step; k=k+1; end
        gate_peak = acu16_fp16;
        if (acu16_dv !== 1'b1) begin errors=errors+1; $display("  FP16-escape: gate d_valid never pulsed (peaked)"); end
        else if (gate_peak !== 1'b1) begin errors=errors+1; $display("  FP16-escape: peaked tile was NOT routed to FP16 (silently INT8!)"); end

        // (2) a near-UNIFORM 16-position tile must STAY INT8 (the gate discriminates).
        for (k=0;k<16;k=k+1) begin
            step; acu16_sv=1; acu16_sl=(k==15); acu16_s = 8'sd30;
        end
        step; acu16_sv=0; acu16_sl=0;
        k=0; while (acu16_dv !== 1'b1 && k<8) begin step; k=k+1; end
        gate_unif = acu16_fp16;
        if (acu16_dv !== 1'b1) begin errors=errors+1; $display("  FP16-escape: gate d_valid never pulsed (uniform)"); end
        else if (gate_unif !== 1'b0) begin errors=errors+1; $display("  FP16-escape: near-uniform tile wrongly routed to FP16"); end

        // (3) drive the FP16 P·V tile with the peaked attention weights + the KVE's rotated
        //     V̂ (fp16, from BLOCK 2b's stash) — the real in-context escape datapath.
        Af16[0] = cq_fp_pkg::real_to_f16(0.86);                       // mass concentrated on token 0
        for (t=1;t<PVF;t=t+1) Af16[t] = cq_fp_pkg::real_to_f16(0.02);
        for (t=0;t<PVF;t=t+1) begin
            step;
            pv16_sv = 1; pv16_a = Af16[t]; pv16_sl = (t==PVF-1);
            for (d=0;d<D;d=d+1) pv16_v[d*16 +: 16] = rotv16[t*D+d];
        end
        step; pv16_sv = 0; pv16_sl = 0;
        pd = 0; while (pv16_cv !== 1'b1 && pd < 8) begin step; pd = pd + 1; end
        if (pv16_cv !== 1'b1) begin errors=errors+1; $display("  FP16 P·V c_valid never pulsed"); end
        else begin
            // sequential-fp32 golden: o[d] = Σ_t f16(A[t])·f16(V̂rot[t][d]), streaming order.
            // (Accumulated in the TB's fp64 real — for this short reduction fp64-seq and
            //  fp32-seq agree to far below fp16 precision; compare RTL fp16 out within tol.)
            gmax16 = 1.0e-9;
            for (d=0;d<D;d=d+1) begin
                gg = 0.0;
                for (t=0;t<PVF;t=t+1) gg = gg + cq_fp_pkg::f16_to_real(Af16[t])*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
                rr16 = (gg<0.0) ? -gg : gg; if (rr16>gmax16) gmax16 = rr16;
            end
            maxrel16 = 0.0;
            for (d=0;d<D;d=d+1) begin
                gg = 0.0;
                for (t=0;t<PVF;t=t+1) gg = gg + cq_fp_pkg::f16_to_real(Af16[t])*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
                rr16 = cq_fp_pkg::f16_to_real(pv16_c[d*16 +: 16]);
                adiff16 = rr16 - gg; if (adiff16<0.0) adiff16=-adiff16;
                if (adiff16/gmax16 > maxrel16) maxrel16 = adiff16/gmax16;
            end
            if (maxrel16 >= PVF_TOL) begin errors=errors+1; $display("  FP16 P·V OUT OF TOL: max rel err %f (tol %.3f)", maxrel16, PVF_TOL); end
        end
        $display("[MatE] FP16 P·V escape: gate routes FP16=%0b (peaked) / FP16=%0b (uniform) -> escape %s; tile Sigma A*Vhat max rel err %f vs seq-fp32 golden (%s, tol %.3f)",
                 gate_peak, gate_unif, (gate_peak==1'b1 && gate_unif==1'b0)?"FIRED & discriminates":"BROKEN",
                 maxrel16, (errors==e0)?"within tol":"FAIL", PVF_TOL);

        // ========== BLOCK 3 (TIU): install slots, accumulate mass, keep-tier + evict ==========
        // masses derived from the tile (per-token amax magnitude, quantized to a weight)
        e0 = errors;
        for (k=0;k<NS;k=k+1) mass[k] = (Vin[k][0] & 8'hFF);          // deterministic per-token weight
        for (k=0;k<NS;k=k+1) begin step; tiu_lv=1; tiu_ls=k[2:0]; end // install NS tokens
        step; tiu_lv=0;
        for (k=0;k<NS;k=k+1) begin step; tiu_av=1; tiu_as=k[2:0]; tiu_aw=mass[k]; end // accumulate mass
        step; tiu_av=0; tiu_thr = 8'd128; step; step;
        // expected keep (score>=thr) and evict (min-mass slot, first-wins on ties)
        exp_evict = 0; mn = mass[0];
        for (k=1;k<NS;k=k+1) if (mass[k] < mn) begin mn = mass[k]; exp_evict = k; end
        for (k=0;k<NS;k=k+1) if (tiu_keep[k] !== (mass[k] >= tiu_thr)) errors = errors + 1;
        // request eviction, then wait on the evict_valid handshake (serial scan ~N_SLOTS+2 cyc)
        tiu_er = 1; step; tiu_er = 0;
        k = 0; while (tiu_ev !== 1'b1 && k < 40) begin step; k = k + 1; end
        if (tiu_ev !== 1'b1) begin errors=errors+1; $display("  TIU evict_valid never pulsed"); end
        else if (tiu_es !== exp_evict[2:0]) begin errors=errors+1; $display("  TIU evict got=%0d exp=%0d", tiu_es, exp_evict); end
        $display("[TIU ] keep-tier (thr=%0d) + eviction victim: %s (evict slot %0d)",
                 tiu_thr, (errors==e0)?"match reference":"MISMATCH", tiu_es);

        // ========== BLOCK 1 (MatE Q·Kᵀ scoring -> ACU gate) ==========
        // The score row is now COMPUTED by the mate_qkt RTL from an INT8 query and the
        // KVE's per-channel-dequantized fp16 keys (cq_dequant_f16 = round_fp16(code·
        // scale)) — replacing the hard-coded stand-in — then quantized to int8 (per-tile
        // symmetric, matching integration_example.quantize_int8) and gated by the
        // precision controller. Keys give a spread of scores in a softmax-friendly
        // range (key l dequantizes to code_l/64, so score_l = Σ_d 1·(code_l/64) =
        // code_l): a real distribution the downstream softmax must resolve, while the
        // gate still routes FP16.
        e0 = errors;
        for (d=0;d<DQK;d=d+1) Qi[d] = 8'sd1;                   // query: +1 on every channel
        ksc = cq_fp_pkg::real_to_f16(1.0/64.0);               // per-channel fp16 key scale
        for (n=0;n<LQK;n=n+1) begin
            case (n)                                          // score targets (moderate spread)
                0: qc = 8'sd3;  1: qc = 8'sd1;  2: qc = 8'sd0;  3: qc = -8'sd1;
                4: qc = 8'sd2;  5: qc = -8'sd2; 6: qc = 8'sd1;  default: qc = -8'sd3;
            endcase
            for (d=0;d<DQK;d=d+1) Kf[n*DQK+d] = cq_fp_pkg::cq_dequant_f16(qc, ksc);
        end
        // stream the DQK head-dim channels through mate_qkt
        for (d=0;d<DQK;d=d+1) begin
            step;
            qkt_sv=1; qkt_q=Qi[d]; qkt_sl=(d==DQK-1);
            for (n=0;n<LQK;n=n+1) qkt_k[n*16 +: 16] = Kf[n*DQK+d];
        end
        step; qkt_sv=0; qkt_sl=0;
        pd=0; while (qkt_cv !== 1'b1 && pd<8) begin step; pd=pd+1; end
        if (qkt_cv !== 1'b1) begin errors=errors+1; $display("  Q·Kᵀ c_valid never pulsed"); end
        else begin
            // (a) check the mate_qkt scores vs the sequential-fp32 golden (rel_err<5e-3)
            gmaxq = 1.0e-9;
            for (n=0;n<LQK;n=n+1) begin
                gg = 0.0;
                for (d=0;d<DQK;d=d+1) gg = gg + $itor(Qi[d])*cq_fp_pkg::f16_to_real(Kf[n*DQK+d]);
                rrq = (gg<0.0)?-gg:gg; if (rrq>gmaxq) gmaxq = rrq;
            end
            maxrelq = 0.0;
            for (n=0;n<LQK;n=n+1) begin
                gg = 0.0;
                for (d=0;d<DQK;d=d+1) gg = gg + $itor(Qi[d])*cq_fp_pkg::f16_to_real(Kf[n*DQK+d]);
                rrq = cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]);
                adq = rrq - gg; if (adq<0.0) adq=-adq;
                if (adq/gmaxq > maxrelq) maxrelq = adq/gmaxq;
            end
            if (maxrelq >= QKT_TOL) begin errors=errors+1; $display("  Q·Kᵀ scores OUT OF TOL: %f (tol %.3f)", maxrelq, QKT_TOL); end
        end
        // (b) quantize the fp16 scores to int8 (per-tile symmetric) and gate the tile
        smaxq = 0.0;
        for (n=0;n<LQK;n=n+1) begin rrq=cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]); if (rrq<0.0) rrq=-rrq; if (rrq>smaxq) smaxq=rrq; end
        sscaleq = (smaxq>1.0e-9) ? (smaxq/127.0) : 1.0;
        mx = 0; sm = 0;
        for (n=0;n<LQK;n=n+1) begin
            rrq = cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]) / sscaleq;
            iq = $rtoi(rrq + (rrq>=0.0?0.5:-0.5)); if (iq>127) iq=127; if (iq<-127) iq=-127;
            sc_i8[n] = iq;
            if ((iq<0?-iq:iq) > mx) mx = (iq<0?-iq:iq);
            sm = sm + (iq<0?-iq:iq);
        end
        for (n=0;n<LQK;n=n+1) begin step; acu_sv=1; acu_sl=(n==LQK-1); acu_s = sc_i8[n][7:0]; end
        step; acu_sv=0; acu_sl=0;
        k = 0; while (acu_dv !== 1'b1 && k < 8) begin step; k = k + 1; end
        exp_fp16 = (mx*4096 > 10*sm);   // RTL gate: max·N > 10·Σ, N = BLOCK_M*BLOCK_N = 4096
        if (acu_dv !== 1'b1) begin errors=errors+1; $display("  ACU d_valid never pulsed"); end
        else if (acu_fp16 !== exp_fp16) begin errors=errors+1; $display("  ACU fp16 got=%0b exp=%0b (max=%0d sum=%0d)", acu_fp16, exp_fp16, mx, sm); end
        $display("[ACU ] Q·Kᵀ(mate_qkt) scores -> precision gate: scores rel-err %f (<%.3f), gate fp16=%0b: %s",
                 maxrelq, QKT_TOL, acu_fp16, (errors==e0)?"match reference":"MISMATCH");

        // ========== BLOCK 2d (decode closed loop: Q·Kᵀ -> softmax -> P·V) ==========
        // The attention weights feeding the FP16 P·V are now COMPUTED by vecu_softmax
        // from the mate_qkt scores (replacing the reference-supplied weights) — the whole
        // decode attention pass is real RTL. Weights are checked against exact fp64
        // softmax; the P·V attention output against the reference attention within a
        // tolerance set from the measured ~2% exp-LUT error propagating through P·V.
        e0 = errors;
        // (1) stream the LQK mate_qkt scores through vecu_softmax, collect the weights
        for (n=0;n<LQK;n=n+1) begin
            step; sm_sv=1; sm_s=qkt_c[n*16 +: 16]; sm_sl=(n==LQK-1);
        end
        step; sm_sv=0; sm_sl=0;
        k=0; pd=0;
        // micro-sequenced vecu_softmax: COMPUTE ~8 cyc/score + EMIT ~8 cyc/weight
        while (k<LQK && pd<(LQK*20+64)) begin
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
            // (2) softmax weights vs reference softmax
            wmaxrel = 0.0;
            for (n=0;n<LQK;n=n+1) begin
                refw_n = $exp(cq_fp_pkg::f16_to_real(qkt_c[n*16 +: 16]) - smax_r) / sumexp;
                adq = cq_fp_pkg::f16_to_real(Wsm[n]) - refw_n; if (adq<0.0) adq=-adq;
                if (adq > wmaxrel) wmaxrel = adq;   // weights <= 1, so abs err ~ rel-to-1
            end
            if (wmaxrel >= SM_TOL) begin errors=errors+1; $display("  softmax weights OUT OF TOL: %f (tol %.3f)", wmaxrel, SM_TOL); end
        end
        // (3) feed the softmax weights to the FP16 P·V over the KVE's rotated V̂
        for (t=0;t<LQK;t=t+1) begin
            step; pv16_sv=1; pv16_a=Wsm[t]; pv16_sl=(t==LQK-1);
            for (d=0;d<D;d=d+1) pv16_v[d*16 +: 16] = rotv16[t*D+d];
        end
        step; pv16_sv=0; pv16_sl=0;
        pd=0; while (pv16_cv !== 1'b1 && pd<8) begin step; pd=pd+1; end
        if (pv16_cv !== 1'b1) begin errors=errors+1; $display("  closed-loop P·V c_valid never pulsed"); end
        else begin
            // reference attention: o_ref[d] = Σ_t softmax_ref[t]·V̂[t][d]
            gmax16 = 1.0e-9;
            for (d=0;d<D;d=d+1) begin
                gg=0.0;
                for (t=0;t<LQK;t=t+1) gg = gg + ($exp(cq_fp_pkg::f16_to_real(qkt_c[t*16 +: 16])-smax_r)/sumexp)*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
                rr16 = (gg<0.0)?-gg:gg; if (rr16>gmax16) gmax16=rr16;
            end
            maxrel16 = 0.0;
            for (d=0;d<D;d=d+1) begin
                gg=0.0;
                for (t=0;t<LQK;t=t+1) gg = gg + ($exp(cq_fp_pkg::f16_to_real(qkt_c[t*16 +: 16])-smax_r)/sumexp)*cq_fp_pkg::f16_to_real(rotv16[t*D+d]);
                rr16 = cq_fp_pkg::f16_to_real(pv16_c[d*16 +: 16]);
                adiff16 = rr16-gg; if (adiff16<0.0) adiff16=-adiff16;
                if (adiff16/gmax16 > maxrel16) maxrel16 = adiff16/gmax16;
            end
            if (maxrel16 >= PVSM_TOL) begin errors=errors+1; $display("  closed-loop P·V OUT OF TOL: %f (tol %.3f)", maxrel16, PVSM_TOL); end
        end
        $display("[VecU] decode Q·Kᵀ->softmax->P·V closed loop (weights = vecu_softmax RTL): softmax err %f (<%.3f), attn-out rel-err %f (<%.3f): %s",
                 wmaxrel, SM_TOL, maxrel16, PVSM_TOL, (errors==e0)?"within tol":"FAIL");

        $display("");
        $display("CROSS-BLOCK COSIM (decode Q·Kᵀ->softmax->P·V all-RTL + ACU gate + KVE + INT8/FP16 P·V + TIU): %s", (errors==0)?"ALL BLOCKS PASS":"FAILED");
        $finish;
    end
endmodule
