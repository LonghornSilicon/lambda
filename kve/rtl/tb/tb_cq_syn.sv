`timescale 1ns / 1ps
// tb_cq_syn.sv — equivalence gate for the synthesizable cores (P4b).
//
// Drives each behavioral core (cq_units.sv, the `real`-math oracle) and its
// synthesizable twin (cq_units_syn.sv) with the SAME inputs and asserts the
// outputs are bit-identical over a broad fp16 sweep. Because the behavioral
// cores are already proven bit-exact vs the golden vectors (make sim_cq /
// reference model), syn == behavioral here ⇒ syn == golden. Grows one core at a
// time: dequant → scale → quant.

module tb_cq_syn;

    // clock/reset for the sequential quant core
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    // representative fp16 mantissas (corners + spread)
    integer mans [0:8];
    integer nman;

    integer errs_total;

    // ---- dequant ----------------------------------------------------------
    reg  signed [7:0] dq_code;
    reg  [15:0]       dq_scl;
    wire [31:0]       dq_beh, dq_syn;
    cq_dequant_unit     u_dq_b (.code(dq_code), .scale_f16(dq_scl), .xhat_f32(dq_beh));
    cq_dequant_unit_syn u_dq_s (.code(dq_code), .scale_f16(dq_scl), .xhat_f32(dq_syn));

    // ---- scale ------------------------------------------------------------
    reg  [15:0] sc_amax;
    reg  [3:0]  sc_bits;
    wire [15:0] sc_beh, sc_syn;
    cq_scale_unit     u_sc_b (.amax_f16(sc_amax), .bits(sc_bits), .scale_f16(sc_beh));
    cq_scale_unit_syn u_sc_s (.amax_f16(sc_amax), .bits(sc_bits), .scale_f16(sc_syn));

    integer ei, mi, ci, t, e;
    integer bi, se, sm;

    task run_scale;
        begin
            t = 0; e = 0;
            for (bi = 0; bi < 2; bi = bi + 1) begin
                sc_bits = (bi == 0) ? 4'd4 : 4'd8;
                for (se = 0; se <= 30; se = se + 1)
                  for (sm = 0; sm < 1024; sm = sm + 1) begin
                    sc_amax = {1'b0, se[4:0], sm[9:0]};
                    #1;
                    t = t + 1;
                    if (sc_beh !== sc_syn) begin
                        e = e + 1;
                        if (e <= 15) $display("  SCL MISMATCH amax=%h bits=%0d  beh=%h syn=%h",
                                              sc_amax, sc_bits, sc_beh, sc_syn);
                    end
                  end
            end
            $display("scale   : %0d combos, %0d mismatches%s", t, e,
                     (e == 0) ? "  -> BIT-EXACT" : "  -> FAILED");
            errs_total = errs_total + e;
        end
    endtask

    task run_dequant;
        begin
            t = 0; e = 0;
            for (ei = 0; ei <= 30; ei = ei + 1)
              for (mi = 0; mi < nman; mi = mi + 1) begin
                dq_scl = {1'b0, ei[4:0], mans[mi][9:0]};
                for (ci = -128; ci <= 127; ci = ci + 1) begin
                    dq_code = ci[7:0];
                    #1;
                    t = t + 1;
                    if (dq_beh !== dq_syn) begin
                        e = e + 1;
                        if (e <= 15) $display("  DEQ MISMATCH code=%0d scl=%h  beh=%h syn=%h",
                                              ci, dq_scl, dq_beh, dq_syn);
                    end
                end
              end
            $display("dequant : %0d combos, %0d mismatches%s", t, e,
                     (e == 0) ? "  -> BIT-EXACT" : "  -> FAILED");
            errs_total = errs_total + e;
        end
    endtask

    // ---- quant -----------------------------------------------------------
    reg  [15:0] qz_x, qz_scl;
    reg  [3:0]  qz_bits;
    reg         qz_start;
    wire        qz_done;
    wire signed [7:0] qz_beh, qz_syn;
    cq_quant_unit     u_qz_b (.x_f16(qz_x), .scale_f16(qz_scl), .bits(qz_bits), .code(qz_beh));
    cq_quant_unit_syn u_qz_s (.clk(clk), .rst_n(rst_n), .start(qz_start),
                              .x_f16(qz_x), .scale_f16(qz_scl), .bits(qz_bits),
                              .code(qz_syn), .done(qz_done));

    // realistic scales (exp_s in [1,30], several mantissas) to sweep x against
    integer scls [0:11];
    integer nscl, qi, qb, qx, qstride;

    task run_quant;
        begin
            // {sign=0, exp, man}
            scls[0]={1'b0,5'd1, 10'd0};    scls[1]={1'b0,5'd7, 10'd0};
            scls[2]={1'b0,5'd7, 10'd341};  scls[3]={1'b0,5'd11,10'd512};
            scls[4]={1'b0,5'd13,10'd1023}; scls[5]={1'b0,5'd15,10'd0};
            scls[6]={1'b0,5'd15,10'd683};  scls[7]={1'b0,5'd18,10'd200};
            scls[8]={1'b0,5'd21,10'd777};  scls[9]={1'b0,5'd25,10'd512};
            scls[10]={1'b0,5'd28,10'd1000};scls[11]={1'b0,5'd30,10'd1023};
            nscl = 12;
            t = 0; e = 0;
            for (qb = 0; qb < 2; qb = qb + 1) begin
                qz_bits = (qb == 0) ? 4'd4 : 4'd8;
                for (qi = 0; qi < nscl; qi = qi + 1) begin
                    qz_scl = scls[qi][15:0];
                    for (qx = 0; qx < 65536; qx = qx + qstride) begin
                        if (qx[14:10] != 5'd31) begin   // skip inf/nan (contract: finite inputs)
                            // clocked handshake to the sequential syn core; the
                            // behavioral oracle qz_beh is combinational from the
                            // same inputs, sampled once qz_done asserts.
                            @(negedge clk); qz_x = qx[15:0]; qz_start = 1'b1;
                            @(negedge clk); qz_start = 1'b0;
                            while (!qz_done) @(negedge clk);
                            t = t + 1;
                            if (qz_beh !== qz_syn) begin
                                e = e + 1;
                                if (e <= 15) $display("  QNT MISMATCH x=%h scl=%h bits=%0d  beh=%0d syn=%0d",
                                                      qz_x, qz_scl, qz_bits, qz_beh, qz_syn);
                            end
                        end
                    end
                end
            end
            $display("quant   : %0d combos, %0d mismatches%s", t, e,
                     (e == 0) ? "  -> BIT-EXACT" : "  -> FAILED");
            errs_total = errs_total + e;
        end
    endtask

    initial begin
        mans[0]=0;   mans[1]=1;   mans[2]=5;   mans[3]=341; mans[4]=512;
        mans[5]=682; mans[6]=1000; mans[7]=1023; mans[8]=768;
        nman = 9;
        errs_total = 0;
        qz_start = 1'b0;
        qstride  = 1;          // full x sweep; raise for a quick smoke run
        // release reset (only the sequential quant core needs it)
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        $display("============================================================");
        $display("CQ-SYN equivalence: synthesizable cores vs behavioral oracle");
        $display("============================================================");

        run_dequant;
        run_scale;
        run_quant;

        $display("------------------------------------------------------------");
        if (errs_total == 0)
            $display("CQ-SYN: ALL CORES BIT-EXACT vs behavioral");
        else
            $display("CQ-SYN: %0d TOTAL MISMATCHES", errs_total);
        $finish;
    end

endmodule
