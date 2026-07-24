// tb_qwen_multi.sv — value-path RTL vs a GRID of real Qwen tensors (bit-exact).
//
// Extends tb_qwen_validate from one slice to a whole manifest of real Qwen (layer,head)
// value slices dumped by analysis/channelquant_hw.py --mode dump-multi. Replays each
// slice through cq_value_path and checks V̂ bit-for-bit vs the fp16-exact codec
// (valhat_i.hex). Compile once per head-dim:
//   D=64  (Qwen2-0.5B):  iverilog -DQD=64  ... +TVDIR=tb/testvectors/qwen/g05b/multi
//   D=128 (Qwen2-1.5B):  iverilog -DQD=128 ... +TVDIR=tb/testvectors/qwen/g15b/multi
`timescale 1ns/1ps
`ifndef QD
 `define QD 128
`endif

module tb_qwen_multi;
    localparam int D  = `QD;
    localparam int DW = 16;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [3:0]        bits_i = 4'd4;
    reg               iv = 0;
    reg  [D*DW-1:0]   ivec = 0;
    wire              busy, ov;
    wire [DW-1:0]     osc;
    wire [D*8-1:0]    ocode, opay;
    reg  [D*8-1:0]    dc = 0;
    reg  [DW-1:0]     ds = 0;
    reg  [$clog2(D)-1:0] didx = 0;
    wire [31:0]       dh;

    cq_value_path #(.D(D), .DW(DW)) dut (
        .clk(clk), .rst_n(rst_n), .bits(bits_i),
        .in_valid(iv), .in_vec(ivec), .busy(busy),
        .out_valid(ov), .out_scale(osc), .out_codes(ocode), .out_pay(opay),
        .dec_codes(dc), .dec_scale(ds), .dec_idx(didx), .dec_hat(dh));

    string TVDIR;
    integer fman, fv, fg, code, nsl, si, Dn, Tn, Ln, Hn, Bn, t, d;
    reg [DW-1:0] Vbits [0:127][0:127];
    reg [31:0]   Ghat  [0:127][0:127];
    reg [DW-1:0] tmp16; reg [31:0] tmp32;
    integer checks = 0, pass = 0, slices = 0;
    string vpath, gpath;

    initial begin
        if (!$value$plusargs("TVDIR=%s", TVDIR)) TVDIR = "tb/testvectors/qwen/g15b/multi";
        fman = $fopen($sformatf("%s/manifest.txt", TVDIR), "r");
        if (fman == 0) begin $display("ERROR: no manifest at %s", TVDIR); $finish; end
        code = $fscanf(fman, "%d\n", nsl);
        $display("D=%0d  manifest: %0d slices from %s", D, nsl, TVDIR);

        rst_n = 0; repeat(3) @(posedge clk); rst_n = 1; @(posedge clk);

        // manifest lines: "i D T layer head"
        for (si = 0; si < nsl; si = si + 1) begin
            code = $fscanf(fman, "%d %d %d %d %d\n", Bn, Dn, Tn, Ln, Hn); // Bn=slice idx, Dn=D, Tn=T
            if (Dn != D) begin
                $display("  slice %0d: D=%0d != %0d, skipped", Bn, Dn, D);
            end else begin
            vpath = $sformatf("%s/val_%0d.hex",    TVDIR, Bn);
            gpath = $sformatf("%s/valhat_%0d.hex", TVDIR, Bn);
            fv = $fopen(vpath, "r"); fg = $fopen(gpath, "r");
            if (fv == 0 || fg == 0) begin $display("  ERROR opening slice %0d", Bn); $finish; end
            code = $fscanf(fv, "%d %d %d\n", Dn, Tn, Bn);   // header D T bits (Bn now = bits)
            for (t = 0; t < Tn; t = t + 1) begin
                for (d = 0; d < Dn; d = d + 1) begin code = $fscanf(fv, "%h", tmp16); Vbits[t][d] = tmp16; end
                for (d = 0; d < Dn; d = d + 1) begin code = $fscanf(fg, "%h", tmp32); Ghat[t][d] = tmp32; end
            end
            $fclose(fv); $fclose(fg);

            for (t = 0; t < Tn; t = t + 1) begin
                @(negedge clk);
                ivec = '0;
                for (d = 0; d < Dn; d = d + 1) ivec[d*DW +: DW] = Vbits[t][d];
                iv = 1'b1; @(negedge clk); iv = 1'b0;
                while (!ov) @(negedge clk);
                dc = ocode; ds = osc;
                for (d = 0; d < Dn; d = d + 1) begin
                    didx = d[$clog2(D)-1:0]; #1;
                    checks = checks + 1;
                    if (dh === Ghat[t][d]) pass = pass + 1;
                    else if (checks - pass <= 5)
                        $display("  MISMATCH slice=%0d (t=%0d,d=%0d): rtl=%08h golden=%08h", si, t, d, dh, Ghat[t][d]);
                end
            end
            slices = slices + 1;
            end
        end
        $fclose(fman);

        $display("");
        $display("Real-Qwen VALUE-path grid: %0d/%0d elements bit-exact across %0d slices (D=%0d)",
                 pass, checks, slices, D);
        if (checks > 0 && pass == checks) $display("ALL TESTS PASSED");
        else $display("FAILED (%0d/%0d)", pass, checks);
        $finish;
    end
endmodule
