// tb_qwen_validate.sv — validate the KVCE value-path RTL against REAL Qwen data.
//
// Replays a real Qwen2 value slice (rtl/tb/testvectors/qwen/qwen_v.hex, fp16) through
// cq_value_path (compress + per-channel decompress) and checks the reconstructed
// V̂ bit-for-bit against qwen_vhat_hw.hex — the fp16-exact codec's output
// (analysis/channelquant_hw.py). A pass proves the silicon codec is bit-identical
// to the hardware-faithful software codec ON REAL MODEL TENSORS, not just synthetic
// vectors — so the Qwen accuracy measured with that codec is the accuracy of the RTL.
`timescale 1ns/1ps

module tb_qwen_validate;
    localparam int D = 64;
    localparam int DW = 16;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [3:0]        bits_i = 4'd4;      // CQ-4 per-token values
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

    integer Dn, Tn, Bn, fv, fg, code, t, d;
    reg [DW-1:0] Vbits [0:63][0:255];
    reg [31:0]   Ghat  [0:63][0:255];
    reg [DW-1:0] tmp16; reg [31:0] tmp32;
    integer checks = 0, pass = 0;

    initial begin
        fv = $fopen("tb/testvectors/qwen/qwen_v.hex", "r");
        fg = $fopen("tb/testvectors/qwen/qwen_vhat_hw.hex", "r");
        if (fv == 0 || fg == 0) begin $display("ERROR: missing qwen_v.hex / qwen_vhat_hw.hex"); $finish; end
        code = $fscanf(fv, "%d %d %d\n", Dn, Tn, Bn);
        for (t = 0; t < Tn; t = t + 1) begin
            for (d = 0; d < Dn; d = d + 1) begin code = $fscanf(fv, "%h", tmp16); Vbits[t][d] = tmp16; end
            for (d = 0; d < Dn; d = d + 1) begin code = $fscanf(fg, "%h", tmp32); Ghat[t][d] = tmp32; end
        end
        $fclose(fv); $fclose(fg);
        $display("loaded real Qwen value slice: D=%0d T=%0d bits=%0d", Dn, Tn, Bn);

        rst_n = 0; repeat(3) @(posedge clk); rst_n = 1; @(posedge clk);

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
                    $display("  MISMATCH (t=%0d,d=%0d): rtl V̂=%08h golden=%08h", t, d, dh, Ghat[t][d]);
            end
        end

        $display("");
        $display("Real-Qwen RTL check: %0d/%0d elements bit-exact (V̂ rtl == fp16-exact codec)", pass, checks);
        if (checks > 0 && pass == checks) $display("ALL TESTS PASSED");
        else $display("FAILED (%0d/%0d)", pass, checks);
        $finish;
    end
endmodule
