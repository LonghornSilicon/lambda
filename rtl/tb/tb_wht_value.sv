// tb_wht_value.sv — cq_wht_value RTL vs the C++/numpy reference V̂, on real Qwen slices.
// Reads the dump-multi grid (val_i.hex input, vhatwht_i.hex reference) and checks the
// WHT-INT3 reconstruction bit-for-bit. Compile per head-dim (-DQD=64 / 128).
`timescale 1ns/1ps
`ifndef QD
 `define QD 128
`endif
module tb_wht_value;
    localparam int D = `QD, DW = 16;
    reg  [D*DW-1:0] iv;
    wire [D*32-1:0] ov;
    cq_wht_value #(.D(D), .DW(DW)) dut (.in_vec(iv), .vhat_vec(ov));

    string TVDIR;
    integer fman, fv, fg, code, nsl, si, Dn, Tn, Ln, Hn, Bn, t, d, idx;
    reg [DW-1:0] vin [0:255][0:127];
    reg [31:0]   gh  [0:255][0:127];
    reg [DW-1:0] t16; reg [31:0] t32;
    integer checks = 0, pass = 0, slices = 0;

    initial begin
        if (!$value$plusargs("TVDIR=%s", TVDIR)) TVDIR = "tb/testvectors/qwen/g15b/multi";
        fman = $fopen($sformatf("%s/manifest.txt", TVDIR), "r");
        if (fman == 0) begin $display("no manifest at %s", TVDIR); $finish; end
        code = $fscanf(fman, "%d\n", nsl);
        $display("WHT-INT3 value  D=%0d  %0d slices from %s", D, nsl, TVDIR);
        for (si = 0; si < nsl; si = si + 1) begin
            code = $fscanf(fman, "%d %d %d %d %d\n", idx, Dn, Tn, Ln, Hn);
            if (Dn == D) begin
                fv = $fopen($sformatf("%s/val_%0d.hex", TVDIR, idx), "r");
                code = $fscanf(fv, "%d %d %d\n", Dn, Tn, Bn);
                for (t = 0; t < Tn; t = t + 1)
                    for (d = 0; d < Dn; d = d + 1) begin code = $fscanf(fv, "%h", t16); vin[t][d] = t16; end
                $fclose(fv);
                fg = $fopen($sformatf("%s/vhatwht_%0d.hex", TVDIR, idx), "r");
                for (t = 0; t < Tn; t = t + 1)
                    for (d = 0; d < Dn; d = d + 1) begin code = $fscanf(fg, "%h", t32); gh[t][d] = t32; end
                $fclose(fg);
                for (t = 0; t < Tn; t = t + 1) begin
                    for (d = 0; d < Dn; d = d + 1) iv[d*DW +: DW] = vin[t][d];
                    #1;
                    for (d = 0; d < Dn; d = d + 1) begin
                        checks = checks + 1;
                        if (ov[d*32 +: 32] === gh[t][d]) pass = pass + 1;
                        else if (checks - pass <= 5)
                            $display("  MISMATCH s%0d (t%0d,d%0d): rtl=%08h ref=%08h", si, t, d, ov[d*32 +: 32], gh[t][d]);
                    end
                end
                slices = slices + 1;
            end
        end
        $fclose(fman);
        $display("");
        $display("Real-Qwen WHT-INT3 value: %0d/%0d elements bit-exact vs reference across %0d slices (D=%0d)",
                 pass, checks, slices, D);
        if (checks > 0 && pass == checks) $display("ALL TESTS PASSED"); else $display("FAILED");
        $finish;
    end
endmodule
