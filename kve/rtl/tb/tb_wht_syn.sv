// tb_wht_syn.sv — wht_unit_syn (synthesizable) vs wht_unit (behavioral) on real Qwen rows.
`timescale 1ns/1ps
`ifndef QD
 `define QD 128
`endif
module tb_wht_syn;
    localparam int D = `QD, DW = 16;
    reg  [D*DW-1:0] iv;
    wire [D*DW-1:0] ov_beh, ov_syn;
    wht_unit     #(.D(D), .DW(DW)) beh (.in_vec(iv), .out_vec(ov_beh));
    wht_unit_syn #(.D(D), .DW(DW)) syn (.in_vec(iv), .out_vec(ov_syn));
    string TVDIR; integer fin, code, Dn, Tn, Bn, t, d, checks=0, pass=0; reg [DW-1:0] tmp;
    initial begin
        if (!$value$plusargs("VAL=%s", TVDIR)) TVDIR = "tb/testvectors/qwen/g15b/multi/val_0.hex";
        fin = $fopen(TVDIR, "r"); if (fin==0) begin $display("no %s", TVDIR); $finish; end
        code = $fscanf(fin, "%d %d %d\n", Dn, Tn, Bn);
        for (t = 0; t < Tn; t = t + 1) begin
            for (d = 0; d < Dn; d = d + 1) begin code = $fscanf(fin, "%h", tmp); iv[d*DW +: DW] = tmp; end
            #1;
            for (d = 0; d < Dn; d = d + 1) begin
                checks = checks + 1;
                if (ov_syn[d*DW +: DW] === ov_beh[d*DW +: DW]) pass = pass + 1;
                else if (checks-pass<=8) $display("  MISMATCH t%0d d%0d: syn=%04h beh=%04h", t, d, ov_syn[d*DW +: DW], ov_beh[d*DW +: DW]);
            end
        end
        $fclose(fin);
        $display("wht_unit_syn vs behavioral wht_unit: %0d/%0d bit-exact (D=%0d)", pass, checks, D);
        if (checks>0 && pass==checks) $display("ALL TESTS PASSED"); else $display("FAILED");
        $finish;
    end
endmodule
