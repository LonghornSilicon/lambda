// tb_wht_unit.sv — unit-check wht_unit vs the numpy fwht_raw on a real value slice.
// Reads val hex ("D T bits" + T rows of D fp16), forward-WHTs each row, dumps rotated fp16.
`timescale 1ns/1ps
`ifndef QD
 `define QD 128
`endif
module tb_wht_unit;
    localparam int D = `QD, DW = 16;
    reg  [D*DW-1:0] iv;
    wire [D*DW-1:0] ov;
    wht_unit #(.D(D), .DW(DW)) dut (.in_vec(iv), .out_vec(ov));

    string TVDIR; integer fin, fout, code, Dn, Tn, Bn, t, d;
    reg [DW-1:0] tmp;
    initial begin
        if (!$value$plusargs("VAL=%s", TVDIR)) TVDIR = "tb/testvectors/qwen/g15b/multi/val_0.hex";
        fin = $fopen(TVDIR, "r");
        if (fin == 0) begin $display("no %s", TVDIR); $finish; end
        fout = $fopen("/tmp/wht_rtl_out.hex", "w");
        code = $fscanf(fin, "%d %d %d\n", Dn, Tn, Bn);
        for (t = 0; t < Tn; t = t + 1) begin
            for (d = 0; d < Dn; d = d + 1) begin code = $fscanf(fin, "%h", tmp); iv[d*DW +: DW] = tmp; end
            #1;
            for (d = 0; d < Dn; d = d + 1) $fwrite(fout, "%04h ", ov[d*DW +: DW]);
            $fwrite(fout, "\n");
        end
        $fclose(fin); $fclose(fout);
        $display("wht_unit: forward-WHT of %0d rows (D=%0d) -> /tmp/wht_rtl_out.hex", Tn, D);
        $finish;
    end
endmodule
