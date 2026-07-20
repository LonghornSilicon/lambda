// tb_wht_pathb.sv — Path B end-to-end: cq_value_path_wht (rotated) -> wht_inverse_out == reference V̂.
`timescale 1ns/1ps
`ifndef QD
 `define QD 128
`endif
module tb_wht_pathb;
    localparam int D = `QD, DW = 16;
    reg  [D*DW-1:0] iv;
    wire [D*8-1:0]  codes; wire [DW-1:0] scale;
    reg  [$clog2(D)-1:0] didx;
    wire [DW-1:0]   drot;
    cq_value_path_wht #(.D(D), .DW(DW)) kve (
        .in_vec(iv), .out_codes(codes), .out_scale(scale),
        .dec_codes(codes), .dec_scale(scale), .dec_idx(didx), .dec_rot_f16(drot));
    reg  [D*DW-1:0] rotvec;      // assembled rotated fp16 reconstruction
    wire [D*32-1:0] vhat;
    wht_inverse_out #(.D(D), .DW(DW)) mate (.rot_out(rotvec), .vhat_out(vhat));

    string TVDIR; integer fin, fg, code, Dn, Tn, Bn, t, d, checks=0, pass=0; reg [DW-1:0] tmp; reg [31:0] g;
    reg [DW-1:0] Vin [0:255][0:127]; reg [31:0] Ghat [0:255][0:127];
    initial begin
        if (!$value$plusargs("TVDIR=%s", TVDIR)) TVDIR = "tb/testvectors/qwen/g15b/multi";
        fin = $fopen($sformatf("%s/val_0.hex", TVDIR), "r");
        fg  = $fopen($sformatf("%s/vhatwht_0.hex", TVDIR), "r");
        if (fin==0||fg==0) begin $display("missing vectors in %s", TVDIR); $finish; end
        code = $fscanf(fin, "%d %d %d\n", Dn, Tn, Bn);
        for (t=0;t<Tn;t=t+1) begin
            for (d=0;d<Dn;d=d+1) begin code=$fscanf(fin,"%h",tmp); Vin[t][d]=tmp; end
            for (d=0;d<Dn;d=d+1) begin code=$fscanf(fg,"%h",g);   Ghat[t][d]=g;   end
        end
        $fclose(fin); $fclose(fg);
        for (t=0;t<Tn;t=t+1) begin
            for (d=0;d<Dn;d=d+1) iv[d*DW +: DW] = Vin[t][d];
            #1;
            for (d=0;d<Dn;d=d+1) begin didx = d[$clog2(D)-1:0]; #1; rotvec[d*DW +: DW] = drot; end
            #1;
            for (d=0;d<Dn;d=d+1) begin
                checks=checks+1;
                if (vhat[d*32 +: 32] === Ghat[t][d]) pass=pass+1;
                else if (checks-pass<=6) $display("  MISMATCH t%0d d%0d: pathB=%08h ref=%08h", t, d, vhat[d*32 +: 32], Ghat[t][d]);
            end
        end
        $display("Path B (KVE rotated -> MatE inverse) vs reference V̂: %0d/%0d bit-exact (D=%0d)", pass, checks, D);
        if (checks>0 && pass==checks) $display("ALL TESTS PASSED"); else $display("FAILED");
        $finish;
    end
endmodule
