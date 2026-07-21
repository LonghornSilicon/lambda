// wht_unit_syn.sv — synthesizable WHT butterfly: structural network of fp16_addsub_syn
// cores (no `real` math), bit-identical to the behavioral wht_unit. The synthesis tier.
`timescale 1ns/1ps
module wht_unit_syn #(parameter int D = 128, parameter int DW = 16)(
    input  wire [D*DW-1:0] in_vec,
    output wire [D*DW-1:0] out_vec
);
    localparam int STAGES = $clog2(D);
    wire [DW-1:0] node [0:STAGES][0:D-1];
    genvar s, i, j;
    generate
        for (i = 0; i < D; i = i + 1) assign node[0][i] = in_vec[i*DW +: DW];
        for (s = 0; s < STAGES; s = s + 1) begin: g_stage
            localparam int H = (1 << s);
            for (i = 0; i < D; i = i + (H << 1)) begin: g_blk
                for (j = 0; j < H; j = j + 1) begin: g_bf
                    fp16_addsub_syn u_add (.a(node[s][i+j]), .b(node[s][i+j+H]), .sub(1'b0), .y(node[s+1][i+j]));
                    fp16_addsub_syn u_sub (.a(node[s][i+j]), .b(node[s][i+j+H]), .sub(1'b1), .y(node[s+1][i+j+H]));
                end
            end
        end
        for (i = 0; i < D; i = i + 1) assign out_vec[i*DW +: DW] = node[STAGES][i];
    endgenerate
endmodule
