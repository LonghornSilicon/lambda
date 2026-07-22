// wht_unit.sv — raw fixed Walsh-Hadamard butterfly over D fp16 words (add/sub only).
//
// Structural log2(D)-stage butterfly of fp16 add/sub cores (cq_fp_pkg::fp16_add/sub),
// matching channelquant_ref.hpp::fwht_raw_f16 exactly (per-stage round-half-even). NO
// normalization — the per-token amax absorbs the sqrt(D) magnitude on compress, and the
// decode applies the exact 1/D (2^-k) scale. Self-inverse up to that 1/D. Combinational.
// D must be a power of two. (WHT value rotation: Abhiram Bandi + Chaithu Talasila.)
`timescale 1ns/1ps

module wht_unit #(
    parameter int D  = 128,
    parameter int DW = 16
)(
    input  wire [D*DW-1:0] in_vec,      // D fp16 words
    output wire [D*DW-1:0] out_vec      // D fp16 words (raw WHT)
);
    import cq_fp_pkg::*;
    localparam int STAGES = $clog2(D);

    // node[s] = the D fp16 words after s butterfly stages; node[0]=in, node[STAGES]=out.
    wire [DW-1:0] node [0:STAGES][0:D-1];

    genvar s, i, j;
    generate
        for (i = 0; i < D; i = i + 1) begin: g_in
            assign node[0][i] = in_vec[i*DW +: DW];
        end
        for (s = 0; s < STAGES; s = s + 1) begin: g_stage
            localparam int H = (1 << s);
            for (i = 0; i < D; i = i + (H << 1)) begin: g_blk
                for (j = 0; j < H; j = j + 1) begin: g_bf
                    // butterfly pair (i+j, i+j+H)
                    assign node[s+1][i+j]     = fp16_add(node[s][i+j], node[s][i+j+H]);
                    assign node[s+1][i+j+H]   = fp16_sub(node[s][i+j], node[s][i+j+H]);
                end
            end
        end
        for (i = 0; i < D; i = i + 1) begin: g_out
            assign out_vec[i*DW +: DW] = node[STAGES][i];
        end
    endgenerate
endmodule
