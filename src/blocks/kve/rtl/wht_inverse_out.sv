// wht_inverse_out.sv — Path B output stage: inverse WHT + x(1/D) on the A·V result.
//
// The CQ-3-rot value tier stores values ROTATED (see cq_wht_value / wht_unit). Because the
// rotation is linear and the attention output is o = Σ_t A[t]·V̂[t], you rotate on write,
// accumulate the weighted sum in ROTATED space (INT8/FP16 P·V, unchanged), and undo the
// rotation ONCE on the output vector — not per cached token. This is that once-per-decode
// stage: it lives on the MatE P·V output, not in the KVE. (WHT value rotation: Abhiram
// Bandi + Chaithu Talasila.) D power of two.
`timescale 1ns/1ps

module wht_inverse_out #(
    parameter int D  = 128,
    parameter int DW = 16
)(
    input  wire [D*DW-1:0] rot_out,     // Σ_t A[t]·V̂rot[t] in rotated space (fp16)
    output wire [D*32-1:0] vhat_out     // attention output V̂ (fp32), rotation undone
);
    import cq_fp_pkg::*;
    wire [D*DW-1:0] y;
    wht_unit #(.D(D), .DW(DW)) u_inv (.in_vec(rot_out), .out_vec(y));   // self-inverse butterfly
    localparam real INV_D = 1.0 / real'(D);                            // exact 2^-k
    genvar d;
    generate
        for (d = 0; d < D; d = d + 1) begin: g_out
            assign vhat_out[d*32 +: 32] = real_to_f32(f16_to_real(y[d*DW +: DW]) * INV_D);
        end
    endgenerate
endmodule
