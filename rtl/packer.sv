// packer.sv — Bit-packing/unpacking for SRAM storage

module packer #(
    parameter integer VECTOR_DIM  = 64,
    parameter integer PQ_BITS     = 3,
    parameter integer QJL_BITS    = 1,
    parameter integer NORM_WIDTH  = 16,
    parameter integer COORD_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,

    // Pack key
    input  wire                    pack_k_valid,
    input  wire [NORM_WIDTH-1:0]   pack_k_norm,
    input  wire [PQ_BITS-1:0]      pack_k_indices [0:VECTOR_DIM-1],
    input  wire [NORM_WIDTH-1:0]   pack_k_res_norm,
    input  wire [0:VECTOR_DIM-1]   pack_k_signs,

    output reg                     pack_k_done,
    output reg  [KEY_PACKED_WIDTH-1:0] pack_k_out,

    // Pack value
    input  wire                    pack_v_valid,
    input  wire [NORM_WIDTH-1:0]   pack_v_norm,
    input  wire [PQ_BITS-1:0]      pack_v_indices [0:VECTOR_DIM-1],

    output reg                     pack_v_done,
    output reg  [VAL_PACKED_WIDTH-1:0] pack_v_out,

    // Unpack key
    input  wire                    unpack_k_valid,
    input  wire [KEY_PACKED_WIDTH-1:0] unpack_k_in,

    output reg                     unpack_k_done,
    output reg  [NORM_WIDTH-1:0]   unpack_k_norm,
    output reg  [PQ_BITS-1:0]      unpack_k_indices [0:VECTOR_DIM-1],
    output reg  [NORM_WIDTH-1:0]   unpack_k_res_norm,
    output reg  [0:VECTOR_DIM-1]   unpack_k_signs,

    // Unpack value
    input  wire                    unpack_v_valid,
    input  wire [VAL_PACKED_WIDTH-1:0] unpack_v_in,

    output reg                     unpack_v_done,
    output reg  [NORM_WIDTH-1:0]   unpack_v_norm,
    output reg  [PQ_BITS-1:0]      unpack_v_indices [0:VECTOR_DIM-1]
);

    localparam integer KEY_PACKED_WIDTH = NORM_WIDTH + VECTOR_DIM * PQ_BITS +
                                          NORM_WIDTH + VECTOR_DIM * QJL_BITS;
    localparam integer VAL_PACKED_WIDTH = NORM_WIDTH + VECTOR_DIM * PQ_BITS;

    integer i, pos;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pack_k_done   <= 1'b0;
            pack_v_done   <= 1'b0;
            unpack_k_done <= 1'b0;
            unpack_v_done <= 1'b0;
            pack_k_out    <= 0;
            pack_v_out    <= 0;
        end else begin
            pack_k_done   <= 1'b0;
            pack_v_done   <= 1'b0;
            unpack_k_done <= 1'b0;
            unpack_v_done <= 1'b0;

            // Pack key
            if (pack_k_valid) begin
                pack_k_out[0 +: NORM_WIDTH] <= pack_k_norm;
                for (i = 0; i < VECTOR_DIM; i = i + 1)
                    pack_k_out[NORM_WIDTH + i*PQ_BITS +: PQ_BITS] <= pack_k_indices[i];
                pack_k_out[NORM_WIDTH + VECTOR_DIM*PQ_BITS +: NORM_WIDTH] <= pack_k_res_norm;
                for (i = 0; i < VECTOR_DIM; i = i + 1)
                    pack_k_out[2*NORM_WIDTH + VECTOR_DIM*PQ_BITS + i] <= pack_k_signs[i];
                pack_k_done <= 1'b1;
            end

            // Pack value
            if (pack_v_valid) begin
                pack_v_out[0 +: NORM_WIDTH] <= pack_v_norm;
                for (i = 0; i < VECTOR_DIM; i = i + 1)
                    pack_v_out[NORM_WIDTH + i*PQ_BITS +: PQ_BITS] <= pack_v_indices[i];
                pack_v_done <= 1'b1;
            end

            // Unpack key
            if (unpack_k_valid) begin
                unpack_k_norm <= unpack_k_in[0 +: NORM_WIDTH];
                for (i = 0; i < VECTOR_DIM; i = i + 1)
                    unpack_k_indices[i] <= unpack_k_in[NORM_WIDTH + i*PQ_BITS +: PQ_BITS];
                unpack_k_res_norm <= unpack_k_in[NORM_WIDTH + VECTOR_DIM*PQ_BITS +: NORM_WIDTH];
                for (i = 0; i < VECTOR_DIM; i = i + 1)
                    unpack_k_signs[i] <= unpack_k_in[2*NORM_WIDTH + VECTOR_DIM*PQ_BITS + i];
                unpack_k_done <= 1'b1;
            end

            // Unpack value
            if (unpack_v_valid) begin
                unpack_v_norm <= unpack_v_in[0 +: NORM_WIDTH];
                for (i = 0; i < VECTOR_DIM; i = i + 1)
                    unpack_v_indices[i] <= unpack_v_in[NORM_WIDTH + i*PQ_BITS +: PQ_BITS];
                unpack_v_done <= 1'b1;
            end
        end
    end

endmodule
