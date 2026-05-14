// norm_unit.sv — L2 norm computation (sum of squares → integer sqrt)
//
// FF count (closed-form):
//   input_reg:     VECTOR_DIM * COORD_WIDTH
//   sq_accum:      SQ_WIDTH
//   sqrt_state:    2 * RESULT_WIDTH + 2
//   output_reg:    NORM_WIDTH
//   control:       4
//   Total = VECTOR_DIM * COORD_WIDTH + SQ_WIDTH + 2*RESULT_WIDTH + NORM_WIDTH + 6

module norm_unit #(
    parameter integer VECTOR_DIM  = 64,
    parameter integer COORD_WIDTH = 16,
    parameter integer NORM_WIDTH  = 16,
    parameter integer COORD_FRAC  = 12,
    parameter integer NORM_FRAC   = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    i_valid,
    input  wire signed [COORD_WIDTH-1:0] i_data,
    input  wire                    i_last,

    output reg                     o_valid,
    output reg  [NORM_WIDTH-1:0]   o_norm
);

    localparam integer LOG2_DIM    = $clog2(VECTOR_DIM);
    localparam integer SQ_WIDTH    = 2 * COORD_WIDTH + LOG2_DIM;
    localparam integer RESULT_WIDTH = SQ_WIDTH / 2;
    localparam integer FRAC_SHIFT  = COORD_FRAC - NORM_FRAC;

    reg [SQ_WIDTH-1:0]  sum_sq;
    reg [$clog2(VECTOR_DIM):0] count;

    // Non-restoring integer square root
    reg                sqrt_busy;
    reg [SQ_WIDTH-1:0] sqrt_val;
    reg [RESULT_WIDTH-1:0] sqrt_result;
    reg [RESULT_WIDTH-1:0] sqrt_bit;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_sq      <= '0;
            count       <= '0;
            o_valid     <= 1'b0;
            o_norm      <= '0;
            sqrt_busy   <= 1'b0;
            sqrt_val    <= '0;
            sqrt_result <= '0;
            sqrt_bit    <= '0;
        end else begin
            o_valid <= 1'b0;

            if (sqrt_busy) begin
                if (sqrt_bit == '0) begin
                    // Sqrt complete — apply fractional shift
                    if (FRAC_SHIFT > 0)
                        o_norm <= sqrt_result[RESULT_WIDTH-1:0] >> FRAC_SHIFT;
                    else
                        o_norm <= sqrt_result[NORM_WIDTH-1:0];
                    o_valid    <= 1'b1;
                    sqrt_busy  <= 1'b0;
                    sum_sq     <= '0;
                    count      <= '0;
                end else begin
                    if ((sqrt_result | sqrt_bit) * (sqrt_result | sqrt_bit) <= sqrt_val)
                        sqrt_result <= sqrt_result | sqrt_bit;
                    sqrt_bit <= sqrt_bit >> 1;
                end
            end else if (i_valid) begin
                sum_sq <= sum_sq + ($signed(i_data) * $signed(i_data));
                count  <= count + 1;

                if (i_last) begin
                    sqrt_val    <= sum_sq + ($signed(i_data) * $signed(i_data));
                    sqrt_result <= '0;
                    sqrt_bit    <= 1 << (RESULT_WIDTH - 1);
                    sqrt_busy   <= 1'b1;
                end
            end
        end
    end

endmodule
