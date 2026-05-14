// qjl_unit.sv — Quantized Johnson-Lindenstrauss projection unit

module qjl_unit #(
    parameter integer VECTOR_DIM  = 64,
    parameter integer COORD_WIDTH = 16,
    parameter integer NORM_WIDTH  = 16,
    parameter integer COORD_FRAC  = 12,
    parameter integer NORM_FRAC   = 8,
    parameter integer SEED        = 42
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    i_valid,
    input  wire signed [COORD_WIDTH-1:0] i_residual [0:VECTOR_DIM-1],

    output reg                     o_valid,
    output reg  [0:VECTOR_DIM-1]   o_signs,
    output reg  [NORM_WIDTH-1:0]   o_res_norm
);

    localparam integer LOG2_DIM = $clog2(VECTOR_DIM);
    localparam integer DOT_WIDTH = COORD_WIDTH + LOG2_DIM + 1;
    localparam integer SQ_WIDTH = 2 * COORD_WIDTH + LOG2_DIM;
    localparam integer RESULT_WIDTH = SQ_WIDTH / 2;
    localparam integer FRAC_SHIFT = COORD_FRAC - NORM_FRAC;

    // QJL matrix
    reg qjl_matrix [0:VECTOR_DIM-1][0:VECTOR_DIM-1];
    initial begin : init_qjl
        longint unsigned state;
        integer ii, jj;
        state = SEED + 64'hDEADBEEF;
        for (ii = 0; ii < VECTOR_DIM; ii = ii + 1) begin
            for (jj = 0; jj < VECTOR_DIM; jj = jj + 1) begin
                state = state * 64'h5851F42D4C957F2D + 64'h14057B7EF767814F;
                qjl_matrix[ii][jj] = (state[63:32] & 1) == 0 ? 1'b0 : 1'b1;
            end
        end
    end

    // State machine
    reg                    busy;
    reg [$clog2(VECTOR_DIM):0] row_idx;
    reg signed [COORD_WIDTH-1:0] residual_buf [0:VECTOR_DIM-1];
    reg [0:VECTOR_DIM-1]   signs_buf;

    // Norm computation
    reg [SQ_WIDTH-1:0] sum_sq;
    reg                norm_computed;
    reg [NORM_WIDTH-1:0] computed_norm;

    // Sqrt state
    reg                sqrt_busy;
    reg [SQ_WIDTH-1:0] sqrt_val;
    reg [RESULT_WIDTH-1:0] sqrt_result;
    reg [RESULT_WIDTH-1:0] sqrt_bit;
    reg                signs_done;

    // Dot product working variable
    reg signed [DOT_WIDTH-1:0] dot_acc;
    integer jj;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy          <= 1'b0;
            row_idx       <= 0;
            o_valid       <= 1'b0;
            o_signs       <= 0;
            o_res_norm    <= 0;
            signs_buf     <= 0;
            sum_sq        <= 0;
            norm_computed <= 1'b0;
            sqrt_busy     <= 1'b0;
            sqrt_val      <= 0;
            sqrt_result   <= 0;
            sqrt_bit      <= 0;
            signs_done    <= 1'b0;
            computed_norm <= 0;
        end else begin
            o_valid <= 1'b0;

            if (!busy && i_valid) begin
                busy       <= 1'b1;
                row_idx    <= 0;
                signs_buf  <= 0;
                signs_done <= 1'b0;
                norm_computed <= 1'b0;

                sum_sq <= 0;
                for (jj = 0; jj < VECTOR_DIM; jj = jj + 1) begin
                    residual_buf[jj] <= i_residual[jj];
                    sum_sq <= sum_sq + ($signed(i_residual[jj]) * $signed(i_residual[jj]));
                end
            end else if (busy && !signs_done) begin
                // One projection per cycle
                dot_acc = 0;
                for (jj = 0; jj < VECTOR_DIM; jj = jj + 1) begin
                    if (qjl_matrix[row_idx][jj])
                        dot_acc = dot_acc - $signed(residual_buf[jj]);
                    else
                        dot_acc = dot_acc + $signed(residual_buf[jj]);
                end
                signs_buf[row_idx] <= (dot_acc >= 0) ? 1'b1 : 1'b0;

                if (row_idx == VECTOR_DIM - 1) begin
                    signs_done <= 1'b1;
                    sqrt_val    <= sum_sq;
                    sqrt_result <= 0;
                    sqrt_bit    <= 1 << (RESULT_WIDTH - 1);
                    sqrt_busy   <= 1'b1;
                end else begin
                    row_idx <= row_idx + 1;
                end
            end else if (sqrt_busy) begin
                if (sqrt_bit == 0) begin
                    if (FRAC_SHIFT > 0)
                        computed_norm <= sqrt_result[RESULT_WIDTH-1:0] >> FRAC_SHIFT;
                    else
                        computed_norm <= sqrt_result[NORM_WIDTH-1:0];
                    norm_computed <= 1'b1;
                    sqrt_busy     <= 1'b0;
                end else begin
                    if ((sqrt_result | sqrt_bit) * (sqrt_result | sqrt_bit) <= sqrt_val)
                        sqrt_result <= sqrt_result | sqrt_bit;
                    sqrt_bit <= sqrt_bit >> 1;
                end
            end

            if (signs_done && norm_computed) begin
                o_valid    <= 1'b1;
                o_signs    <= signs_buf;
                o_res_norm <= computed_norm;
                busy       <= 1'b0;
                signs_done <= 1'b0;
                norm_computed <= 1'b0;
            end
        end
    end

endmodule
