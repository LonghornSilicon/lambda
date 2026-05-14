// rotation_unit.sv — Walsh-Hadamard Transform with random sign flips

module rotation_unit #(
    parameter integer VECTOR_DIM  = 64,
    parameter integer COORD_WIDTH = 16,
    parameter integer SEED        = 42
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    i_valid,
    input  wire                    i_inverse,
    input  wire signed [COORD_WIDTH-1:0] i_data [0:VECTOR_DIM-1],

    output reg                     o_valid,
    output reg  signed [COORD_WIDTH-1:0] o_data [0:VECTOR_DIM-1]
);

    localparam integer LOG2_DIM = $clog2(VECTOR_DIM);
    localparam integer WHT_WIDTH = COORD_WIDTH + LOG2_DIM;
    localparam integer FWD_SHIFT = LOG2_DIM / 2;
    localparam integer INV_SHIFT = LOG2_DIM - FWD_SHIFT;

    // Sign flips from seed (LCG)
    reg sign_flips [0:VECTOR_DIM-1];
    initial begin : init_signs
        longint unsigned state;
        integer ii;
        state = SEED;
        for (ii = 0; ii < VECTOR_DIM; ii = ii + 1) begin
            state = state * 64'h5851F42D4C957F2D + 64'h14057B7EF767814F;
            sign_flips[ii] = (state[63:32] & 1) == 0 ? 1'b0 : 1'b1;
        end
    end

    // Pipeline: stage 0 = sign flip, stage 1 = WHT + shift + output
    reg                     p0_valid;
    reg                     p0_inverse;
    reg signed [WHT_WIDTH-1:0] p0_data [0:VECTOR_DIM-1];

    // WHT working buffers (module-level)
    reg signed [WHT_WIDTH+1:0] wht_buf [0:VECTOR_DIM-1];
    reg signed [WHT_WIDTH+1:0] wht_a, wht_b;

    integer i, j, s, half;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p0_valid <= 1'b0;
            o_valid  <= 1'b0;
            for (i = 0; i < VECTOR_DIM; i = i + 1) begin
                p0_data[i] <= 0;
                o_data[i]  <= 0;
            end
        end else begin
            // Stage 0: sign flips (forward) or passthrough (inverse)
            p0_valid   <= i_valid;
            p0_inverse <= i_inverse;
            if (i_valid) begin
                for (i = 0; i < VECTOR_DIM; i = i + 1) begin
                    if (!i_inverse && sign_flips[i])
                        p0_data[i] <= -$signed({{(WHT_WIDTH-COORD_WIDTH){i_data[i][COORD_WIDTH-1]}}, i_data[i]});
                    else
                        p0_data[i] <= $signed({{(WHT_WIDTH-COORD_WIDTH){i_data[i][COORD_WIDTH-1]}}, i_data[i]});
                end
            end

            // Stage 1: WHT butterfly + shift + inverse sign flips
            o_valid <= p0_valid;
            if (p0_valid) begin
                // Load buffer
                for (i = 0; i < VECTOR_DIM; i = i + 1)
                    wht_buf[i] = p0_data[i];

                // Butterfly stages
                for (s = 0; s < LOG2_DIM; s = s + 1) begin
                    half = 1 << s;
                    for (i = 0; i < VECTOR_DIM; i = i + (half * 2)) begin
                        for (j = i; j < i + half; j = j + 1) begin
                            wht_a = wht_buf[j];
                            wht_b = wht_buf[j + half];
                            wht_buf[j]        = wht_a + wht_b;
                            wht_buf[j + half] = wht_a - wht_b;
                        end
                    end
                end

                // Shift and optional inverse sign flips
                for (i = 0; i < VECTOR_DIM; i = i + 1) begin
                    if (!p0_inverse) begin
                        wht_a = (wht_buf[i] + (1 <<< (FWD_SHIFT - 1))) >>> FWD_SHIFT;
                        o_data[i] <= wht_a[COORD_WIDTH-1:0];
                    end else begin
                        wht_a = (wht_buf[i] + (1 <<< (INV_SHIFT - 1))) >>> INV_SHIFT;
                        if (sign_flips[i])
                            o_data[i] <= -wht_a[COORD_WIDTH-1:0];
                        else
                            o_data[i] <= wht_a[COORD_WIDTH-1:0];
                    end
                end
            end
        end
    end

endmodule
