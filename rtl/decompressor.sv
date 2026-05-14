// decompressor.sv — PolarQuant + QJL decompression pipeline

module decompressor #(
    parameter integer VECTOR_DIM  = 64,
    parameter integer COORD_WIDTH = 16,
    parameter integer COORD_FRAC  = 12,
    parameter integer PQ_BITS     = 3,
    parameter integer NORM_WIDTH  = 16,
    parameter integer NORM_FRAC   = 8,
    parameter integer SEED        = 42
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    i_valid,
    input  wire                    i_is_key,
    input  wire [NORM_WIDTH-1:0]   i_norm,
    input  wire [PQ_BITS-1:0]      i_indices [0:VECTOR_DIM-1],
    input  wire [NORM_WIDTH-1:0]   i_res_norm,
    input  wire [0:VECTOR_DIM-1]   i_signs,

    output reg                     o_valid,
    output reg signed [COORD_WIDTH-1:0] o_data [0:VECTOR_DIM-1]
);

    localparam integer LOG2_DIM   = $clog2(VECTOR_DIM);
    localparam integer WHT_WIDTH  = COORD_WIDTH + LOG2_DIM;
    localparam integer INV_SHIFT  = LOG2_DIM - (LOG2_DIM / 2);
    localparam integer DOT_WIDTH  = COORD_WIDTH + LOG2_DIM + 1;
    localparam integer NUM_CENTROIDS = 1 << PQ_BITS;
    localparam integer MUL_WIDTH  = COORD_WIDTH + NORM_WIDTH;
    localparam integer FRAC_SHIFT = COORD_FRAC + NORM_FRAC - COORD_FRAC; // = NORM_FRAC

    // Centroids
    reg signed [COORD_WIDTH-1:0] centroids [0:NUM_CENTROIDS-1];

    // Sign flips
    reg sign_flips [0:VECTOR_DIM-1];

    // QJL matrix
    reg qjl_matrix [0:VECTOR_DIM-1][0:VECTOR_DIM-1];

    // sqrt(pi/2)
    reg signed [COORD_WIDTH-1:0] sqrt_pi_2;

    function automatic signed [COORD_WIDTH-1:0] to_fixed(input real val);
        if (val < 0)
            return $rtoi(val * (1 << COORD_FRAC) - 0.5);
        else
            return $rtoi(val * (1 << COORD_FRAC) + 0.5);
    endfunction

    initial begin : init_tables
        real sigma;
        longint unsigned state;
        integer ii, jj;

        sigma = 1.0 / $sqrt($itor(VECTOR_DIM));

        centroids[0] = to_fixed(-2.1520 * sigma);
        centroids[1] = to_fixed(-1.3439 * sigma);
        centroids[2] = to_fixed(-0.7560 * sigma);
        centroids[3] = to_fixed(-0.2451 * sigma);
        centroids[4] = to_fixed( 0.2451 * sigma);
        centroids[5] = to_fixed( 0.7560 * sigma);
        centroids[6] = to_fixed( 1.3439 * sigma);
        centroids[7] = to_fixed( 2.1520 * sigma);

        sqrt_pi_2 = to_fixed(1.2533141);

        state = SEED;
        for (ii = 0; ii < VECTOR_DIM; ii = ii + 1) begin
            state = state * 64'h5851F42D4C957F2D + 64'h14057B7EF767814F;
            sign_flips[ii] = (state[63:32] & 1) == 0 ? 1'b0 : 1'b1;
        end

        state = SEED + 64'hDEADBEEF;
        for (ii = 0; ii < VECTOR_DIM; ii = ii + 1) begin
            for (jj = 0; jj < VECTOR_DIM; jj = jj + 1) begin
                state = state * 64'h5851F42D4C957F2D + 64'h14057B7EF767814F;
                qjl_matrix[ii][jj] = (state[63:32] & 1) == 0 ? 1'b0 : 1'b1;
            end
        end
    end

    // Pipeline registers
    reg                    s1_valid;
    reg [NORM_WIDTH-1:0]   s1_norm;
    reg signed [COORD_WIDTH-1:0] s1_corrected [0:VECTOR_DIM-1];

    reg                    s2_valid;
    reg [NORM_WIDTH-1:0]   s2_norm;
    reg signed [COORD_WIDTH-1:0] s2_unrotated [0:VECTOR_DIM-1];

    // Working variables
    reg signed [MUL_WIDTH-1:0] scale_num;
    reg signed [COORD_WIDTH-1:0] scale;
    reg signed [DOT_WIDTH-1:0] qjl_acc;
    reg signed [MUL_WIDTH-1:0] correction;
    reg signed [WHT_WIDTH+1:0] wht_buf [0:VECTOR_DIM-1];
    reg signed [WHT_WIDTH+1:0] wht_a, wht_b;
    reg signed [MUL_WIDTH-1:0] product;

    integer i, j, s, half;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s2_valid <= 1'b0;
            o_valid  <= 1'b0;
            for (i = 0; i < VECTOR_DIM; i = i + 1) begin
                s1_corrected[i] <= 0;
                s2_unrotated[i] <= 0;
                o_data[i] <= 0;
            end
        end else begin

            // Stage 1: Centroid lookup + QJL correction
            s1_valid <= i_valid;
            s1_norm  <= i_norm;
            if (i_valid) begin
                scale_num = ($signed(sqrt_pi_2) * $signed({1'b0, i_res_norm}));
                if (FRAC_SHIFT > 0)
                    scale_num = (scale_num + (1 <<< (FRAC_SHIFT - 1))) >>> FRAC_SHIFT;
                scale = scale_num / $signed(VECTOR_DIM);

                for (i = 0; i < VECTOR_DIM; i = i + 1) begin
                    s1_corrected[i] <= centroids[i_indices[i]];

                    if (i_is_key) begin
                        qjl_acc = 0;
                        for (j = 0; j < VECTOR_DIM; j = j + 1) begin
                            if (qjl_matrix[j][i])
                                qjl_acc = i_signs[j] ? qjl_acc - 1 : qjl_acc + 1;
                            else
                                qjl_acc = i_signs[j] ? qjl_acc + 1 : qjl_acc - 1;
                        end
                        correction = qjl_acc * $signed(scale);
                        s1_corrected[i] <= centroids[i_indices[i]] +
                                           correction[COORD_WIDTH-1:0];
                    end
                end
            end

            // Stage 2: Inverse WHT + sign flips
            s2_valid <= s1_valid;
            s2_norm  <= s1_norm;
            if (s1_valid) begin
                for (i = 0; i < VECTOR_DIM; i = i + 1)
                    wht_buf[i] = $signed({{(WHT_WIDTH-COORD_WIDTH+2){s1_corrected[i][COORD_WIDTH-1]}},
                                          s1_corrected[i]});

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

                for (i = 0; i < VECTOR_DIM; i = i + 1) begin
                    wht_a = (wht_buf[i] + (1 <<< (INV_SHIFT - 1))) >>> INV_SHIFT;
                    if (sign_flips[i])
                        s2_unrotated[i] <= -wht_a[COORD_WIDTH-1:0];
                    else
                        s2_unrotated[i] <= wht_a[COORD_WIDTH-1:0];
                end
            end

            // Stage 3: Norm rescaling
            o_valid <= s2_valid;
            if (s2_valid) begin
                for (i = 0; i < VECTOR_DIM; i = i + 1) begin
                    product = $signed(s2_unrotated[i]) * $signed({1'b0, s2_norm});
                    if (FRAC_SHIFT > 0)
                        product = (product + (1 <<< (FRAC_SHIFT - 1))) >>> FRAC_SHIFT;
                    o_data[i] <= product[COORD_WIDTH-1:0];
                end
            end
        end
    end

endmodule
