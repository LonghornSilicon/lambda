// quantizer.sv — 3-bit PolarQuant nearest-centroid quantizer

module quantizer #(
    parameter integer COORD_WIDTH = 16,
    parameter integer PQ_BITS     = 3,
    parameter integer VECTOR_DIM  = 64,
    parameter integer COORD_FRAC  = 12
) (
    input  wire                    clk,
    input  wire                    rst_n,

    input  wire                    i_valid,
    input  wire signed [COORD_WIDTH-1:0] i_coord,

    output reg                     o_valid,
    output reg  [PQ_BITS-1:0]      o_index,
    output reg signed [COORD_WIDTH-1:0] o_centroid
);

    localparam integer NUM_CENTROIDS  = 1 << PQ_BITS;
    localparam integer NUM_BOUNDARIES = NUM_CENTROIDS - 1;

    reg signed [COORD_WIDTH-1:0] centroids  [0:NUM_CENTROIDS-1];
    reg signed [COORD_WIDTH-1:0] boundaries [0:NUM_BOUNDARIES-1];

    function automatic signed [COORD_WIDTH-1:0] to_fixed(input real val);
        if (val < 0)
            return $rtoi(val * (1 << COORD_FRAC) - 0.5);
        else
            return $rtoi(val * (1 << COORD_FRAC) + 0.5);
    endfunction

    initial begin : init_tables
        real sigma;
        sigma = 1.0 / $sqrt($itor(VECTOR_DIM));

        centroids[0] = to_fixed(-2.1520 * sigma);
        centroids[1] = to_fixed(-1.3439 * sigma);
        centroids[2] = to_fixed(-0.7560 * sigma);
        centroids[3] = to_fixed(-0.2451 * sigma);
        centroids[4] = to_fixed( 0.2451 * sigma);
        centroids[5] = to_fixed( 0.7560 * sigma);
        centroids[6] = to_fixed( 1.3439 * sigma);
        centroids[7] = to_fixed( 2.1520 * sigma);

        boundaries[0] = to_fixed(-1.7480 * sigma);
        boundaries[1] = to_fixed(-1.0500 * sigma);
        boundaries[2] = to_fixed(-0.5006 * sigma);
        boundaries[3] = to_fixed( 0.0000 * sigma);
        boundaries[4] = to_fixed( 0.5006 * sigma);
        boundaries[5] = to_fixed( 1.0500 * sigma);
        boundaries[6] = to_fixed( 1.7480 * sigma);
    end

    // Combinational index computation
    reg [PQ_BITS-1:0] idx_comb;
    integer b;
    always @(*) begin
        idx_comb = 0;
        for (b = 0; b < NUM_BOUNDARIES; b = b + 1) begin
            if ($signed(i_coord) >= $signed(boundaries[b]))
                idx_comb = b + 1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_valid    <= 1'b0;
            o_index    <= 0;
            o_centroid <= 0;
        end else begin
            o_valid <= i_valid;
            if (i_valid) begin
                o_index    <= idx_comb;
                o_centroid <= centroids[idx_comb];
            end
        end
    end

endmodule
