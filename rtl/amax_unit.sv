// amax_unit.sv — ChannelQuant per-axis amax reduction (streaming scale front-end)
//
// Replaces norm_unit's L2 norm. Computes the max-absolute over the active scale
// axis (contract §1-§3); the amax feeds cq_scale_unit -> fp16 scale.
//   - VALUE path (mode_channel=0): amax over the DIM elements of the current
//     token, emitted the cycle after in_valid.
//   - KEY path (mode_channel=1): per-channel running max accumulated over a
//     group of tokens; frozen and emitted on group_done (contract §3.1).
//
// SYNTHESIZABLE (no `real`). Key trick: for finite IEEE-754 half, the magnitude
// (sign bit cleared) is a monotonic function of the unsigned 15-bit {exp,man}
// field, so max|fp16| == unsigned-integer max of {1'b0, x[14:0]}. Verified
// bit-exact vs the golden per-token/per-channel scales by tb_amax_unit.sv (all 9
// vectors) — its output drives the proven cq_scale_unit.

`default_nettype none

module amax_unit #(
    parameter int DIM = 64,   // head_dim D
    parameter int DW  = 16    // element width (fp16)
) (
    input  wire                   clk,
    input  wire                   rst_n,

    input  wire                   in_valid,      // a token vector is presented
    input  wire [DIM*DW-1:0]      vec,           // DIM elements, element d at [d*DW +: DW]
    input  wire                   mode_channel,  // 0 = per-token (V), 1 = per-channel (K)
    input  wire                   group_start,   // (mode_channel) first token of a group
    input  wire                   group_done,    // (mode_channel) last token of a group -> emit

    output wire [DW-1:0]          scale_token,   // per-token amax (mode 0)
    output wire [DIM*DW-1:0]      scale_chan,    // per-channel amax (mode 1, valid at group_done)
    output reg                    out_valid
);

    // ---- per-element magnitude (sign bit cleared): {0, x[DW-2:0]} -------------
    wire [DW-1:0] magv [0:DIM-1];
    genvar gm;
    generate
        for (gm = 0; gm < DIM; gm = gm + 1) begin : g_mag
            assign magv[gm] = {1'b0, vec[gm*DW + DW-2 -: DW-1]};
        end
    endgenerate

    // ---- combinational per-token amax over DIM elements ----------------------
    reg [DW-1:0] tok_amax_c;
    integer j;
    always @* begin
        tok_amax_c = magv[0];
        for (j = 1; j < DIM; j = j + 1)
            if (magv[j] > tok_amax_c) tok_amax_c = magv[j];
    end

    // ---- per-channel running max (key path) ----------------------------------
    reg [DW-1:0] chan_max    [0:DIM-1];   // accumulating group max
    reg [DW-1:0] chan_frozen [0:DIM-1];   // frozen at group_done
    reg [DW-1:0] tok_reg;                 // registered per-token amax
    reg [DW-1:0] nm;                      // scratch: this channel's next max
    integer c;

    // The next-max is computed inline in the sequential block (a blocking scratch,
    // not a combinational unpacked array) so yosys does not mem2reg it — that
    // conversion left constant-conflicting DFF drivers that fail LibreLane's
    // pre-opt synth-check (benign, but counted).
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            tok_reg   <= '0;
            for (c = 0; c < DIM; c = c + 1) begin
                chan_max[c]    <= '0;
                chan_frozen[c] <= '0;
            end
        end else begin
            out_valid <= 1'b0;
            if (!mode_channel) begin
                // value path: per-token amax, one-cycle latency
                if (in_valid) begin
                    tok_reg   <= tok_amax_c;
                    out_valid <= 1'b1;
                end
            end else begin
                // key path: accumulate per-channel max; freeze + emit on group_done
                for (c = 0; c < DIM; c = c + 1) begin
                    if (in_valid && group_start) nm = magv[c];
                    else if (in_valid)           nm = (magv[c] > chan_max[c]) ? magv[c] : chan_max[c];
                    else                         nm = chan_max[c];
                    chan_max[c] <= nm;
                    if (group_done) chan_frozen[c] <= nm;
                end
                if (group_done) out_valid <= 1'b1;
            end
        end
    end

    assign scale_token = tok_reg;
    genvar g;
    generate
        for (g = 0; g < DIM; g = g + 1) begin : g_out
            assign scale_chan[g*DW +: DW] = chan_frozen[g];
        end
    endgenerate

endmodule

`default_nettype wire
