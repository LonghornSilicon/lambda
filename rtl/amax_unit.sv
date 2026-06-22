// amax_unit.sv — ChannelQuant scale extraction (replaces norm_unit's L2 norm)
//
// SKELETON / WIP — not yet wired into RTL_SRC or the top-level. Part of the
// TurboQuant+ -> ChannelQuant revamp. See findings/channelquant_block_revamp.md
// (§3) and ../channelquant/REVAMP_SPEC.md (§3).
//
// Computes the quantization scale as a per-axis absolute maximum:
//   - VALUE path (per-token): amax over the DIM elements of the current token.
//   - KEY path (per-channel): running per-channel amax accumulated over a group
//     of G tokens; frozen and emitted when the group fills.
//
// Exact numeric format (fixed-point vs fp16 scale), rounding, and clamp range
// are pinned by ../channelquant/docs/HW_CONTRACT.md — implement against it, do
// not invent.
//
// TODO(P1): per-token amax reduction tree (value path).
// TODO(P2): per-channel running-max bank + group_done freeze (key path).

`default_nettype none

module amax_unit #(
    parameter int DIM = 64,   // head_dim D (parameterize — old block hardcoded 64)
    parameter int DW  = 16    // element width (input activation, e.g. Q4.12)
) (
    input  wire                   clk,
    input  wire                   rst_n,

    input  wire                   in_valid,      // a token vector is presented
    input  wire [DIM*DW-1:0]      vec,           // DIM elements, packed
    input  wire                   mode_channel,  // 0 = per-token (V), 1 = per-channel (K)
    input  wire                   group_start,   // (mode_channel) reset the running max
    input  wire                   group_done,    // (mode_channel) freeze + emit channel scales

    output wire [DW-1:0]          scale_token,   // per-token amax (mode 0)
    output wire [DIM*DW-1:0]      scale_chan,    // per-channel amax (mode 1, valid at group_done)
    output wire                   out_valid
);

    // --- skeleton: hold outputs at reset until P1/P2 implements the reductions ---
    // The real implementation tracks max(|vec[i]|) per the active mode.
    assign scale_token = '0;
    assign scale_chan  = '0;
    assign out_valid   = 1'b0;

    // synthesis-time guard so the WIP module is obvious if accidentally elaborated
    // pragma: ChannelQuant amax_unit is a skeleton; replace before RTL_SRC inclusion.

endmodule

`default_nettype wire
