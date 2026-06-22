// scale_bank.sv — quantization-scale storage for ChannelQuant
//
// SKELETON / WIP — not yet wired into RTL_SRC or the top-level. Part of the
// TurboQuant+ -> ChannelQuant revamp. See findings/channelquant_block_revamp.md
// (§2, §3) and ../channelquant/REVAMP_SPEC.md (§3).
//
// Holds the per-axis scales that decompress multiplies back in:
//   - KEY path: DIM per-channel scales, frozen per group (written from amax_unit
//     at group_done, read at decompress for the whole group).
//   - VALUE path: one per-token scale, pushed per token (FIFO), popped at
//     decompress in token order.
//
// Scale numeric format pinned by ../channelquant/docs/HW_CONTRACT.md.
//
// TODO(P1): per-token scale FIFO (value path).
// TODO(P2): DIM-entry per-channel register bank with group double-buffer so the
//           next group can accumulate while the current group decompresses.

`default_nettype none

module scale_bank #(
    parameter int DIM   = 64,   // head_dim D -> per-channel bank depth
    parameter int DW    = 16,   // scale width
    parameter int DEPTH = 16    // per-token scale FIFO depth (>= SRAM token capacity served)
) (
    input  wire                clk,
    input  wire                rst_n,

    // per-channel (KEY) write: frozen group scales from amax_unit
    input  wire                chan_wr,
    input  wire [DIM*DW-1:0]   chan_scales,
    // per-channel read: indexed at decompress
    input  wire [$clog2(DIM)-1:0] chan_rd_idx,
    output wire [DW-1:0]       chan_scale,

    // per-token (VALUE) push/pop
    input  wire                tok_push,
    input  wire [DW-1:0]       tok_scale_in,
    input  wire                tok_pop,
    output wire [DW-1:0]       tok_scale_out
);

    // --- skeleton: defaults until P1/P2 implements the banks ---
    assign chan_scale    = '0;
    assign tok_scale_out = '0;

    // pragma: ChannelQuant scale_bank is a skeleton; replace before RTL_SRC inclusion.

endmodule

`default_nettype wire
