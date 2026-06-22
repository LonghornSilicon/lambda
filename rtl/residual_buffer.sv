// residual_buffer.sv — in-flight FP16 hold for the streaming per-channel KEY path
//
// SKELETON / WIP — not yet wired into RTL_SRC or the top-level. Part of the
// TurboQuant+ -> ChannelQuant revamp. See findings/channelquant_block_revamp.md
// (§3.2) and ../channelquant/REVAMP_SPEC.md (§3.1).
//
// THE defining new mechanism of the ChannelQuant block. Per-channel key scaling
// needs the per-channel (column) max over a GROUP of G tokens, which is unknown
// until the group fills. So incoming key vectors accumulate here in FP16 until
// the group reaches G tokens; then the group is quantized as a block and this
// buffer clears. On decompress, tokens in the current (not-yet-quantized) group
// are served from this FP16 buffer directly, selected by token index.
//
// G (group size) is pinned by ../channelquant/docs/HW_CONTRACT.md (start 128).
//
// TODO(P2): ring/double-buffer storage, fill counter, group_full pulse, and the
//           decompress-side index select between this buffer and the quantized
//           SRAM payload. Mind the burst when a full group flushes vs the
//           streaming value path (see revamp spec §7 risk 2 — may need
//           double-buffering to avoid stalling writes / starving the ACU read).

`default_nettype none

module residual_buffer #(
    parameter int DIM = 64,    // head_dim D
    parameter int DW  = 16,    // FP16 element width
    parameter int G   = 128    // key group size (tokens)
) (
    input  wire                clk,
    input  wire                rst_n,

    // write side (compress): key vectors stream in
    input  wire                wr_valid,
    input  wire [DIM*DW-1:0]   wr_vec,
    output wire                group_full,   // pulses when G tokens buffered -> quantize the group

    // read side (decompress): serve an in-flight token by index
    input  wire                rd_req,
    input  wire [$clog2(G)-1:0] rd_idx,
    output wire [DIM*DW-1:0]   rd_vec,
    output wire                rd_in_flight  // 1 if rd_idx is still in this buffer (not yet quantized)
);

    // --- skeleton: defaults until P2 implements the buffer + group FSM ---
    assign group_full   = 1'b0;
    assign rd_vec       = '0;
    assign rd_in_flight = 1'b0;

    // pragma: ChannelQuant residual_buffer is a skeleton; replace before RTL_SRC inclusion.

endmodule

`default_nettype wire
