// kv_sram.sv — KV-store memory array behind a clean, PDK-agnostic interface.
//
// This is the *swappable* storage element of the KV cache. The DEFAULT
// implementation here is a behavioral synchronous memory (a reg array), which
// keeps simulation and non-GF180 flows working exactly as before. On GF180 the
// chipathon build (LonghornSilicon/chipathon-lambda-acu) replaces this module
// with a wrapper that tiles the real `gf180mcu_fd_ip_sram` hard macro to the
// same interface — so `sram_controller` (and the whole KVE) is unchanged.
//
// INTERFACE (single-port in practice — see note):
//   write : we,  waddr, wdata
//   read  : re,  raddr -> rdata   (REGISTERED, 1-cycle latency)
//
// Faithful-SRAM semantics (so the behavioral model matches the macro):
//   * Registered read: `rdata` appears the cycle AFTER `re` (like the macro's Q).
//   * NO reset of the array or `rdata` — real SRAM powers up undefined. The KVE
//     only reads addresses it has written (sram_controller gates reads on
//     `valid_bits`), so uninitialized storage is never observed.
//   * Single-port safe: the KVE FSM writes (ST_STORE) and reads (ST_RLOAD) in
//     distinct states, never the same cycle, so a real single-port macro maps
//     directly (the GF180 wrapper muxes waddr/raddr onto the one macro port).
//
// FF/area note: with the default behavioral impl this synthesizes to a
// DEPTH*WIDTH flip-flop array (the "flop-array proxy"); the GF180 macro build
// replaces it with real 6T-bitcell SRAM.

`default_nettype none

module kv_sram #(
    parameter integer DEPTH = 16,
    parameter integer WIDTH = 288,
    parameter integer AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire              clk,

    // write port
    input  wire              we,
    input  wire [AW-1:0]     waddr,
    input  wire [WIDTH-1:0]  wdata,

    // read port (registered)
    input  wire              re,
    input  wire [AW-1:0]     raddr,
    output reg  [WIDTH-1:0]  rdata
);

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;
        if (re)
            rdata <= mem[raddr];
    end

endmodule

`default_nettype wire
