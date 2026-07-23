// kv_sram.sv — GF180 hard-macro implementation of the KV-store memory.
//
// Drop-in replacement for the behavioral `kv_sram` in the kv-cache-engine repo
// (rtl/blocks/kve/kv_sram.sv): SAME module name + ports, so `sram_controller`
// (and the whole KVE) is unchanged. The GF180 build compiles THIS file instead
// of the behavioral one, so the KV store lands on a REAL gf180mcu_fd_ip_sram
// hard macro (6T bitcell SRAM) instead of a flip-flop register array.
//
// TILING: the GF180 open SRAM IP is single-port, synchronous, 512 words × 8 bits
// (gf180mcu_fd_ip_sram__sram512x8m8wm1, registered Q). A WIDTH-bit × DEPTH-word
// KV store is built by placing NB = ceil(WIDTH/8) banks side by side (byte
// lanes), all sharing one address/control. DEPTH must be ≤ 512 (one bank deep);
// that is real KV capacity — 512 cached tokens — vs the depth-2 flop proxy.
//
// SINGLE-PORT MAPPING: the KVE FSM writes (ST_STORE) and reads (ST_RLOAD) in
// distinct cycles, never both at once (the invariant the behavioral model also
// relies on), so the two logical ports mux safely onto the one macro port:
//   CEN  = ~(we|re)   chip-enable (active-low): asserted on any access
//   GWEN = ~we        global write-enable (active-low): low = write
//   WEN  = we?0:FF    per-bit write-enable (active-low): all-write on writes
//   A    = we?waddr:raddr
//   D    = wdata (byte lane),  Q -> rdata (registered, 1-cycle — matches behav.)
//
// Read latency (1 cycle) equals the behavioral model's registered read, so
// sram_controller's rd_valid handshake timing is unchanged.

`default_nettype none

module kv_sram #(
    parameter integer DEPTH = 512,
    parameter integer WIDTH = 80,
    parameter integer AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire              clk,

    // write port
    input  wire              we,
    input  wire [AW-1:0]     waddr,
    input  wire [WIDTH-1:0]  wdata,

    // read port (registered, 1-cycle)
    input  wire              re,
    input  wire [AW-1:0]     raddr,
    output wire [WIDTH-1:0]  rdata

    `ifdef USE_POWER_PINS
    , inout wire VDD
    , inout wire VSS
    `endif
);

    // number of 8-bit byte-lane banks
    localparam integer NB = (WIDTH + 7) / 8;

    // single-port control mux (write and read never collide — see header)
    wire              access = we | re;
    wire              cen    = ~access;          // active-low chip enable
    wire              gwen   = ~we;              // active-low global write enable
    wire [7:0]        wen    = we ? 8'h00 : 8'hFF; // active-low per-bit write enable
    wire [AW-1:0]     addr   = we ? waddr : raddr;
    wire [8:0]        a9     = {{(9-AW){1'b0}}, addr};   // AW<=9 (DEPTH<=512)

    // pad wdata / collect Q to a byte-aligned bus
    wire [NB*8-1:0]   d_bus  = {{(NB*8-WIDTH){1'b0}}, wdata};
    wire [NB*8-1:0]   q_bus;
    assign rdata = q_bus[WIDTH-1:0];

    genvar gi;
    generate
        for (gi = 0; gi < NB; gi = gi + 1) begin : lane
            gf180mcu_fd_ip_sram__sram512x8m8wm1 u_bank (
                `ifdef USE_POWER_PINS
                .VDD (VDD),
                .VSS (VSS),
                `endif
                .CLK (clk),
                .CEN (cen),
                .GWEN(gwen),
                .WEN (wen),
                .A   (a9),
                .D   (d_bus[gi*8 +: 8]),
                .Q   (q_bus[gi*8 +: 8])
            );
        end
    endgenerate

endmodule

`default_nettype wire
