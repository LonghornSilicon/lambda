// sram_controller.sv — KV-store control shell around a swappable memory (kv_sram).
//
// The raw storage array now lives in `kv_sram` (behavioral by default; the GF180
// build swaps in a tiled `gf180mcu_fd_ip_sram` hard macro to the same ports).
// This shell keeps the KVE-facing control: occupancy/valid tracking, bounds and
// valid-gating, and the `rd_valid` handshake. Behavior is unchanged from the
// previous inline-array version:
//   * write: mem[wr_addr] <= wr_data   when wr_en & in-range
//   * read : rd_data <= mem[rd_addr], rd_valid pulses one cycle later,
//            only when rd_en & in-range & the slot has been written (valid_bits)
//   * registered (1-cycle) read latency — matches the SRAM macro's Q timing
//
// kv_sram holds no reset (real SRAM powers up undefined); reads are valid-gated
// here so uninitialized storage is never observed. Single-port safe: the top
// FSM writes and reads in distinct states, never the same cycle.
//
// FF count (control only): valid_bits(SRAM_DEPTH) + occupancy(ADDR_WIDTH+1) +
// rd_valid(1). The DEPTH*WIDTH storage is in kv_sram (flop-array proxy in sim /
// real SRAM macro on GF180).

module sram_controller #(
    parameter integer SRAM_DEPTH  = 16,
    parameter integer DATA_WIDTH  = 288,
    parameter integer ADDR_WIDTH  = $clog2(SRAM_DEPTH)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Write port
    input  wire                    wr_en,
    input  wire [ADDR_WIDTH-1:0]   wr_addr,
    input  wire [DATA_WIDTH-1:0]   wr_data,

    // Read port
    input  wire                    rd_en,
    input  wire [ADDR_WIDTH-1:0]   rd_addr,
    output wire [DATA_WIDTH-1:0]   rd_data,
    output reg                     rd_valid,

    // Status
    output reg  [ADDR_WIDTH:0]     occupancy,
    output wire                    full
);

    reg [SRAM_DEPTH-1:0] valid_bits;

    assign full = (occupancy >= SRAM_DEPTH);

    // Qualified memory-access strobes (bounds + valid-gating, as before).
    wire mem_we = wr_en && (wr_addr < SRAM_DEPTH);
    wire mem_re = rd_en && (rd_addr < SRAM_DEPTH) && valid_bits[rd_addr];

    // ---- swappable storage array (behavioral default / GF180 SRAM macro) ----
    kv_sram #(
        .DEPTH (SRAM_DEPTH),
        .WIDTH (DATA_WIDTH),
        .AW    (ADDR_WIDTH)
    ) u_mem (
        .clk   (clk),
        .we    (mem_we),
        .waddr (wr_addr),
        .wdata (wr_data),
        .re    (mem_re),
        .raddr (rd_addr),
        .rdata (rd_data)
    );

    // ---- control: occupancy / valid tracking + rd_valid handshake ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid   <= 1'b0;
            occupancy  <= '0;
            valid_bits <= '0;
        end else begin
            rd_valid <= 1'b0;

            if (mem_we) begin
                if (!valid_bits[wr_addr]) begin
                    valid_bits[wr_addr] <= 1'b1;
                    occupancy <= occupancy + 1;
                end
            end

            if (mem_re)
                rd_valid <= 1'b1;
        end
    end

endmodule
