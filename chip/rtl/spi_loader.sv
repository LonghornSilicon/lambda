// spi_loader.sv — serial host loader for the Lambda ACU workshop macro.
//
// STATUS: SKELETON (synthesizable, but not the final protocol). This is a
// real starting point + a documented host contract, NOT a complete loader.
// TODO items are called out inline.
//
// WHY THIS EXISTS
// ---------------
// The Lambda decode-attention datapath is far wider than the pads it gets in
// the GF180 workshop slot. A single decode step streams D=128 INT8 Q/K/V
// vectors (and reads back a D-wide attention output), but the workshop slot
// exposes only ~20 bidir pads + 1 input pad. So every tensor has to be
// *streamed* in over a narrow serial link, latched into on-chip buffers, the
// attention computed, and the result *streamed* back out. This module is that
// narrow link: a byte-oriented SPI *slave*.
//
// PAD BUDGET (workshop slot = 1 input pad, 20 bidir pads; see padring fork
// Mauricio-xx/chipathon-2026-gf180mcu-padring, SLOT_WORKSHOP):
//   spi_sclk  (in)  : serial clock from host
//   spi_cs_n  (in)  : active-low frame select
//   spi_mosi  (in)  : host -> chip data
//   spi_miso  (out) : chip -> host data
// The remaining bidir pads are free for debug/observation (see lambda_acu.sv).
//
// HOST PROTOCOL (SPI mode 0: CPOL=0, CPHA=0, MSB-first)
// -----------------------------------------------------
// A frame is delimited by spi_cs_n going low .. high. Byte layout:
//
//   byte 0      : CMD
//                   0x01 WRITE  — write DATA bytes starting at ADDR (auto-inc)
//                   0x02 READ   — read  DATA bytes starting at ADDR (auto-inc)
//                   0x03 START  — pulse start; run one decode-attention step
//                   0x04 STATUS — next MISO byte returns the STATUS register
//   byte 1      : ADDR[15:8]
//   byte 2      : ADDR[7:0]
//   byte 3..n   : DATA (streamed; internal address auto-increments per byte)
//
// ADDRESS MAP (16-bit, byte-addressed; wide fields little-endian on the wire)
//   0x0000  CTRL      [0]=start [1]=precision (0=INT8 path,1=FP16 path) [2]=flush
//   0x0001  STATUS    [0]=busy  [1]=done      [2]=err   (read-only)
//   0x0002  SEQ_LEN   number of cached tokens for this step (16-bit, 2 bytes)
//   0x0004  HEAD_DIM  active head dim D (defaults to param D)
//   0x0100.. QVEC     Q vector, D INT8 bytes         (host -> chip)
//   0x0200.. KVEC     K vector stream, D INT8 bytes  (host -> chip, per token)
//   0x0300.. VVEC     V vector stream, D INT8 bytes  (host -> chip, per token)
//   0x0800.. OUT      attention output, D bytes      (chip -> host, READ)
// TODO: finalize the map once kve/mate_qkt/vecu_softmax stream interfaces land.
//
// This skeleton implements: CS framing, sclk edge detection (oversampled in
// the core clock domain — TODO: replace with a proper async SPI FE if the host
// SCLK approaches core clk/4), MSB-first byte (de)assembly, the CMD/ADDR/DATA
// FSM, a couple of real CSRs, and a generic byte-write strobe + byte-read mux
// that lambda_acu wires to the block buffers. It is intentionally a stub: the
// per-block buffer fabric is TODO and lives in lambda_acu.sv.

`default_nettype none

module spi_loader #(
    parameter integer ADDR_WIDTH = 16,
    parameter integer DATA_WIDTH = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // ---- serial pads (sampled/driven by chip_core) ----
    input  wire                     spi_sclk,
    input  wire                     spi_cs_n,
    input  wire                     spi_mosi,
    output reg                      spi_miso,

    // ---- decoded byte bus to the datapath buffer fabric (lambda_acu) ----
    output reg  [ADDR_WIDTH-1:0]    bus_addr,     // current byte address
    output reg  [DATA_WIDTH-1:0]    bus_wdata,    // byte to write
    output reg                      bus_we,       // 1-cycle write strobe
    output reg                      bus_re,       // 1-cycle read-request strobe
    input  wire [DATA_WIDTH-1:0]    bus_rdata,    // byte returned for READ

    // ---- control / status handshakes to the datapath ----
    output reg                      start,        // pulse: run one decode step
    output reg  [1:0]               precision_sel,// CTRL[2:1] mirror (mode bits)
    input  wire                     busy,         // datapath asserts while running
    input  wire                     done,         // datapath pulses on completion
    input  wire [7:0]               status_in     // full STATUS byte (from lambda_acu)
);

    // ---------------- CDC: sample the serial inputs -------------------------
    // Oversample the host SPI lines in the core clock domain. Assumes
    // f(spi_sclk) << f(clk). TODO: swap for a metastability-hardened SPI front
    // end (2-flop sync + edge FIFO) before tapeout.
    reg [2:0] sclk_sync;
    reg [1:0] cs_sync;
    reg [1:0] mosi_sync;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sclk_sync <= '0; cs_sync <= 2'b11; mosi_sync <= '0;
        end else begin
            sclk_sync <= {sclk_sync[1:0], spi_sclk};
            cs_sync   <= {cs_sync[0],     spi_cs_n};
            mosi_sync <= {mosi_sync[0],   spi_mosi};
        end
    end
    wire sclk_rise = (sclk_sync[2:1] == 2'b01);
    wire cs_active = ~cs_sync[1];
    wire cs_start  = (cs_sync == 2'b10); // falling edge of cs_n -> frame start

    // ---------------- byte assembly (MSB first) -----------------------------
    reg [2:0] bit_cnt;
    reg [7:0] rx_shift;
    reg       byte_ready;   // 1-cycle: rx_shift holds a fresh byte

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_cnt <= 3'd0; rx_shift <= 8'd0; byte_ready <= 1'b0;
        end else begin
            byte_ready <= 1'b0;
            if (!cs_active) begin
                bit_cnt <= 3'd0;
            end else if (sclk_rise) begin
                rx_shift <= {rx_shift[6:0], mosi_sync[1]};
                bit_cnt  <= bit_cnt + 3'd1;
                if (bit_cnt == 3'd7)
                    byte_ready <= 1'b1;
            end
        end
    end

    // ---------------- frame FSM ---------------------------------------------
    localparam [2:0] S_CMD  = 3'd0,
                     S_ADRH = 3'd1,
                     S_ADRL = 3'd2,
                     S_DATA = 3'd3;

    localparam [7:0] CMD_WRITE  = 8'h01,
                     CMD_READ   = 8'h02,
                     CMD_START  = 8'h03,
                     CMD_STATUS = 8'h04;

    reg [2:0] state;
    reg [7:0] cmd_r;

    // Real CSRs held here; the wide tensor buffers live in lambda_acu.
    reg [15:0] csr_seq_len;
    reg [15:0] csr_head_dim;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_CMD;
            cmd_r         <= 8'd0;
            bus_addr      <= '0;
            bus_wdata     <= '0;
            bus_we        <= 1'b0;
            bus_re        <= 1'b0;
            start         <= 1'b0;
            precision_sel <= 2'b00;
            csr_seq_len   <= 16'd0;
            csr_head_dim  <= 16'd0;
        end else begin
            // default strobes low
            bus_we <= 1'b0;
            bus_re <= 1'b0;
            start  <= 1'b0;

            // POST-ACCESS auto-increment. bus_we/bus_re are high the cycle AFTER
            // each byte_ready, while bus_addr still equals the accessed address —
            // so the fabric write/read targets that address this cycle, and the
            // pointer advances for the next byte. (Incrementing in the same
            // byte_ready cycle that sets the strobe would land the byte one
            // address too high, since both take effect on the same edge.)
            if (bus_we || bus_re) bus_addr <= bus_addr + 1'b1;

            if (cs_start) begin
                state <= S_CMD;             // new frame
            end else if (byte_ready) begin
                case (state)
                    S_CMD: begin
                        cmd_r <= rx_shift;
                        if (rx_shift == CMD_START) begin
                            start <= 1'b1;   // START has no addr/data payload
                            state <= S_CMD;
                        end else begin
                            state <= S_ADRH;
                        end
                    end
                    S_ADRH: begin
                        bus_addr[15:8] <= rx_shift;
                        state          <= S_ADRL;
                    end
                    S_ADRL: begin
                        bus_addr[7:0] <= rx_shift;
                        state         <= S_DATA;
                        // (no read pre-issue: the MISO shifter reloads from the
                        //  current bus_addr one cycle after each byte boundary)
                    end
                    S_DATA: begin
                        if (cmd_r == CMD_WRITE) begin
                            bus_wdata <= rx_shift;
                            bus_we    <= 1'b1;
                            // local CSR shadow writes
                            case (bus_addr)
                                16'h0000: begin
                                    start         <= rx_shift[0];
                                    precision_sel <= rx_shift[2:1];
                                end
                                16'h0002: csr_seq_len[7:0]   <= rx_shift;
                                16'h0003: csr_seq_len[15:8]  <= rx_shift;
                                16'h0004: csr_head_dim[7:0]  <= rx_shift;
                                16'h0005: csr_head_dim[15:8] <= rx_shift;
                                default: ; // wide tensor region: handled by fabric
                            endcase
                        end else if (cmd_r == CMD_READ) begin
                            bus_re <= 1'b1;  // request next byte
                        end
                        // (address advances via the post-access increment above)
                    end
                    default: state <= S_CMD;
                endcase
            end
        end
    end

    // ---------------- MISO response (reads / status) ------------------------
    // MSB-first serial response, SPI mode 0. The response byte is stable for the
    // whole byte (the fabric read data at the current bus_addr, which advances
    // only at byte boundaries via the post-access increment; or the STATUS
    // register for CMD_STATUS). We drive MISO COMBINATIONALLY as bit (7-bit_cnt)
    // of that byte — MSB first — so it is valid the instant the host raises sclk,
    // before the (oversampled, ~2-clk) rising-edge detect advances bit_cnt. The
    // host samples MISO early in the high phase. No shift register, no reload
    // race: the byte address and the bit index fully determine MISO.
    //
    //   response byte = STATUS register (CMD_STATUS) or the fabric read data.
    wire [7:0] resp_byte = (cmd_r == CMD_STATUS) ? status_in : bus_rdata;
    always @* spi_miso = cs_active ? resp_byte[3'd7 - bit_cnt] : 1'b0;

    // keep unused width-derived signals from being pruned in lint
    logic _unused;
    assign _unused = &{1'b0, csr_seq_len, csr_head_dim};

endmodule

`default_nettype wire
