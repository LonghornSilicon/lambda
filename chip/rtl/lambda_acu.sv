// lambda_acu.sv — Lambda ACU (decode-attention datapath) top skeleton.
//
// STATUS: SKELETON. Wires the serial host loader (spi_loader) to a byte-
// addressed buffer fabric and an output-observation mux, and reserves the
// instantiation sites for every decode-attention macro. The real macro
// instances are TODO — each is dropped in as its RTL is hardened for GF180
// (the .sv sources already live under rtl/blocks/, copied from the sibling
// repos; see rtl/blocks/PROVENANCE.md). This top is what chip_core.sv
// instantiates into the workshop slot.
//
// DECODE-ATTENTION DATAPATH (one decode step)
//   host --(SPI)--> Q,K,V INT8 vectors
//        kve                 : KV-cache codec (ChannelQuant) — store/stream K,V
//        token_importance_unit: H2O heavy-hitter scoring / eviction pick
//        precision_controller : per-tile INT8-vs-FP16 decision (d_fp16)
//        mate_qkt   [Phase 1] : Q·Kᵀ scores        (RTL IN PROGRESS)
//        vecu_softmax [Phase2]: softmax/normalize   (RTL NOT STARTED)
//        mate_pv / mate_pv_fp16: P·V reduction (INT8 signed-off / FP16 tol)
//   result --(SPI)--> host
//
// The 128-wide INT8 tensors do not fit the ~20 workshop pads, so spi_loader
// streams them into the buffers below and streams the attention result back.

`default_nettype none

module lambda_acu #(
    parameter integer D          = 128, // head dim (INT8 lanes) — real target
    parameter integer ADDR_WIDTH = 16,
    parameter integer NUM_BIDIR  = 20   // workshop-slot bidir pad count
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // serial host pads (routed up from chip_core / the workshop bidir pads)
    input  wire                      spi_sclk,
    input  wire                      spi_cs_n,
    input  wire                      spi_mosi,
    output wire                      spi_miso,

    // spare observation outputs -> remaining bidir pads
    output wire [NUM_BIDIR-1:0]      obs_out
);

    // ---------------- serial loader ----------------------------------------
    wire [ADDR_WIDTH-1:0] bus_addr;
    wire [7:0]            bus_wdata;
    wire                  bus_we;
    wire                  bus_re;
    reg  [7:0]            bus_rdata;
    wire                  start;
    wire [1:0]            precision_sel;

    // Datapath status back to the host. TODO: drive `busy`/`done` from the
    // real block-chain sequencer once the macros are instantiated. For now the
    // skeleton reports a trivial done-after-start so the host handshake is
    // exercisable in the smoke test.
    reg  busy_r, done_r;

    spi_loader #(
        .ADDR_WIDTH (ADDR_WIDTH),
        .DATA_WIDTH (8)
    ) u_spi (
        .clk           (clk),
        .rst_n         (rst_n),
        .spi_sclk      (spi_sclk),
        .spi_cs_n      (spi_cs_n),
        .spi_mosi      (spi_mosi),
        .spi_miso      (spi_miso),
        .bus_addr      (bus_addr),
        .bus_wdata     (bus_wdata),
        .bus_we        (bus_we),
        .bus_re        (bus_re),
        .bus_rdata     (bus_rdata),
        .start         (start),
        .precision_sel (precision_sel),
        .busy          (busy_r),
        .done          (done_r)
    );

    // ---------------- input tensor buffers (streamed in) --------------------
    // Byte-addressed staging for the Q/K/V vectors. Region bases mirror the
    // spi_loader address map. TODO: replace flat byte RAM with the real
    // ping-pong / streaming interfaces the macros expect (kve AXI-Stream,
    // mate_* valid/last token stream).
    localparam integer QBASE = 16'h0100;
    localparam integer KBASE = 16'h0200;
    localparam integer VBASE = 16'h0300;
    localparam integer OBASE = 16'h0800;

    reg [7:0] q_buf [0:D-1];
    reg [7:0] k_buf [0:D-1];
    reg [7:0] v_buf [0:D-1];
    reg [7:0] o_buf [0:D-1];   // attention result staging (host reads this)

    wire in_q = (bus_addr >= QBASE) && (bus_addr < QBASE + D);
    wire in_k = (bus_addr >= KBASE) && (bus_addr < KBASE + D);
    wire in_v = (bus_addr >= VBASE) && (bus_addr < VBASE + D);
    wire in_o = (bus_addr >= OBASE) && (bus_addr < OBASE + D);

    always_ff @(posedge clk) begin
        if (bus_we) begin
            if (in_q) q_buf[bus_addr - QBASE] <= bus_wdata;
            if (in_k) k_buf[bus_addr - KBASE] <= bus_wdata;
            if (in_v) v_buf[bus_addr - VBASE] <= bus_wdata;
        end
    end

    // Read mux for host READ frames.
    always_ff @(posedge clk) begin
        if (bus_re) begin
            bus_rdata <= in_o ? o_buf[bus_addr - OBASE] : 8'h00;
        end
    end

    // ---------------- placeholder sequencer --------------------------------
    // Trivial start->done pulse so the SPI handshake is testable. The REAL
    // sequencer walks: kve read -> mate_qkt -> vecu_softmax -> tiu -> mate_pv,
    // gated by precision_controller.d_fp16. TODO.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy_r <= 1'b0; done_r <= 1'b0;
        end else begin
            done_r <= 1'b0;
            if (start)      busy_r <= 1'b1;
            else if (busy_r) begin
                busy_r <= 1'b0;   // 1-cycle placeholder "compute"
                done_r <= 1'b1;
            end
        end
    end

    // =======================================================================
    // MACRO INSTANTIATION SITES (decode-attention datapath).
    // Each block is hardened as its own GF180 macro (librelane/<block>.yaml)
    // and dropped in here. Left as TODO so this skeleton elaborates standalone;
    // uncommenting a block also means wiring its stream ports to the buffers
    // above and adding its .sv/netlist to the tb + chip_top build.
    // =======================================================================

    // TODO macro: kve (kv_cache_engine) — ChannelQuant KV codec.
    //   src: rtl/blocks/kve/*.sv   yaml: librelane/kve.yaml   (RTL: REAL, Sky130-signed)
    //   kv_cache_engine #(.VECTOR_DIM(D), .TIER(1) ...) u_kve ( .clk, .rst_n, AXI-Lite + AXI-Stream ... );

    // TODO macro: token_importance_unit — H2O heavy-hitter eviction.
    //   src: rtl/blocks/token_importance_unit.sv  yaml: librelane/token_importance_unit.yaml (RTL: REAL, Sky130-signed)
    //   token_importance_unit #(.N_SLOTS(...)) u_tiu ( .clk, .rst_n, acc/ld/evict ... );

    // TODO macro: precision_controller — INT8-vs-FP16 per-tile decision.
    //   src: rtl/blocks/precision_controller.sv   yaml: librelane/precision_controller.yaml (RTL: REAL, Sky130-signed)
    //   precision_controller #(.BLOCK_M, .BLOCK_N) u_pc ( .clk, .rst_n, s_valid/s_data/s_last -> d_valid/d_fp16 );
    //   -> d_fp16 selects mate_pv (INT8) vs mate_pv_fp16 (FP16). precision_sel from host can force the mode.

    // TODO macro: mate_qkt [PHASE 1 — RTL IN PROGRESS] — Q·Kᵀ score tile.
    //   src: (not yet in rtl/blocks)   yaml: librelane/mate_qkt.yaml (STUB)

    // TODO macro: vecu_softmax [PHASE 2 — RTL NOT STARTED] — softmax/normalize.
    //   src: (not yet in rtl/blocks)   yaml: librelane/vecu_softmax.yaml (STUB)

    // TODO macro: mate_pv — INT8 P·V reduction (signed-off).
    //   src: rtl/blocks/mate_pv.sv   yaml: librelane/mate_pv.yaml (RTL: REAL, Sky130-signed)
    //   mate_pv #(.N(D)) u_pv ( .clk,.rst_n, s_valid,a_data,v_data,s_last -> c_valid,c_data );

    // TODO macro: mate_pv_fp16 — FP16 P·V reduction (tolerance path).
    //   src: rtl/blocks/mate_pv_fp16.sv  yaml: librelane/mate_pv_fp16.yaml (RTL: REAL, Sky130-signed)

    // ---------------- observation outputs ----------------------------------
    // Expose liveness on the spare bidir pads: an 8-bit free-running heartbeat
    // (always defined, so a logic analyzer sees the chip is alive) plus the
    // busy/done handshake bits. TODO: replace the heartbeat byte with a real
    // debug mux (o_buf tap, block-chain state) once the datapath is live.
    reg [7:0] heartbeat;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) heartbeat <= 8'd0;
        else        heartbeat <= heartbeat + 8'd1;
    end
    assign obs_out = { {(NUM_BIDIR-10){1'b0}}, done_r, busy_r, heartbeat };

    // keep as-yet-unwired signals from being pruned until the datapath lands
    logic _unused;
    assign _unused = &{1'b0, precision_sel, in_k, in_v, o_buf[0]};

endmodule

`default_nettype wire
