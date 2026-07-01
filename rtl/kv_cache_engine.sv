// kv_cache_engine.sv — Top-level KV Cache Engine (ChannelQuant codec)
//
// Asymmetric uniform-integer KV compression (docs/HW_CONTRACT.md):
//   K path: per-channel INT4 over a group of G tokens (+ optional top-k FP16
//           outlier channels in CQ-4+); CQ-8 keys are per-token INT8.
//   V path: per-token INT4 (CQ-4/CQ-4+) or INT8 (CQ-8).
//
// This revision streams the PER-TOKEN datapath through the FSM: incoming tokens
// are compressed by cq_value_path (amax → scale → quant → pack) and stored as
// {fp16 scale, packed payload} in SRAM; reads decompress (unpack → dequant) and
// stream the fp32 reconstruction out. This is the full codec for CQ-8 (per-token
// K and V) and for the value stream of every tier. The per-channel grouped KEY
// path (CQ-4/CQ-4+ keys, cq_key_path) is wired in the next integration step.
//
// Interfaces:
//   - AXI-Lite control (register window)
//   - AXI-Stream write (incoming KV vectors, COORD_WIDTH fp16 elements)
//   - AXI-Stream read (decompressed KV, OUT_WIDTH fp32 elements — contract §1)

module kv_cache_engine #(
    parameter integer VECTOR_DIM    = 64,   // D: head dim (64 or 128)
    parameter integer TIER          = 1,    // 0 = CQ-8, 1 = CQ-4, 2 = CQ-4+
    parameter integer KEY_GROUP     = 128,  // G: tokens per per-channel key group
    parameter integer OUTLIER_K     = 0,    // top-k FP16 key channels (CQ-4+)
    parameter integer SCALE_WIDTH   = 16,   // fp16 per-axis scale width
    parameter integer SRAM_DEPTH    = 16,
    parameter integer COORD_WIDTH   = 16,   // fp16 input element width
    parameter integer OUT_WIDTH     = 32    // fp32 decompressed output element (contract §1)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // ---- AXI-Lite Control ----
    input  wire [7:0]              axil_awaddr,
    input  wire                    axil_awvalid,
    output wire                    axil_awready,
    input  wire [31:0]             axil_wdata,
    input  wire                    axil_wvalid,
    output wire                    axil_wready,
    output wire [1:0]              axil_bresp,
    output reg                     axil_bvalid,
    input  wire                    axil_bready,
    input  wire [7:0]              axil_araddr,
    input  wire                    axil_arvalid,
    output wire                    axil_arready,
    output reg  [31:0]             axil_rdata,
    output wire [1:0]              axil_rresp,
    output reg                     axil_rvalid,
    input  wire                    axil_rready,

    // ---- AXI-Stream Write (incoming KV vectors, fp16) ----
    input  wire [COORD_WIDTH-1:0]  s_axis_kv_tdata,
    input  wire                    s_axis_kv_tvalid,
    output reg                     s_axis_kv_tready,
    input  wire                    s_axis_kv_tlast,
    input  wire                    s_axis_kv_tuser,  // 0=K, 1=V

    // ---- AXI-Stream Read (decompressed output, fp32) ----
    output reg  [OUT_WIDTH-1:0]    m_axis_kv_tdata,
    output reg                     m_axis_kv_tvalid,
    input  wire                    m_axis_kv_tready,
    output reg                     m_axis_kv_tlast,

    // ---- Eviction signal to Memory Hierarchy Controller ----
    output wire                    evict_needed,
    output wire [$clog2(SRAM_DEPTH)-1:0] evict_addr
);

    // -----------------------------------------------------------------------
    // Derived parameters
    // -----------------------------------------------------------------------
    localparam integer ADDR_WIDTH  = $clog2(SRAM_DEPTH);
    localparam integer VAL_BPV     = (TIER == 0) ? 8 : 4;   // per-token payload bits/elem
    localparam integer KEY_BPV     = (TIER == 0) ? 8 : 4;
    localparam integer PAY_BITS    = VECTOR_DIM * VAL_BPV;  // packed payload bits/token
    // Compressed token in SRAM = {fp16 scale, packed payload}.
    localparam integer SRAM_WIDTH  = SCALE_WIDTH + PAY_BITS;

    localparam [31:0] ISA_VERSION  = 32'h00_02_00_00; // v0.2.0.0 (ChannelQuant)

    // -----------------------------------------------------------------------
    // Register map (AXI-Lite)
    // -----------------------------------------------------------------------
    localparam [7:0] REG_CTRL             = 8'h00;
    localparam [7:0] REG_STATUS           = 8'h04;
    localparam [7:0] REG_INFO_DIM         = 8'h08;
    localparam [7:0] REG_INFO_TIER        = 8'h0C;
    localparam [7:0] REG_INFO_GROUP       = 8'h10;
    localparam [7:0] REG_INFO_SRAM_DEPTH  = 8'h14;
    localparam [7:0] REG_INFO_CR_K        = 8'h18;
    localparam [7:0] REG_INFO_CR_V        = 8'h1C;
    localparam [7:0] REG_INFO_VERSION     = 8'h20;
    localparam [7:0] REG_OCCUPANCY        = 8'h24;
    localparam [7:0] REG_WRITE_ADDR       = 8'h28;
    localparam [7:0] REG_READ_ADDR        = 8'h2C;
    localparam [7:0] REG_KV_SELECT        = 8'h30;
    localparam [7:0] REG_IRQ_MASK         = 8'h34;
    localparam [7:0] REG_IRQ_STATUS       = 8'h38;
    localparam [7:0] REG_INFO_OUTLIER_K   = 8'h3C;
    localparam [7:0] REG_INFO_SCALE_DEPTH = 8'h40;
    localparam [7:0] REG_INFO_RESID_DEPTH = 8'h44;

    reg        ctrl_enable;
    reg        ctrl_reset;
    reg [ADDR_WIDTH-1:0] write_addr;
    reg [ADDR_WIDTH-1:0] read_addr;
    reg        kv_select;
    reg [3:0]  irq_mask;
    reg [3:0]  irq_status;
    reg        read_req;             // pulse: a decompress/read was requested

    // -----------------------------------------------------------------------
    // Input token assembly
    // -----------------------------------------------------------------------
    reg  [VECTOR_DIM*COORD_WIDTH-1:0] tok_vec;   // assembled token (fp16 elems)
    reg  [$clog2(VECTOR_DIM):0]       in_count;
    reg                               input_is_key;

    // -----------------------------------------------------------------------
    // Value-path datapath core (per-token compress + decompress)
    // -----------------------------------------------------------------------
    reg                        cqv_in_valid;
    wire                       cqv_out_valid;
    wire [SCALE_WIDTH-1:0]     cqv_scale;
    wire [VECTOR_DIM*8-1:0]    cqv_codes;
    wire [VECTOR_DIM*8-1:0]    cqv_pay;
    reg  [VECTOR_DIM*8-1:0]    dec_codes;
    reg  [SCALE_WIDTH-1:0]     dec_scale;
    wire [$clog2(VECTOR_DIM)-1:0] dec_idx;
    wire [31:0]                dec_hat;   // one reconstructed fp32 channel (dec_idx)

    cq_value_path #(.D(VECTOR_DIM), .DW(COORD_WIDTH)) u_vpath (
        .clk(clk), .rst_n(rst_n), .bits(VAL_BPV[3:0]),
        .in_valid(cqv_in_valid), .in_vec(tok_vec), .busy(),
        .out_valid(cqv_out_valid), .out_scale(cqv_scale),
        .out_codes(cqv_codes), .out_pay(cqv_pay),
        .dec_codes(dec_codes), .dec_scale(dec_scale),
        .dec_idx(dec_idx), .dec_hat(dec_hat)
    );
    // decompress streams one channel per output beat; select the current one.
    assign dec_idx = out_count[$clog2(VECTOR_DIM)-1:0];

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    localparam [2:0] ST_IDLE     = 3'd0,
                     ST_COLLECT  = 3'd1,
                     ST_COMPRESS = 3'd2,
                     ST_STORE    = 3'd3,
                     ST_RLOAD    = 3'd4,   // launch SRAM read
                     ST_RWAIT    = 3'd5,   // capture read data, unpack
                     ST_OUTPUT   = 3'd6;   // stream fp32 beats
    reg [2:0] state;
    reg       idle;

    // SRAM
    reg                    sram_wr_en;
    reg [ADDR_WIDTH-1:0]   sram_wr_addr;
    reg [SRAM_WIDTH-1:0]   sram_wr_data;
    reg                    sram_rd_en;
    reg [ADDR_WIDTH-1:0]   sram_rd_addr;
    wire [SRAM_WIDTH-1:0]  sram_rd_data;
    wire                   sram_rd_valid;
    wire [ADDR_WIDTH:0]    sram_occupancy;
    wire                   sram_full;
    reg [$clog2(VECTOR_DIM):0] out_count;

    sram_controller #(
        .SRAM_DEPTH (SRAM_DEPTH),
        .DATA_WIDTH (SRAM_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_sram (
        .clk(clk), .rst_n(rst_n),
        .wr_en(sram_wr_en), .wr_addr(sram_wr_addr), .wr_data(sram_wr_data),
        .rd_en(sram_rd_en), .rd_addr(sram_rd_addr), .rd_data(sram_rd_data),
        .rd_valid(sram_rd_valid), .occupancy(sram_occupancy), .full(sram_full)
    );

    assign evict_needed = sram_full;
    assign evict_addr   = '0;

    // unpack a stored payload into per-element signed codes (contract §5).
    // int8: byte per element; int4: nibble per element, sign-extended.
    reg [VECTOR_DIM*8-1:0] unpacked_codes;
    integer u;
    always @* begin
        unpacked_codes = '0;
        for (u = 0; u < VECTOR_DIM; u = u + 1) begin
            if (VAL_BPV == 8) begin
                unpacked_codes[u*8 +: 8] = sram_rd_data[u*8 +: 8];
            end else begin
                // nibble u: low if even, high if odd; sign-extend 4->8
                if (u[0] == 1'b0)
                    unpacked_codes[u*8 +: 8] = {{4{sram_rd_data[(u>>1)*8 + 3]}},
                                                 sram_rd_data[(u>>1)*8     +: 4]};
                else
                    unpacked_codes[u*8 +: 8] = {{4{sram_rd_data[(u>>1)*8 + 7]}},
                                                 sram_rd_data[(u>>1)*8 + 4 +: 4]};
            end
        end
    end

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            in_count         <= '0;
            out_count        <= '0;
            s_axis_kv_tready <= 1'b1;
            m_axis_kv_tvalid <= 1'b0;
            m_axis_kv_tdata  <= '0;
            m_axis_kv_tlast  <= 1'b0;
            sram_wr_en       <= 1'b0;
            sram_rd_en       <= 1'b0;
            sram_rd_addr     <= '0;
            cqv_in_valid     <= 1'b0;
            idle             <= 1'b1;
        end else begin
            sram_wr_en   <= 1'b0;
            sram_rd_en   <= 1'b0;
            cqv_in_valid <= 1'b0;

            if (ctrl_reset) begin
                state    <= ST_IDLE;
                in_count <= '0;
                idle     <= 1'b1;
            end

            case (state)
                ST_IDLE: begin
                    idle             <= 1'b1;
                    s_axis_kv_tready <= ctrl_enable;
                    m_axis_kv_tvalid <= 1'b0;   // deassert after an output burst
                    m_axis_kv_tlast  <= 1'b0;
                    if (read_req) begin
                        state <= ST_RLOAD;
                        idle  <= 1'b0;
                    end else if (s_axis_kv_tvalid && s_axis_kv_tready) begin
                        state        <= ST_COLLECT;
                        idle         <= 1'b0;
                        tok_vec[0 +: COORD_WIDTH] <= s_axis_kv_tdata;
                        in_count     <= 1;
                        input_is_key <= ~s_axis_kv_tuser;
                    end
                end

                ST_COLLECT: begin
                    if (s_axis_kv_tvalid && s_axis_kv_tready) begin
                        tok_vec[in_count*COORD_WIDTH +: COORD_WIDTH] <= s_axis_kv_tdata;
                        in_count <= in_count + 1;
                        if (s_axis_kv_tlast || in_count == VECTOR_DIM - 1) begin
                            state            <= ST_COMPRESS;
                            s_axis_kv_tready <= 1'b0;
                            cqv_in_valid     <= 1'b1;   // present token to the datapath
                        end
                    end
                end

                ST_COMPRESS: begin
                    // wait for the value-path to finish (serial: ~D+2 cycles)
                    if (cqv_out_valid) state <= ST_STORE;
                end

                ST_STORE: begin
                    sram_wr_en   <= 1'b1;
                    sram_wr_addr <= write_addr;
                    sram_wr_data <= {cqv_scale, cqv_pay[PAY_BITS-1:0]};
                    in_count     <= '0;
                    state        <= ST_IDLE;
                    s_axis_kv_tready <= ctrl_enable;
                end

                ST_RLOAD: begin
                    sram_rd_en   <= 1'b1;
                    sram_rd_addr <= read_addr;
                    state        <= ST_RWAIT;
                end

                ST_RWAIT: begin
                    if (sram_rd_valid) begin
                        dec_codes <= unpacked_codes;
                        dec_scale <= sram_rd_data[SRAM_WIDTH-1 -: SCALE_WIDTH];
                        out_count <= '0;
                        state     <= ST_OUTPUT;
                    end
                end

                ST_OUTPUT: begin
                    // one fp32 beat per cycle (consumer holds tready; see IDLE clear)
                    m_axis_kv_tdata  <= dec_hat;   // channel dec_idx = out_count
                    m_axis_kv_tvalid <= 1'b1;
                    m_axis_kv_tlast  <= (out_count == VECTOR_DIM - 1);
                    if (out_count == VECTOR_DIM - 1) begin
                        state     <= ST_IDLE;
                        out_count <= '0;
                    end else begin
                        out_count <= out_count + 1;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // AXI-Lite register interface
    // -----------------------------------------------------------------------
    localparam integer VAL_EFF_DEN = VECTOR_DIM * VAL_BPV + SCALE_WIDTH;
    localparam integer KEY_EFF_DEN = (TIER == 0)
                                   ? (VECTOR_DIM * KEY_BPV + SCALE_WIDTH)
                                   : (VECTOR_DIM * KEY_BPV +
                                      (SCALE_WIDTH * VECTOR_DIM) / KEY_GROUP);
    localparam [31:0] CR_V_FIXED = (VECTOR_DIM * COORD_WIDTH * 256) / VAL_EFF_DEN;
    localparam [31:0] CR_K_FIXED = (VECTOR_DIM * COORD_WIDTH * 256) / KEY_EFF_DEN;

    assign axil_awready = 1'b1;
    assign axil_wready  = 1'b1;
    assign axil_bresp   = 2'b00;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axil_bvalid  <= 1'b0;
            ctrl_enable  <= 1'b0;
            ctrl_reset   <= 1'b0;
            write_addr   <= '0;
            read_addr    <= '0;
            kv_select    <= 1'b0;
            irq_mask     <= '0;
            irq_status   <= '0;
            read_req     <= 1'b0;
        end else begin
            ctrl_reset  <= 1'b0;
            axil_bvalid <= 1'b0;
            read_req    <= 1'b0;
            if (axil_awvalid && axil_wvalid) begin
                axil_bvalid <= 1'b1;
                case (axil_awaddr)
                    REG_CTRL: begin
                        ctrl_reset  <= axil_wdata[0];
                        ctrl_enable <= axil_wdata[1];
                    end
                    REG_WRITE_ADDR: write_addr <= axil_wdata[ADDR_WIDTH-1:0];
                    REG_READ_ADDR:  begin
                        read_addr <= axil_wdata[ADDR_WIDTH-1:0];
                        read_req  <= 1'b1;              // writing READ_ADDR launches a decompress
                    end
                    REG_KV_SELECT:  kv_select  <= axil_wdata[0];
                    REG_IRQ_MASK:   irq_mask   <= axil_wdata[3:0];
                    REG_IRQ_STATUS: irq_status <= irq_status & ~axil_wdata[3:0];
                    default: ;
                endcase
            end
        end
    end

    assign axil_arready = 1'b1;
    assign axil_rresp   = 2'b00;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axil_rvalid <= 1'b0;
            axil_rdata  <= '0;
        end else begin
            axil_rvalid <= axil_arvalid;
            if (axil_arvalid) begin
                case (axil_araddr)
                    REG_CTRL:            axil_rdata <= {30'b0, ctrl_enable, 1'b0};
                    REG_STATUS:          axil_rdata <= {28'b0, sram_full, 1'b0, 1'b0, idle};
                    REG_INFO_DIM:        axil_rdata <= VECTOR_DIM;
                    REG_INFO_TIER:       axil_rdata <= TIER;
                    REG_INFO_GROUP:      axil_rdata <= KEY_GROUP;
                    REG_INFO_SRAM_DEPTH: axil_rdata <= SRAM_DEPTH;
                    REG_INFO_CR_K:       axil_rdata <= CR_K_FIXED;
                    REG_INFO_CR_V:       axil_rdata <= CR_V_FIXED;
                    REG_INFO_VERSION:    axil_rdata <= ISA_VERSION;
                    REG_OCCUPANCY:       axil_rdata <= {{(32-ADDR_WIDTH-1){1'b0}}, sram_occupancy};
                    REG_WRITE_ADDR:      axil_rdata <= {{(32-ADDR_WIDTH){1'b0}}, write_addr};
                    REG_READ_ADDR:       axil_rdata <= {{(32-ADDR_WIDTH){1'b0}}, read_addr};
                    REG_KV_SELECT:       axil_rdata <= {31'b0, kv_select};
                    REG_IRQ_MASK:        axil_rdata <= {28'b0, irq_mask};
                    REG_IRQ_STATUS:      axil_rdata <= {28'b0, irq_status};
                    REG_INFO_OUTLIER_K:  axil_rdata <= OUTLIER_K;
                    REG_INFO_SCALE_DEPTH:axil_rdata <= VECTOR_DIM;
                    REG_INFO_RESID_DEPTH:axil_rdata <= KEY_GROUP;
                    default:             axil_rdata <= 32'hDEAD_BEEF;
                endcase
            end
        end
    end

endmodule
