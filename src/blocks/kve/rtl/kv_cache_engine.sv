// kv_cache_engine.sv — Top-level KV Cache Engine (ChannelQuant codec)
//
// Asymmetric uniform-integer KV compression (docs/HW_CONTRACT.md):
//   K path: per-channel INT4 over a group of G tokens (+ optional top-k FP16
//           outlier channels in CQ-4+); CQ-8 keys are per-token INT8.
//   V path: per-token INT4 (CQ-4/CQ-4+) or INT8 (CQ-8).
//
// Two datapaths stream through the FSM:
//   - VALUE path (cq_value_path): every value token, plus ALL keys when TIER==0
//     (CQ-8 keys are per-token). amax -> scale -> quant -> pack; SRAM record
//     {tag=0, fp16 scale, packed payload}. Decompress -> fp32 (contract §1).
//   - grouped KEY path (cq_key_path), active for TIER==1/2 keys: buffer a group
//     of G tokens, freeze D per-channel scales, emit per-token INT4 keep codes.
//     SRAM record per key token is UNIFIED per-channel: {tag=1, D×fp16 field,
//     D×INT4 codes}. Keep channel c: field=group scale, code=INT4. Outlier
//     channel c: field=raw fp16 input, code=+1 -> decompress code·field widens
//     the fp16 exactly (no separate sidecar region / no read-side mask). Read
//     reuses cq_key_path's D combinational dequant units.
//
// Group flush is automatic at G tokens (full-group streaming); partial-group
// (g<G) flush is a documented follow-up (the datapath supports it).
//
// Interfaces:
//   - AXI-Lite control (register window)
//   - AXI-Stream write (incoming KV vectors, COORD_WIDTH fp16 elements)
//   - AXI-Stream read (decompressed KV, OUT_WIDTH fp32 elements — contract §1)

module kv_cache_engine #(
    // D: real head dim is 64/128 (set per-instantiation; every TB overrides). The
    // DEFAULT is a small GATE PROXY: the synthesized default drives the flop-only
    // synth/formal gates (3 & 4), whose cost scales with the key path's per-channel
    // datapath + two behavioral flop MEMORIES. At D=64 the gate-4 formal-equivalence
    // induction blows past its ~10-min budget, so the proxy shrinks D (and KEY_GROUP
    // / SRAM_DEPTH, which size residual_buffer=G*D*DW and SRAM=depth*width). This is
    // exactly the "keep the flop proxy small" rationale the block already used for
    // SRAM_DEPTH. OpenLane (gate 6) shrinks further via config.json SYNTH_PARAMETERS.
    parameter integer VECTOR_DIM    = 16,
    parameter integer TIER          = 1,    // 0 = CQ-8, 1 = CQ-4, 2 = CQ-4+
    parameter integer KEY_GROUP     = 2,    // shipped 128 (contract §3.1); proxy default
    parameter integer OUTLIER_K     = 0,    // top-k FP16 key channels (CQ-4+)
    parameter integer SCALE_WIDTH   = 16,   // fp16 per-axis scale width
    parameter integer SRAM_DEPTH    = 2,    // proxy default; real capacity per-instantiation
    parameter integer COORD_WIDTH   = 16,   // fp16 input element width
    parameter integer OUT_WIDTH     = 32,   // fp32 decompressed output element (contract §1)
    parameter         MASK_FILE     = ""    // outlier-mask ROM hex (only used if OUTLIER_K>0)
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
    localparam integer PAY_BITS    = VECTOR_DIM * VAL_BPV;  // packed value payload bits/token
    localparam integer VAL_REC_BITS= SCALE_WIDTH + PAY_BITS;

    // grouped-key path is active for the per-channel INT4 tiers (CQ-4/CQ-4+)
    localparam integer KEY_GROUPED = (TIER != 0) ? 1 : 0;
    localparam integer GW          = (KEY_GROUP > 1) ? $clog2(KEY_GROUP) : 1;
    localparam [GW-1:0] GRP_LAST   = KEY_GROUP - 1;   // last in-group token index (fits GW bits)

    // unified per-channel key record: {D fp16 field, D INT4 codes}
    localparam integer KEY_CODES_BITS = VECTOR_DIM * 4;
    localparam integer KEY_FIELD_BITS = VECTOR_DIM * COORD_WIDTH;
    localparam integer KEY_REC_BITS   = KEY_FIELD_BITS + KEY_CODES_BITS;

    localparam integer REC_BITS   = (KEY_GROUPED && (KEY_REC_BITS > VAL_REC_BITS))
                                  ? KEY_REC_BITS : VAL_REC_BITS;
    // one tag bit only when the key path can produce a differently-decoded record
    localparam integer SRAM_WIDTH = REC_BITS + KEY_GROUPED;
    localparam integer TAG_BIT    = REC_BITS;   // valid only when KEY_GROUPED

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
    // Input token assembly (shared by both datapaths — one token at a time)
    // -----------------------------------------------------------------------
    reg  [VECTOR_DIM*COORD_WIDTH-1:0] tok_vec;   // assembled token (fp16 elems), shift-filled
    reg  [$clog2(VECTOR_DIM):0]       in_count;
    reg                               input_is_key;

    // -----------------------------------------------------------------------
    // Outlier mask (feeds cq_key_path + the store-side scatter). k=0 -> bypassed.
    // -----------------------------------------------------------------------
    wire [VECTOR_DIM-1:0] outlier_mask_bus;
    generate
        if (OUTLIER_K > 0) begin : g_mask_rom
            reg [7:0] mask_mem [0:VECTOR_DIM-1];
            initial $readmemh(MASK_FILE, mask_mem);
            genvar mc;
            for (mc = 0; mc < VECTOR_DIM; mc = mc + 1)
                assign outlier_mask_bus[mc] = mask_mem[mc][0];
        end else begin : g_mask_zero
            assign outlier_mask_bus = '0;
        end
    endgenerate

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

    reg [$clog2(VECTOR_DIM):0] out_count;

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
    // Grouped KEY-path datapath core (only instantiated for CQ-4/CQ-4+)
    // -----------------------------------------------------------------------
    reg                        kp_in_valid;
    reg                        kp_group_start;
    reg                        kp_group_last;
    wire                       kp_group_valid;
    wire [VECTOR_DIM*COORD_WIDTH-1:0] kp_scales_bus;
    wire                       kp_tok_valid;
    wire [GW-1:0]              kp_tok_idx;
    wire [VECTOR_DIM*8-1:0]    kp_tok_codes;   // compacted keep codes (byte i = i-th keep)
    wire [VECTOR_DIM*COORD_WIDTH-1:0] kp_emit_vec;  // emitting token's raw fp16
    reg  [VECTOR_DIM*8-1:0]    kp_dec_codes;   // per-channel code (read-back)
    reg  [VECTOR_DIM*COORD_WIDTH-1:0] kp_dec_scales; // per-channel field (read-back)
    wire [31:0]                kp_dec_hat;    // one reconstructed fp32 key channel (dec_idx)

    generate
        if (KEY_GROUPED) begin : g_kpath
            cq_key_path #(.D(VECTOR_DIM), .DW(COORD_WIDTH), .G(KEY_GROUP)) u_kpath (
                .clk(clk), .rst_n(rst_n), .outlier_mask(outlier_mask_bus),
                .in_valid(kp_in_valid), .in_vec(tok_vec),
                .group_start(kp_group_start), .group_last(kp_group_last),
                .group_valid(kp_group_valid), .scales_bus(kp_scales_bus), .g_out(),
                .tok_valid(kp_tok_valid), .tok_idx(kp_tok_idx),
                .tok_pay(), .tok_codes(kp_tok_codes), .emit_vec(kp_emit_vec),
                .dec_codes(kp_dec_codes), .dec_scales(kp_dec_scales),
                .dec_idx(out_count[$clog2(VECTOR_DIM)-1:0]), .dec_hat(kp_dec_hat)
            );
        end else begin : g_no_kpath
            assign kp_group_valid = 1'b0;
            assign kp_scales_bus   = '0;
            assign kp_tok_valid    = 1'b0;
            assign kp_tok_idx      = '0;
            assign kp_tok_codes    = '0;
            assign kp_emit_vec     = '0;
            assign kp_dec_hat      = '0;
        end
    endgenerate

    // Store-side scatter: build the unified per-channel key record for the token
    // currently emitted by cq_key_path. Keep channel -> {group scale, INT4 code};
    // outlier channel -> {raw fp16, code +1}. Combinational; used in ST_KEMIT.
    reg [KEY_FIELD_BITS-1:0] key_field16;
    reg [KEY_CODES_BITS-1:0] key_codes4;
    reg [$clog2(VECTOR_DIM):0] ki_s;
    integer cs;
    always @* begin
        key_field16 = '0;
        key_codes4  = '0;
        ki_s        = '0;
        for (cs = 0; cs < VECTOR_DIM; cs = cs + 1) begin
            if (outlier_mask_bus[cs]) begin
                key_field16[cs*COORD_WIDTH +: COORD_WIDTH] = kp_emit_vec[cs*COORD_WIDTH +: COORD_WIDTH];
                key_codes4 [cs*4 +: 4]                     = 4'd1;
            end else begin
                key_field16[cs*COORD_WIDTH +: COORD_WIDTH] = kp_scales_bus[cs*COORD_WIDTH +: COORD_WIDTH];
                key_codes4 [cs*4 +: 4]                     = kp_tok_codes[ki_s*8 +: 4];
                ki_s = ki_s + 1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // FSM
    // -----------------------------------------------------------------------
    localparam [3:0] ST_IDLE     = 4'd0,
                     ST_COLLECT  = 4'd1,
                     ST_COMPRESS = 4'd2,
                     ST_STORE    = 4'd3,
                     ST_RLOAD    = 4'd4,   // launch SRAM read
                     ST_RWAIT    = 4'd5,   // capture read data, unpack
                     ST_OUTPUT   = 4'd6,   // stream fp32 beats
                     ST_KCOLLECT = 4'd7,   // collect remaining beats of a key token
                     ST_KFEED    = 4'd8,   // present the token to cq_key_path
                     ST_KACCEPT  = 4'd9,   // wait for the next key token in the group
                     ST_KEMIT    = 4'd10;  // capture cq_key_path emissions -> SRAM
    reg [3:0] state;
    reg       idle;
    reg       out_is_key;                 // selects key vs value dequant on read

    reg [GW-1:0]         grp_tok_cnt;      // tokens fed to cq_key_path in the group
    reg [ADDR_WIDTH-1:0] grp_base;         // SRAM base address for the group

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

    // unpack a stored VALUE payload into per-element signed codes (contract §5).
    // int8: byte per element; int4: nibble per element, sign-extended. Payload
    // lives in the low PAY_BITS of the record.
    reg [VECTOR_DIM*8-1:0] unpacked_codes;
    integer u;
    always @* begin
        unpacked_codes = '0;
        for (u = 0; u < VECTOR_DIM; u = u + 1) begin
            if (VAL_BPV == 8) begin
                unpacked_codes[u*8 +: 8] = sram_rd_data[u*8 +: 8];
            end else begin
                if (u[0] == 1'b0)
                    unpacked_codes[u*8 +: 8] = {{4{sram_rd_data[(u>>1)*8 + 3]}},
                                                 sram_rd_data[(u>>1)*8     +: 4]};
                else
                    unpacked_codes[u*8 +: 8] = {{4{sram_rd_data[(u>>1)*8 + 7]}},
                                                 sram_rd_data[(u>>1)*8 + 4 +: 4]};
            end
        end
    end

    // unpack a stored KEY record: sign-extend the per-channel INT4 codes to bytes
    // and lift the per-channel fp16 field straight out (feeds cq_key_path dequant).
    reg [VECTOR_DIM*8-1:0]            key_codes_ext;
    reg [VECTOR_DIM*COORD_WIDTH-1:0]  key_field_rd;
    integer w;
    always @* begin
        key_codes_ext = '0;
        key_field_rd  = '0;
        for (w = 0; w < VECTOR_DIM; w = w + 1) begin
            key_codes_ext[w*8 +: 8]              = {{4{sram_rd_data[w*4 + 3]}}, sram_rd_data[w*4 +: 4]};
            key_field_rd[w*COORD_WIDTH +: COORD_WIDTH] =
                sram_rd_data[KEY_CODES_BITS + w*COORD_WIDTH +: COORD_WIDTH];
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
            kp_in_valid      <= 1'b0;
            kp_group_start   <= 1'b0;
            kp_group_last    <= 1'b0;
            grp_tok_cnt      <= '0;
            grp_base         <= '0;
            out_is_key       <= 1'b0;
            kp_dec_codes     <= '0;
            kp_dec_scales    <= '0;
            idle             <= 1'b1;
        end else begin
            sram_wr_en     <= 1'b0;
            sram_rd_en     <= 1'b0;
            cqv_in_valid   <= 1'b0;
            kp_in_valid    <= 1'b0;
            kp_group_start <= 1'b0;
            kp_group_last  <= 1'b0;

            if (ctrl_reset) begin
                state       <= ST_IDLE;
                in_count    <= '0;
                grp_tok_cnt <= '0;
                idle        <= 1'b1;
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
                        idle         <= 1'b0;
                        // shift-fill (no indexed write): element k lands at [k*DW +: DW]
                        tok_vec      <= {s_axis_kv_tdata, tok_vec[VECTOR_DIM*COORD_WIDTH-1:COORD_WIDTH]};
                        in_count     <= 1;
                        input_is_key <= ~s_axis_kv_tuser;
                        if (KEY_GROUPED && ~s_axis_kv_tuser) begin
                            grp_base <= write_addr;     // new group base (grp_tok_cnt==0)
                            state    <= ST_KCOLLECT;
                        end else begin
                            state    <= ST_COLLECT;
                        end
                    end
                end

                // ---- VALUE path (and TIER-0 keys): per-token ----
                ST_COLLECT: begin
                    if (s_axis_kv_tvalid && s_axis_kv_tready) begin
                        tok_vec  <= {s_axis_kv_tdata, tok_vec[VECTOR_DIM*COORD_WIDTH-1:COORD_WIDTH]};
                        in_count <= in_count + 1;
                        if (s_axis_kv_tlast || in_count == VECTOR_DIM - 1) begin
                            state            <= ST_COMPRESS;
                            s_axis_kv_tready <= 1'b0;
                            cqv_in_valid     <= 1'b1;   // present token to the datapath
                        end
                    end
                end

                ST_COMPRESS: begin
                    if (cqv_out_valid) state <= ST_STORE;
                end

                ST_STORE: begin
                    sram_wr_en   <= 1'b1;
                    sram_wr_addr <= write_addr;
                    sram_wr_data <= '0;
                    sram_wr_data[PAY_BITS-1:0]          <= cqv_pay[PAY_BITS-1:0];
                    sram_wr_data[PAY_BITS +: SCALE_WIDTH] <= cqv_scale;
                    if (KEY_GROUPED) sram_wr_data[TAG_BIT] <= 1'b0;   // value record
                    in_count     <= '0;
                    state        <= ST_IDLE;
                    s_axis_kv_tready <= ctrl_enable;
                end

                // ---- grouped KEY path: collect G tokens, then emit records ----
                ST_KCOLLECT: begin
                    if (s_axis_kv_tvalid && s_axis_kv_tready) begin
                        tok_vec  <= {s_axis_kv_tdata, tok_vec[VECTOR_DIM*COORD_WIDTH-1:COORD_WIDTH]};
                        in_count <= in_count + 1;
                        if (s_axis_kv_tlast || in_count == VECTOR_DIM - 1) begin
                            state            <= ST_KFEED;
                            s_axis_kv_tready <= 1'b0;
                        end
                    end
                end

                ST_KFEED: begin
                    // present the assembled token to cq_key_path
                    kp_in_valid    <= 1'b1;
                    kp_group_start <= (grp_tok_cnt == '0);
                    kp_group_last  <= (grp_tok_cnt == GRP_LAST);
                    if (grp_tok_cnt == GRP_LAST) begin
                        grp_tok_cnt <= '0;
                        state       <= ST_KEMIT;
                    end else begin
                        grp_tok_cnt <= grp_tok_cnt + 1'b1;
                        in_count    <= '0;
                        state       <= ST_KACCEPT;
                    end
                end

                ST_KACCEPT: begin
                    // wait for the next key token of the group
                    s_axis_kv_tready <= ctrl_enable;
                    if (s_axis_kv_tvalid && s_axis_kv_tready) begin
                        tok_vec  <= {s_axis_kv_tdata, tok_vec[VECTOR_DIM*COORD_WIDTH-1:COORD_WIDTH]};
                        in_count <= 1;
                        state    <= ST_KCOLLECT;
                    end
                end

                ST_KEMIT: begin
                    // cq_key_path scales the group then pulses tok_valid per token.
                    if (kp_tok_valid) begin
                        sram_wr_en   <= 1'b1;
                        sram_wr_addr <= grp_base + kp_tok_idx;   // widen/truncate to ADDR_WIDTH
                        sram_wr_data <= '0;
                        sram_wr_data[KEY_CODES_BITS-1:0]                <= key_codes4;
                        sram_wr_data[KEY_CODES_BITS +: KEY_FIELD_BITS]  <= key_field16;
                        sram_wr_data[TAG_BIT]                           <= 1'b1;   // key record
                    end
                    if (kp_group_valid) begin
                        state            <= ST_IDLE;
                        s_axis_kv_tready <= ctrl_enable;
                    end
                end

                // ---- read / decompress ----
                ST_RLOAD: begin
                    sram_rd_en   <= 1'b1;
                    sram_rd_addr <= read_addr;
                    state        <= ST_RWAIT;
                end

                ST_RWAIT: begin
                    if (sram_rd_valid) begin
                        out_count  <= '0;
                        if (KEY_GROUPED && sram_rd_data[TAG_BIT]) begin
                            out_is_key    <= 1'b1;
                            kp_dec_codes  <= key_codes_ext;
                            kp_dec_scales <= key_field_rd;
                        end else begin
                            out_is_key <= 1'b0;
                            dec_codes  <= unpacked_codes;
                            dec_scale  <= sram_rd_data[PAY_BITS +: SCALE_WIDTH];
                        end
                        state <= ST_OUTPUT;
                    end
                end

                ST_OUTPUT: begin
                    // one fp32 beat per cycle (consumer holds tready; see IDLE clear)
                    m_axis_kv_tdata  <= out_is_key ? kp_dec_hat : dec_hat;
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
