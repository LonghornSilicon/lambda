`timescale 1ns / 1ps

module tb_kv_cache_engine;

    localparam integer VECTOR_DIM    = 64;
    localparam integer COORD_WIDTH   = 16;
    localparam integer SCALE_WIDTH   = 16;
    // Plumbing/occupancy TB: uses TIER=0 (CQ-8, per-token K and V) so each streamed
    // key token stores one record (occupancy +1/token). Grouped per-channel INT4
    // keys (TIER 1/2) are exercised bit-exact by tb_top_stream (make sim_top).
    localparam integer TIER          = 0;    // CQ-8 (per-token K int8 / per-token V int8)
    localparam integer KEY_GROUP     = 128;  // G
    localparam integer OUTLIER_K     = 0;
    localparam integer SRAM_DEPTH    = 16;
    localparam integer CLK_PERIOD    = 10;
    // per-token compress latency: the value path bit-serially divides each of the
    // D channels (~24 cycles/channel), so wait generously before occupancy checks.
    localparam integer TOKWAIT       = 26*VECTOR_DIM + 128;

    reg  clk = 0;
    reg  rst_n = 0;

    reg  [7:0]  axil_awaddr;
    reg         axil_awvalid;
    wire        axil_awready;
    reg  [31:0] axil_wdata;
    reg         axil_wvalid;
    wire        axil_wready;
    wire [1:0]  axil_bresp;
    wire        axil_bvalid;
    reg         axil_bready;
    reg  [7:0]  axil_araddr;
    reg         axil_arvalid;
    wire        axil_arready;
    wire [31:0] axil_rdata;
    wire [1:0]  axil_rresp;
    wire        axil_rvalid;
    reg         axil_rready;

    reg  [COORD_WIDTH-1:0] s_axis_kv_tdata;
    reg                    s_axis_kv_tvalid;
    wire                   s_axis_kv_tready;
    reg                    s_axis_kv_tlast;
    reg                    s_axis_kv_tuser;

    wire [31:0]            m_axis_kv_tdata;   // fp32 decompressed output (contract §1)
    wire                   m_axis_kv_tvalid;
    reg                    m_axis_kv_tready;
    wire                   m_axis_kv_tlast;

    wire evict_needed;
    wire [$clog2(SRAM_DEPTH)-1:0] evict_addr;

    kv_cache_engine #(
        .VECTOR_DIM    (VECTOR_DIM),
        .TIER          (TIER),
        .KEY_GROUP     (KEY_GROUP),
        .OUTLIER_K     (OUTLIER_K),
        .SCALE_WIDTH   (SCALE_WIDTH),
        .SRAM_DEPTH    (SRAM_DEPTH),
        .COORD_WIDTH   (COORD_WIDTH)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .axil_awaddr     (axil_awaddr),
        .axil_awvalid    (axil_awvalid),
        .axil_awready    (axil_awready),
        .axil_wdata      (axil_wdata),
        .axil_wvalid     (axil_wvalid),
        .axil_wready     (axil_wready),
        .axil_bresp      (axil_bresp),
        .axil_bvalid     (axil_bvalid),
        .axil_bready     (axil_bready),
        .axil_araddr     (axil_araddr),
        .axil_arvalid    (axil_arvalid),
        .axil_arready    (axil_arready),
        .axil_rdata      (axil_rdata),
        .axil_rresp      (axil_rresp),
        .axil_rvalid     (axil_rvalid),
        .axil_rready     (axil_rready),
        .s_axis_kv_tdata  (s_axis_kv_tdata),
        .s_axis_kv_tvalid (s_axis_kv_tvalid),
        .s_axis_kv_tready (s_axis_kv_tready),
        .s_axis_kv_tlast  (s_axis_kv_tlast),
        .s_axis_kv_tuser  (s_axis_kv_tuser),
        .m_axis_kv_tdata  (m_axis_kv_tdata),
        .m_axis_kv_tvalid (m_axis_kv_tvalid),
        .m_axis_kv_tready (m_axis_kv_tready),
        .m_axis_kv_tlast  (m_axis_kv_tlast),
        .evict_needed    (evict_needed),
        .evict_addr      (evict_addr)
    );

    always #(CLK_PERIOD/2) clk = ~clk;

    integer test_count = 0;
    integer pass_count = 0;

    // Shared test vector buffer
    reg [COORD_WIDTH-1:0] test_vec [0:VECTOR_DIM-1];

    task automatic check(input string name, input logic cond);
        test_count = test_count + 1;
        if (cond) begin
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: %s", name);
        end
    endtask

    task automatic axil_write(input [7:0] addr, input [31:0] data);
        @(posedge clk);
        axil_awaddr  <= addr;
        axil_awvalid <= 1'b1;
        axil_wdata   <= data;
        axil_wvalid  <= 1'b1;
        axil_bready  <= 1'b1;
        @(posedge clk);
        axil_awvalid <= 1'b0;
        axil_wvalid  <= 1'b0;
        @(posedge clk);
        axil_bready  <= 1'b0;
    endtask

    task automatic axil_read(input [7:0] addr, output [31:0] data);
        @(posedge clk);
        axil_araddr  <= addr;
        axil_arvalid <= 1'b1;
        axil_rready  <= 1'b1;
        @(posedge clk);
        axil_arvalid <= 1'b0;
        @(posedge clk);
        data = axil_rdata;
        axil_rready  <= 1'b0;
    endtask

    // Stream vector from test_vec buffer (avoids passing arrays to tasks)
    task automatic stream_from_buf(input is_value);
        integer idx;
        for (idx = 0; idx < VECTOR_DIM; idx = idx + 1) begin
            @(posedge clk);
            s_axis_kv_tdata  <= test_vec[idx];
            s_axis_kv_tvalid <= 1'b1;
            s_axis_kv_tlast  <= (idx == VECTOR_DIM - 1) ? 1'b1 : 1'b0;
            s_axis_kv_tuser  <= is_value;
            while (!s_axis_kv_tready) @(posedge clk);
        end
        @(posedge clk);
        s_axis_kv_tvalid <= 1'b0;
        s_axis_kv_tlast  <= 1'b0;
    endtask

    // LCG
    reg [63:0] lcg_state;
    integer ii;

    reg [31:0] read_data;

    initial begin
        rst_n = 0;
        axil_awaddr  = 0;  axil_awvalid = 0;
        axil_wdata   = 0;  axil_wvalid  = 0;
        axil_bready  = 0;
        axil_araddr  = 0;  axil_arvalid = 0;
        axil_rready  = 0;
        s_axis_kv_tdata  = 0;
        s_axis_kv_tvalid = 0;
        s_axis_kv_tlast  = 0;
        s_axis_kv_tuser  = 0;
        m_axis_kv_tready = 1;

        repeat (TOKWAIT) @(posedge clk);
        rst_n = 1;
        repeat (TOKWAIT) @(posedge clk);

        // Test: INFO register reads
        $display("\n[Test: INFO registers]");

        axil_read(8'h08, read_data);
        check("INFO_DIM == 64", read_data == VECTOR_DIM);

        axil_read(8'h0C, read_data);
        check("INFO_TIER == CQ-8", read_data == TIER);

        axil_read(8'h10, read_data);
        check("INFO_GROUP == 128", read_data == KEY_GROUP);

        axil_read(8'h14, read_data);
        check("INFO_SRAM_DEPTH == 16", read_data == SRAM_DEPTH);

        axil_read(8'h20, read_data);
        check("INFO_VERSION == v0.2", read_data == 32'h00020000);

        axil_read(8'h3C, read_data);
        check("INFO_OUTLIER_K == 0", read_data == OUTLIER_K);

        axil_read(8'h40, read_data);
        check("INFO_SCALE_DEPTH == D", read_data == VECTOR_DIM);

        axil_read(8'h44, read_data);
        check("INFO_RESID_DEPTH == G", read_data == KEY_GROUP);

        // Test: STATUS idle
        $display("\n[Test: STATUS register]");
        axil_read(8'h04, read_data);
        check("STATUS.idle == 1", read_data[0] == 1'b1);

        // Test: Zero key vector
        $display("\n[Test: Zero key vector stream]");
        axil_write(8'h00, 32'h0000_0002); // enable
        axil_write(8'h28, 32'h0000_0000); // write_addr = 0

        for (ii = 0; ii < VECTOR_DIM; ii = ii + 1)
            test_vec[ii] = 0;
        stream_from_buf(0);
        repeat (TOKWAIT) @(posedge clk);

        axil_read(8'h24, read_data);
        check("Occupancy >= 1", read_data >= 1);

        // Test: Ramp key vector
        $display("\n[Test: Ramp key vector stream]");
        axil_write(8'h28, 32'h0000_0001);

        for (ii = 0; ii < VECTOR_DIM; ii = ii + 1)
            test_vec[ii] = (ii - VECTOR_DIM/2) * 100;
        stream_from_buf(0);
        repeat (TOKWAIT) @(posedge clk);

        axil_read(8'h24, read_data);
        check("Occupancy >= 2", read_data >= 2);

        // Test: Value vector
        $display("\n[Test: Value vector stream]");
        axil_write(8'h28, 32'h0000_0002);

        lcg_state = 64'd77777;
        for (ii = 0; ii < VECTOR_DIM; ii = ii + 1) begin
            lcg_state = lcg_state * 64'h5851F42D4C957F2D + 64'h14057B7EF767814F;
            test_vec[ii] = lcg_state[63:48];
        end
        stream_from_buf(1);
        repeat (TOKWAIT) @(posedge clk);

        axil_read(8'h24, read_data);
        check("Occupancy >= 3", read_data >= 3);

        // Test: Soft reset
        $display("\n[Test: Soft reset]");
        axil_write(8'h00, 32'h0000_0001);
        repeat (TOKWAIT) @(posedge clk);
        axil_read(8'h04, read_data);
        check("idle after reset", read_data[0] == 1'b1);

        // Test: IRQ registers
        $display("\n[Test: IRQ registers]");
        axil_write(8'h34, 32'h0000_000F);
        axil_read(8'h34, read_data);
        check("IRQ_MASK readback", read_data[3:0] == 4'hF);

        // Test: Multiple stores
        $display("\n[Test: Multiple stores]");
        axil_write(8'h00, 32'h0000_0002);

        for (ii = 0; ii < 10; ii = ii + 1) begin
            axil_write(8'h28, ii);
            lcg_state = 64'd10000 + ii;
            begin : gen_vec
                integer jj;
                for (jj = 0; jj < VECTOR_DIM; jj = jj + 1) begin
                    lcg_state = lcg_state * 64'h5851F42D4C957F2D + 64'h14057B7EF767814F;
                    test_vec[jj] = lcg_state[63:48];
                end
            end
            stream_from_buf(0);
            repeat (TOKWAIT) @(posedge clk);
        end

        axil_read(8'h24, read_data);
        check("Occupancy >= 10", read_data >= 10);

        // Test: Compression ratio constants
        $display("\n[Test: Compression ratio constants]");
        axil_read(8'h18, read_data);
        check("CR_K > 0", read_data > 0);
        axil_read(8'h1C, read_data);
        check("CR_V > 0", read_data > 0);

        // Summary
        $display("\n============================================================");
        $display("%0d/%0d tests passed", pass_count, test_count);
        if (pass_count == test_count)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

    initial begin
        #1000000;
        $display("TIMEOUT after 1ms");
        $finish;
    end

endmodule
