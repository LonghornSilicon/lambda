`timescale 1ns / 1ps

module tb_realdata;

    localparam integer VECTOR_DIM  = 64;
    localparam integer COORD_WIDTH = 16;
    localparam integer SRAM_DEPTH  = 16;
    localparam integer CLK_PERIOD  = 10;
    // per-token compress latency: value path bit-serially divides D channels.
    localparam integer TOKWAIT     = 26*VECTOR_DIM + 128;

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

    // TIER=0 (CQ-8, per-token) so each streamed key token stores one record
    // (occupancy replay check). Grouped keys are covered by tb_top_stream.
    kv_cache_engine #(
        .VECTOR_DIM  (VECTOR_DIM),
        .TIER        (0),
        .SRAM_DEPTH  (SRAM_DEPTH),
        .COORD_WIDTH (COORD_WIDTH)
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

    reg [COORD_WIDTH-1:0] input_vectors [0:VECTOR_DIM*32-1];
    integer num_elements;

    task automatic axil_write(input [7:0] addr, input [31:0] data);
        @(posedge clk);
        axil_awaddr <= addr; axil_awvalid <= 1; axil_wdata <= data; axil_wvalid <= 1; axil_bready <= 1;
        @(posedge clk);
        axil_awvalid <= 0; axil_wvalid <= 0;
        @(posedge clk);
        axil_bready <= 0;
    endtask

    task automatic axil_read(input [7:0] addr, output [31:0] data);
        @(posedge clk);
        axil_araddr <= addr; axil_arvalid <= 1; axil_rready <= 1;
        @(posedge clk);
        axil_arvalid <= 0;
        @(posedge clk);
        data = axil_rdata;
        axil_rready <= 0;
    endtask

    // Shared vector buffer for streaming
    reg [COORD_WIDTH-1:0] vec_buf [0:VECTOR_DIM-1];

    task automatic stream_vec_buf(input is_value);
        integer idx;
        for (idx = 0; idx < VECTOR_DIM; idx = idx + 1) begin
            @(posedge clk);
            s_axis_kv_tdata  <= vec_buf[idx];
            s_axis_kv_tvalid <= 1;
            s_axis_kv_tlast  <= (idx == VECTOR_DIM - 1) ? 1 : 0;
            s_axis_kv_tuser  <= is_value;
            while (!s_axis_kv_tready) @(posedge clk);
        end
        @(posedge clk);
        s_axis_kv_tvalid <= 0;
        s_axis_kv_tlast  <= 0;
    endtask

    integer ii, t, jj;
    integer pass_count, test_count;
    reg [31:0] rd;

    initial begin
        axil_awaddr = 0; axil_awvalid = 0;
        axil_wdata = 0;  axil_wvalid = 0;
        axil_bready = 0;
        axil_araddr = 0; axil_arvalid = 0;
        axil_rready = 0;
        s_axis_kv_tdata = 0;
        s_axis_kv_tvalid = 0;
        s_axis_kv_tlast = 0;
        s_axis_kv_tuser = 0;
        m_axis_kv_tready = 1;

        pass_count = 0;
        test_count = 0;

        $readmemh("testvectors/input_vectors.hex", input_vectors);

        num_elements = 0;
        for (ii = 0; ii < VECTOR_DIM * 32; ii = ii + 1) begin
            if (input_vectors[ii] !== {COORD_WIDTH{1'bx}})
                num_elements = ii + 1;
        end

        $display("Loaded %0d elements (%0d vectors)",
                 num_elements, num_elements / VECTOR_DIM);

        rst_n = 0;
        repeat (TOKWAIT) @(posedge clk);
        rst_n = 1;
        repeat (TOKWAIT) @(posedge clk);

        axil_write(8'h00, 32'h0000_0002);

        for (t = 0; t < num_elements / VECTOR_DIM && t < SRAM_DEPTH; t = t + 1) begin
            axil_write(8'h28, t);

            for (jj = 0; jj < VECTOR_DIM; jj = jj + 1)
                vec_buf[jj] = input_vectors[t * VECTOR_DIM + jj];

            stream_vec_buf(0);
            repeat (TOKWAIT) @(posedge clk);
        end

        axil_read(8'h24, rd);
        test_count = test_count + 1;
        if (rd >= t) begin
            pass_count = pass_count + 1;
            $display("PASS: Occupancy %0d >= expected %0d", rd, t);
        end else begin
            $display("FAIL: Occupancy %0d < expected %0d", rd, t);
        end

        $display("\n============================================================");
        $display("Replay: %0d vectors streamed, %0d/%0d checks passed",
                 t, pass_count, test_count);
        if (pass_count == test_count)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

    initial begin
        #2000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
