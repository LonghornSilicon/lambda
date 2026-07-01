// tb_top_stream.sv — top-level end-to-end ChannelQuant parity through the AXI
// interfaces (per-token path). Drives kv_cache_engine (TIER=0, CQ-8) with a
// golden tensor over AXI-Stream write, triggers decompress-on-read per token,
// and checks the fp32 output vs expected_v_hat / expected_k_hat (CQ-8 keys are
// per-token, same datapath). Proves the codec is correctly wired into the top
// FSM + SRAM, not just the datapath modules in isolation.
//
// Run:  make sim_top   (rtl/Makefile)

`timescale 1ns/1ps
`include "cq_fp_pkg.sv"

module tb_top_stream;
  import cq_fp_pkg::*;

  localparam int D = 64, T = 128;
  localparam int DEPTH = 128;
  localparam string TV = "tb/testvectors/channelquant/hex/d64_T128_G64__CQ8";

  reg clk=0; always #5 clk=~clk;
  reg rst_n;

  // AXI-Lite
  reg  [7:0] awaddr; reg awvalid; wire awready;
  reg  [31:0] wdata; reg wvalid; wire wready;
  wire [1:0] bresp; wire bvalid; reg bready;
  reg  [7:0] araddr; reg arvalid; wire arready;
  wire [31:0] rdata; wire [1:0] rresp; wire rvalid; reg rready;
  // AXI-Stream
  reg  [15:0] s_tdata; reg s_tvalid; wire s_tready; reg s_tlast, s_tuser;
  wire [31:0] m_tdata; wire m_tvalid; reg m_tready; wire m_tlast;
  wire evict_needed; wire [$clog2(DEPTH)-1:0] evict_addr;

  kv_cache_engine #(.VECTOR_DIM(D), .TIER(0), .SRAM_DEPTH(DEPTH)) dut (
    .clk(clk), .rst_n(rst_n),
    .axil_awaddr(awaddr), .axil_awvalid(awvalid), .axil_awready(awready),
    .axil_wdata(wdata), .axil_wvalid(wvalid), .axil_wready(wready),
    .axil_bresp(bresp), .axil_bvalid(bvalid), .axil_bready(bready),
    .axil_araddr(araddr), .axil_arvalid(arvalid), .axil_arready(arready),
    .axil_rdata(rdata), .axil_rresp(rresp), .axil_rvalid(rvalid), .axil_rready(rready),
    .s_axis_kv_tdata(s_tdata), .s_axis_kv_tvalid(s_tvalid), .s_axis_kv_tready(s_tready),
    .s_axis_kv_tlast(s_tlast), .s_axis_kv_tuser(s_tuser),
    .m_axis_kv_tdata(m_tdata), .m_axis_kv_tvalid(m_tvalid), .m_axis_kv_tready(m_tready),
    .m_axis_kv_tlast(m_tlast), .evict_needed(evict_needed), .evict_addr(evict_addr)
  );

  logic [15:0] in_v [0:D*T-1];
  logic [15:0] in_k [0:D*T-1];
  logic [31:0] vhat [0:D*T-1];
  logic [31:0] khat [0:D*T-1];
  logic [31:0] outb [0:D-1];

  task automatic awrite(input [7:0] a, input [31:0] dv);
    begin
      @(negedge clk); awaddr=a; wdata=dv; awvalid=1; wvalid=1;
      @(negedge clk); awvalid=0; wvalid=0;
    end
  endtask

  // stream one token (D fp16 elems) at write_addr `addr`; is_v selects K/V.
  // Negedge-driven AXI master: stimulus stable across the posedge; a beat
  // advances only when tready is seen high at the negedge (avoids the spurious
  // extra beat that would fire if tvalid were held during the store).
  task automatic wr_token(input int base, input int addr, input logic is_v, input logic key);
    int d;
    begin
      awrite(8'h28, addr);              // WRITE_ADDR
      d=0;
      while (d<D) begin
        @(negedge clk);
        s_tdata = key ? in_k[base+d] : in_v[base+d];
        s_tvalid=1; s_tuser=is_v; s_tlast=(d==D-1);
        if (s_tready) d=d+1;            // this beat will be accepted at the posedge
      end
      @(negedge clk); s_tvalid=0; s_tlast=0;
      // wait until the engine is ready again (compress + store complete)
      while (!s_tready) @(negedge clk);
    end
  endtask

  // trigger decompress-on-read of `addr`; collect D fp32 beats (negedge-sampled)
  task automatic rd_token(input int addr);
    int d;
    begin
      awrite(8'h2C, addr);              // READ_ADDR launches a decompress
      d=0;
      while (d<D) begin
        @(negedge clk);
        if (m_tvalid) begin outb[d]=m_tdata; d=d+1; end
      end
    end
  endtask

  integer vi, t, d, fails;
  initial begin
    rst_n=0; awvalid=0; wvalid=0; bready=1; araddr=0; arvalid=0; rready=1;
    s_tdata=0; s_tvalid=0; s_tlast=0; s_tuser=0; m_tready=1;
    $readmemh({TV,"/input_v.f16.hex"},        in_v);
    $readmemh({TV,"/input_k.f16.hex"},        in_k);
    $readmemh({TV,"/expected_v_hat.f32.hex"}, vhat);
    $readmemh({TV,"/expected_k_hat.f32.hex"}, khat);
    repeat(4) @(posedge clk); rst_n=1; @(posedge clk);
    awrite(8'h00, 32'h2);   // enable
    fails=0;

    // ---- VALUES: write all T tokens, read back, check ----
    for (t=0; t<T; t=t+1) wr_token(t*D, t, 1'b1, 1'b0);
    for (t=0; t<T; t=t+1) begin
      rd_token(t);
      for (d=0; d<D; d=d+1)
        if (outb[d] !== vhat[t*D+d]) begin fails++; if (fails<=6)
          $display("  V_hat (t%0d,d%0d): got %08h exp %08h", t, d, outb[d], vhat[t*D+d]); end
    end

    // ---- KEYS (CQ-8 per-token): same path ----
    for (t=0; t<T; t=t+1) wr_token(t*D, t, 1'b0, 1'b1);
    for (t=0; t<T; t=t+1) begin
      rd_token(t);
      for (d=0; d<D; d=d+1)
        if (outb[d] !== khat[t*D+d]) begin fails++; if (fails<=6)
          $display("  K_hat (t%0d,d%0d): got %08h exp %08h", t, d, outb[d], khat[t*D+d]); end
    end

    $display("============================================================");
    if (fails==0) $display("TOP-STREAM PARITY: CQ-8 K+V bit-exact through the top (D=%0d T=%0d)", D, T);
    else          $display("TOP-STREAM PARITY: %0d MISMATCHES", fails);
    $display("============================================================");
    $finish;
  end

  // safety timeout
  initial begin #20000000; $display("TIMEOUT"); $finish; end

endmodule
