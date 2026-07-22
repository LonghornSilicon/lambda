// tb_top_stream.sv — top-level end-to-end ChannelQuant parity through the AXI
// interfaces, on the full CQ-4+ config (TIER=2, D=64, G=64, k=2). One DUT, two
// checks:
//
//   (A) per-token INT4 VALUES: stream value tokens through cq_value_path in the
//       top, decompress-on-read, check fp32 V_hat bit-exact vs expected_v_hat.
//
//   (B) grouped per-channel INT4 KEYS (+ k=2 fp16 outliers): stream one full key
//       group (G=64 tokens) through cq_key_path in the top, wait for the group to
//       flush, decompress-on-read each token, check fp32 K_hat bit-exact vs golden
//       — keep channels via per-channel dequant, outlier channels via the unified
//       {fp16 field, code +1} widen. Proves the grouped key path + outlier lane +
//       SRAM layout are correctly wired into the top FSM.
//
// Run:  make sim_top   (rtl/Makefile)

`timescale 1ns/1ps
`include "cq_fp_pkg.sv"

module tb_top_stream;
  import cq_fp_pkg::*;

  localparam int D = 64, T = 128, G = 64;
  localparam int TA = 8;     // per-token value tokens checked (value path already broadly proven)
  localparam int DEPTH = 128;
  localparam string TV   = "tb/testvectors/channelquant/hex/d64_T128_G64__CQ4plus";
  localparam string KMASK= "tb/testvectors/channelquant/hex/d64_T128_G64__CQ4plus/outlier_mask.u8.hex";

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

  kv_cache_engine #(.VECTOR_DIM(D), .TIER(2), .KEY_GROUP(G), .OUTLIER_K(2),
                    .SRAM_DEPTH(DEPTH), .MASK_FILE(KMASK)) dut (
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

  // stream one VALUE token (D fp16 elems) at write_addr `addr`
  task automatic wr_value(input int base, input int addr);
    int d;
    begin
      awrite(8'h28, addr);              // WRITE_ADDR
      d=0;
      while (d<D) begin
        @(negedge clk);
        s_tdata = in_v[base+d];
        s_tvalid=1; s_tuser=1'b1; s_tlast=(d==D-1);   // tuser=1 -> VALUE
        if (s_tready) d=d+1;
      end
      @(negedge clk); s_tvalid=0; s_tlast=0;
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

  // stream a full key group (n_tok tokens, D beats each) back-to-back
  task automatic stream_key_group(input int tok_base, input int n_tok);
    int t, d;
    begin
      for (t=0; t<n_tok; t=t+1) begin
        d=0;
        while (d<D) begin
          @(negedge clk);
          s_tdata = in_k[(tok_base+t)*D + d];
          s_tvalid=1; s_tuser=1'b0; s_tlast=(d==D-1);   // tuser=0 -> KEY
          if (s_tready) d=d+1;
        end
      end
      @(negedge clk); s_tvalid=0; s_tlast=0;
    end
  endtask

  // poll STATUS.idle until the FSM returns to IDLE (group flush complete)
  task automatic wait_idle;
    int guard; reg [31:0] st;
    begin
      guard=0; st=0;
      while (!st[0] && guard<2000000) begin
        @(negedge clk); araddr=8'h04; arvalid=1; rready=1;
        @(negedge clk); arvalid=0;
        @(negedge clk); st=rdata; rready=0;
        guard=guard+1;
      end
    end
  endtask

  integer t, d, vf, kf;
  initial begin
    rst_n=0; awaddr=0; awvalid=0; wvalid=0; wdata=0; bready=1;
    araddr=0; arvalid=0; rready=1; s_tdata=0; s_tvalid=0; s_tlast=0; s_tuser=0; m_tready=1;
    $readmemh({TV,"/input_v.f16.hex"},        in_v);
    $readmemh({TV,"/input_k.f16.hex"},        in_k);
    $readmemh({TV,"/expected_v_hat.f32.hex"}, vhat);
    $readmemh({TV,"/expected_k_hat.f32.hex"}, khat);
    repeat(4) @(posedge clk); rst_n=1; @(posedge clk);
    vf=0; kf=0;
    awrite(8'h00, 32'h2);   // enable

    // ---- (A) per-token INT4 VALUES ----
    for (t=0; t<TA; t=t+1) wr_value(t*D, t);
    for (t=0; t<TA; t=t+1) begin
      rd_token(t);
      for (d=0; d<D; d=d+1)
        if (outb[d] !== vhat[t*D+d]) begin vf++; if (vf<=6)
          $display("  V_hat (t%0d,d%0d): got %08h exp %08h", t, d, outb[d], vhat[t*D+d]); end
    end
    // ---- (B) grouped per-channel INT4 KEYS (CQ-4+) ----
    awrite(8'h28, 32'h0);   // WRITE_ADDR = 0 (group base)
    stream_key_group(0, G); // one full group of G=64 key tokens
    wait_idle();            // wait for the group to flush (records written, FSM idle)
    for (t=0; t<G; t=t+1) begin
      rd_token(t);          // decompress token t (addr t)
      for (d=0; d<D; d=d+1)
        if (outb[d] !== khat[t*D+d]) begin kf++; if (kf<=8)
          $display("  [CQ4+] K_hat (t%0d,d%0d): got %08h exp %08h", t, d, outb[d], khat[t*D+d]); end
    end

    $display("============================================================");
    if (vf==0) $display("TOP-STREAM (A) per-token INT4 V bit-exact through the top (D=%0d)", D);
    else       $display("TOP-STREAM (A): %0d MISMATCHES", vf);
    if (kf==0) $display("TOP-STREAM (B) grouped CQ-4+ keys bit-exact through the top (D=%0d G=%0d k=2)", D, G);
    else       $display("TOP-STREAM (B): %0d MISMATCHES", kf);
    if (vf==0 && kf==0) $display("TOP-STREAM PARITY: ALL PASS");
    $display("============================================================");
    $finish;
  end

  // safety timeout
  initial begin #400000000; $display("TIMEOUT"); $finish; end

endmodule
