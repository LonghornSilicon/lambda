// tb_value_path.sv — end-to-end streaming parity for cq_value_path (VALUE path).
//
// Streams each golden vector's value tokens through cq_value_path and checks,
// bit-for-bit vs the golden files:
//   - per-token fp16 scale      (val_scales)
//   - packed payload byte stream (val_payload; int4 D/2 B/token, int8 D B/token)
//   - reconstructed fp32 V_hat   (expected_v_hat) via the decompress path
// for all 9 vectors (values are per-token in every tier; bits=8 CQ-8 else 4).
// Two DUT instances (D=64, D=128); each vector is routed to the matching one.
//
// Run:  make sim_vpath   (rtl/Makefile)

`timescale 1ns/1ps
`include "cq_fp_pkg.sv"

module tb_value_path;
  import cq_fp_pkg::*;

  localparam int DW = 16;
  localparam string TVDIR = "tb/testvectors/channelquant/hex";

  reg clk = 0;
  always #5 clk = ~clk;
  reg rst_n;

  // ---- two value-path DUTs (D=64, D=128) ----
  reg  [3:0]        bits;
  reg               v64_iv;   reg [64*DW-1:0]  v64_ivec;  wire v64_ov;
  wire [DW-1:0]     v64_osc;  wire [64*8-1:0]  v64_ocode, v64_opay;
  reg  [64*8-1:0]   v64_dc;   reg [DW-1:0]     v64_ds;
  reg  [5:0]        v64_didx; wire [31:0]      v64_dh;
  cq_value_path #(.D(64)) dut64 (
    .clk(clk), .rst_n(rst_n), .bits(bits),
    .in_valid(v64_iv), .in_vec(v64_ivec), .busy(), .out_valid(v64_ov),
    .out_scale(v64_osc), .out_codes(v64_ocode), .out_pay(v64_opay),
    .dec_codes(v64_dc), .dec_scale(v64_ds), .dec_idx(v64_didx), .dec_hat(v64_dh));

  reg               v128_iv;  reg [128*DW-1:0] v128_ivec; wire v128_ov;
  wire [DW-1:0]     v128_osc; wire [128*8-1:0] v128_ocode, v128_opay;
  reg  [128*8-1:0]  v128_dc;  reg [DW-1:0]     v128_ds;
  reg  [6:0]        v128_didx; wire [31:0]     v128_dh;
  cq_value_path #(.D(128)) dut128 (
    .clk(clk), .rst_n(rst_n), .bits(bits),
    .in_valid(v128_iv), .in_vec(v128_ivec), .busy(), .out_valid(v128_ov),
    .out_scale(v128_osc), .out_codes(v128_ocode), .out_pay(v128_opay),
    .dec_codes(v128_dc), .dec_scale(v128_ds), .dec_idx(v128_didx), .dec_hat(v128_dh));

  // ---- vectors ----
  localparam int NVEC = 9;
  string vname [0:NVEC-1];
  int    vD[0:NVEC-1], vT[0:NVEC-1], vBits[0:NVEC-1];
  initial begin
    vname[0]="d64_T128_G64__CQ8";      vD[0]=64;  vT[0]=128; vBits[0]=8;
    vname[1]="d64_T128_G64__CQ4";      vD[1]=64;  vT[1]=128; vBits[1]=4;
    vname[2]="d64_T128_G64__CQ4plus";  vD[2]=64;  vT[2]=128; vBits[2]=4;
    vname[3]="d64_T70_G64__CQ8";       vD[3]=64;  vT[3]=70;  vBits[3]=8;
    vname[4]="d64_T70_G64__CQ4";       vD[4]=64;  vT[4]=70;  vBits[4]=4;
    vname[5]="d64_T70_G64__CQ4plus";   vD[5]=64;  vT[5]=70;  vBits[5]=4;
    vname[6]="d128_T100_G128__CQ8";    vD[6]=128; vT[6]=100; vBits[6]=8;
    vname[7]="d128_T100_G128__CQ4";    vD[7]=128; vT[7]=100; vBits[7]=4;
    vname[8]="d128_T100_G128__CQ4plus";vD[8]=128; vT[8]=100; vBits[8]=4;
  end

  localparam int MAXN = 128*128;
  logic [15:0] in_v  [0:MAXN-1];
  logic [15:0] g_vsc [0:MAXN-1];
  logic [7:0]  g_vpay[0:MAXN-1];
  logic [31:0] g_vhat[0:MAXN-1];

  int total_fail;

  // Run all T tokens of vector vi through the D=64 DUT; check scale/payload/hat.
  task automatic run64(input int vi, output int fails);
    int T, t, d, nb, pbyte; logic [15:0] sc;
    reg [64*8-1:0] codes; logic [7:0] by;
    int pidx;
    begin
      T=vT[vi]; fails=0; pidx=0; nb=(vBits[vi]==4)?32:64;
      for (t=0; t<T; t=t+1) begin
        @(negedge clk);
        v64_ivec = '0;
        for (d=0; d<64; d=d+1) v64_ivec[d*DW +: DW] = in_v[t*64+d];
        v64_iv = 1'b1;
        @(negedge clk); v64_iv = 1'b0;
        while (!v64_ov) @(negedge clk);
        sc = v64_osc; codes = v64_ocode;
        // scale
        if (sc !== g_vsc[t]) begin fails++; if (fails<=4)
          $display("  [%s] V scale tok %0d: got %04h exp %04h", vname[vi], t, sc, g_vsc[t]); end
        // payload bytes for this token
        for (pbyte=0; pbyte<nb; pbyte=pbyte+1) begin
          by = v64_opay[pbyte*8 +: 8];
          if (by !== g_vpay[pidx]) begin fails++; if (fails<=4)
            $display("  [%s] V pay byte %0d: got %02h exp %02h", vname[vi], pidx, by, g_vpay[pidx]); end
          pidx++;
        end
        // decompress this token's codes (one channel per dec_idx)
        v64_dc = codes; v64_ds = sc;
        for (d=0; d<64; d=d+1) begin
          v64_didx = d[5:0]; #1;
          if (v64_dh !== g_vhat[t*64+d]) begin fails++; if (fails<=4)
            $display("  [%s] V_hat (%0d,%0d): got %08h exp %08h", vname[vi], t, d,
                     v64_dh, g_vhat[t*64+d]); end
        end
      end
    end
  endtask

  task automatic run128(input int vi, output int fails);
    int T, t, d, nb, pbyte; logic [15:0] sc;
    reg [128*8-1:0] codes; logic [7:0] by;
    int pidx;
    begin
      T=vT[vi]; fails=0; pidx=0; nb=(vBits[vi]==4)?64:128;
      for (t=0; t<T; t=t+1) begin
        @(negedge clk);
        v128_ivec = '0;
        for (d=0; d<128; d=d+1) v128_ivec[d*DW +: DW] = in_v[t*128+d];
        v128_iv = 1'b1;
        @(negedge clk); v128_iv = 1'b0;
        while (!v128_ov) @(negedge clk);
        sc = v128_osc; codes = v128_ocode;
        if (sc !== g_vsc[t]) begin fails++; if (fails<=4)
          $display("  [%s] V scale tok %0d: got %04h exp %04h", vname[vi], t, sc, g_vsc[t]); end
        for (pbyte=0; pbyte<nb; pbyte=pbyte+1) begin
          by = v128_opay[pbyte*8 +: 8];
          if (by !== g_vpay[pidx]) begin fails++; if (fails<=4)
            $display("  [%s] V pay byte %0d: got %02h exp %02h", vname[vi], pidx, by, g_vpay[pidx]); end
          pidx++;
        end
        v128_dc = codes; v128_ds = sc;
        for (d=0; d<128; d=d+1) begin
          v128_didx = d[6:0]; #1;
          if (v128_dh !== g_vhat[t*128+d]) begin fails++; if (fails<=4)
            $display("  [%s] V_hat (%0d,%0d): got %08h exp %08h", vname[vi], t, d,
                     v128_dh, g_vhat[t*128+d]); end
        end
      end
    end
  endtask

  int vi, fv, n;
  initial begin
    rst_n=0; bits=4;
    v64_iv=0; v64_ivec=0; v64_dc=0; v64_ds=0; v64_didx=0;
    v128_iv=0; v128_ivec=0; v128_dc=0; v128_ds=0; v128_didx=0;
    repeat(3) @(posedge clk);
    rst_n=1; @(posedge clk);
    total_fail=0;

    for (vi=0; vi<NVEC; vi=vi+1) begin
      n = vD[vi]*vT[vi];
      $readmemh($sformatf("%s/%s/input_v.f16.hex",     TVDIR, vname[vi]), in_v,  0, n-1);
      $readmemh($sformatf("%s/%s/val_scales.f16.hex",  TVDIR, vname[vi]), g_vsc, 0, vT[vi]-1);
      $readmemh($sformatf("%s/%s/expected_v_hat.f32.hex",TVDIR, vname[vi]), g_vhat,0, n-1);
      if (vBits[vi]==8) $readmemh($sformatf("%s/%s/val_payload.u8.hex", TVDIR, vname[vi]), g_vpay, 0, n-1);
      else              $readmemh($sformatf("%s/%s/val_payload.u8.hex", TVDIR, vname[vi]), g_vpay, 0, (n+1)/2-1);

      bits = vBits[vi][3:0];
      if (vD[vi]==64) run64(vi, fv); else run128(vi, fv);

      $display("%-26s D=%0d T=%0d bits=%0d : V %s  (%0d)",
               vname[vi], vD[vi], vT[vi], vBits[vi], (fv==0)?"PASS":"FAIL", fv);
      total_fail += fv;
    end

    $display("============================================================");
    if (total_fail==0) $display("VALUE-PATH PARITY: ALL %0d VECTORS BIT-EXACT (scale+payload+V_hat)", NVEC);
    else               $display("VALUE-PATH PARITY: %0d TOTAL MISMATCHES", total_fail);
    $display("============================================================");
    $finish;
  end

endmodule
