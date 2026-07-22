// tb_amax_unit.sv — streaming parity for amax_unit vs the golden scales.
//
// Drives amax_unit token-by-token (value: per-token; key: per-channel over a
// group, with partial final groups) and feeds its amax through the proven
// cq_scale_unit, asserting the resulting fp16 scale equals the golden
// val_scales / key_scales byte-for-byte for all 9 vectors. This proves the
// streaming scale front-end (the P2 amax reduction + group FSM control) matches
// the contract, reusing the P3-proven amax->scale conversion.
//
// Run:  make sim_amax   (rtl/Makefile)

`timescale 1ns/1ps
`include "cq_fp_pkg.sv"

module tb_amax_unit;
  import cq_fp_pkg::*;

  localparam int MAXD = 128;
  localparam int DW   = 16;
  localparam string TVDIR = "tb/testvectors/channelquant/hex";

  // ---- clock ----
  reg clk = 0;
  always #5 clk = ~clk;

  // ---- DUT (sized to max D; unused upper channels driven 0) ----
  reg               in_valid, mode_channel, group_start, group_done;
  reg  [MAXD*DW-1:0] vec;
  wire [DW-1:0]     scale_token;
  wire [MAXD*DW-1:0] scale_chan;
  wire              out_valid;
  reg               rst_n;

  amax_unit #(.DIM(MAXD), .DW(DW)) dut (
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .vec(vec),
    .mode_channel(mode_channel), .group_start(group_start), .group_done(group_done),
    .scale_token(scale_token), .scale_chan(scale_chan), .out_valid(out_valid)
  );

  // ---- amax -> fp16 scale (proven core) ----
  reg  [15:0] u_amax;  reg [3:0] u_bits;  wire [15:0] u_scale;
  cq_scale_unit u_scale_i (.amax_f16(u_amax), .bits(u_bits), .scale_f16(u_scale));

  // ---- vector table (same 9 as tb_channelquant) ----
  localparam int NVEC = 9;
  string vname [0:NVEC-1];
  int    vD[0:NVEC-1], vT[0:NVEC-1], vBits[0:NVEC-1], vG[0:NVEC-1], vTier[0:NVEC-1];
  initial begin
    vname[0]="d64_T128_G64__CQ8";      vD[0]=64;  vT[0]=128; vBits[0]=8; vG[0]=0;   vTier[0]=0;
    vname[1]="d64_T128_G64__CQ4";      vD[1]=64;  vT[1]=128; vBits[1]=4; vG[1]=64;  vTier[1]=1;
    vname[2]="d64_T128_G64__CQ4plus";  vD[2]=64;  vT[2]=128; vBits[2]=4; vG[2]=64;  vTier[2]=2;
    vname[3]="d64_T70_G64__CQ8";       vD[3]=64;  vT[3]=70;  vBits[3]=8; vG[3]=0;   vTier[3]=0;
    vname[4]="d64_T70_G64__CQ4";       vD[4]=64;  vT[4]=70;  vBits[4]=4; vG[4]=64;  vTier[4]=1;
    vname[5]="d64_T70_G64__CQ4plus";   vD[5]=64;  vT[5]=70;  vBits[5]=4; vG[5]=64;  vTier[5]=2;
    vname[6]="d128_T100_G128__CQ8";    vD[6]=128; vT[6]=100; vBits[6]=8; vG[6]=0;   vTier[6]=0;
    vname[7]="d128_T100_G128__CQ4";    vD[7]=128; vT[7]=100; vBits[7]=4; vG[7]=128; vTier[7]=1;
    vname[8]="d128_T100_G128__CQ4plus";vD[8]=128; vT[8]=100; vBits[8]=4; vG[8]=128; vTier[8]=2;
  end

  localparam int MAXN = 128*128;
  logic [15:0] in_v  [0:MAXN-1];
  logic [15:0] in_k  [0:MAXN-1];
  logic [15:0] g_vsc [0:MAXN-1];
  logic [15:0] g_ksc [0:MAXN-1];
  logic [7:0]  mask  [0:MAXD-1];
  int keep [0:MAXD-1];

  int total_fail;

  // build the packed vector for token t (channel d at [d*DW +: DW]); pad upper 0
  task automatic load_vec(input int base, input int D, input logic key);
    int d;
    begin
      vec = '0;
      for (d=0; d<D; d=d+1) vec[d*DW +: DW] = key ? in_k[base+d] : in_v[base+d];
    end
  endtask

  // present one token for a cycle. Stimulus changes on negedge so inputs are
  // stable across the posedge (avoids the TB/DUT active-region race on in_valid).
  task automatic step(input logic mc, input logic gs, input logic gd);
    begin
      @(negedge clk);
      in_valid = 1'b1; mode_channel = mc; group_start = gs; group_done = gd;
      @(negedge clk);
      in_valid = 1'b0; group_start = 1'b0; group_done = 1'b0;
    end
  endtask

  // ---- per-token scale check (values all tiers; CQ-8 keys) ----
  task automatic check_pertoken(input int vi, input logic key, input int B, output int fails);
    int D, T, t; logic [15:0] amax, sc, exp;
    begin
      D=vD[vi]; T=vT[vi]; fails=0;
      for (t=0; t<T; t=t+1) begin
        load_vec(t*D, D, key);
        step(1'b0, 1'b0, 1'b0);       // value mode, 1-cycle latency
        #1;
        amax = scale_token;
        u_amax = amax; u_bits = B[3:0]; #1; sc = u_scale;
        exp = key ? g_ksc[t] : g_vsc[t];
        if (sc !== exp) begin
          fails++;
          if (fails<=4) $display("  [%s] %s scale tok %0d: got %04h exp %04h",
                                 vname[vi], key?"K":"V", t, sc, exp);
        end
      end
    end
  endtask

  // ---- per-channel grouped key scale check (CQ-4 / CQ-4+) ----
  task automatic check_keys(input int vi, output int fails);
    int D, T, G, nk, c, a, b, t, ci, sc_base;
    logic [15:0] amax, sc;
    begin
      D=vD[vi]; T=vT[vi]; G=vG[vi]; fails=0;
      // keep[] = non-outlier channels (mask=1 -> outlier)
      nk=0;
      for (c=0;c<D;c=c+1) if (!mask[c][0]) begin keep[nk]=c; nk++; end

      sc_base=0;
      for (a=0; a<T; a=a+G) begin
        b=a+G; if (b>T) b=T;
        for (t=a; t<b; t=t+1) begin
          load_vec(t*D, D, 1'b1);
          step(1'b1, (t==a), (t==b-1));   // key mode; mark group start/end
        end
        #1;   // scale_chan frozen at group_done edge
        for (ci=0; ci<nk; ci=ci+1) begin
          amax = scale_chan[keep[ci]*DW +: DW];
          u_amax = amax; u_bits = 4'd4; #1; sc = u_scale;
          if (sc !== g_ksc[sc_base+ci]) begin
            fails++;
            if (fails<=6) $display("  [%s] key scale grp@%0d ch %0d: got %04h exp %04h",
                                   vname[vi], a, keep[ci], sc, g_ksc[sc_base+ci]);
          end
        end
        sc_base += nk;
      end
    end
  endtask

  int vi, fv, fk;
  initial begin
    rst_n=0; in_valid=0; mode_channel=0; group_start=0; group_done=0; vec='0;
    repeat(3) @(posedge clk);
    rst_n=1; @(posedge clk);
    total_fail=0;

    for (vi=0; vi<NVEC; vi=vi+1) begin
      $readmemh($sformatf("%s/%s/input_v.f16.hex",    TVDIR, vname[vi]), in_v,  0, vD[vi]*vT[vi]-1);
      $readmemh($sformatf("%s/%s/val_scales.f16.hex", TVDIR, vname[vi]), g_vsc, 0, vT[vi]-1);
      $readmemh($sformatf("%s/%s/input_k.f16.hex",    TVDIR, vname[vi]), in_k,  0, vD[vi]*vT[vi]-1);

      // key_scales length: per-token (CQ-8) or per-group*nk (CQ-4/+)
      begin
        int nk, c, slen, ga, gb;
        if (vTier[vi]==0) begin
          $readmemh($sformatf("%s/%s/key_scales.f16.hex", TVDIR, vname[vi]), g_ksc, 0, vT[vi]-1);
        end else begin
          $readmemh($sformatf("%s/%s/outlier_mask.u8.hex", TVDIR, vname[vi]), mask, 0, vD[vi]-1);
          nk=0; for (c=0;c<vD[vi];c=c+1) if (!mask[c][0]) nk++;
          slen=0; for (ga=0; ga<vT[vi]; ga=ga+vG[vi]) begin gb=ga+vG[vi]; if (gb>vT[vi]) gb=vT[vi]; slen+=nk; end
          $readmemh($sformatf("%s/%s/key_scales.f16.hex", TVDIR, vname[vi]), g_ksc, 0, slen-1);
        end
      end

      check_pertoken(vi, 1'b0, vBits[vi], fv);          // values
      if (vTier[vi]==0) check_pertoken(vi, 1'b1, 8, fk); // CQ-8 keys per-token
      else              check_keys(vi, fk);              // CQ-4/+ keys per-channel

      $display("%-26s D=%0d T=%0d G=%0d tier=%0d : V %s / K %s  (%0d+%0d)",
               vname[vi], vD[vi], vT[vi], vG[vi], vTier[vi],
               (fv==0)?"PASS":"FAIL", (fk==0)?"PASS":"FAIL", fv, fk);
      total_fail += fv + fk;
    end

    $display("============================================================");
    if (total_fail==0) $display("AMAX_UNIT PARITY: ALL %0d VECTORS BIT-EXACT (V+K scales)", NVEC);
    else               $display("AMAX_UNIT PARITY: %0d TOTAL SCALE MISMATCHES", total_fail);
    $display("============================================================");
    $finish;
  end

endmodule
