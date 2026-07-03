// tb_key_path.sv — end-to-end streaming parity for cq_key_path (KEY path).
//
// Streams each per-channel golden vector's key tokens through cq_key_path,
// group by group (full g=G and partial g<G), and checks bit-for-bit vs golden:
//   - D per-channel fp16 scales   (key_scales, keep channels)
//   - packed INT4 payload stream   (key_payload)
//   - reconstructed fp32 K_hat     (expected_k_hat): keep channels via the DUT's
//     dequant, outlier channels via the FP16 sidecar identity (the top's job)
//   - outlier sidecar              (sidecar == input fp16 at outlier channels)
// for the 6 CQ-4 / CQ-4+ vectors (CQ-8 keys are per-token -> cq_value_path).
//
// One D=128 DUT covers both head dims: for D=64 vectors the upper 64 channels
// are masked as outliers (excluded from the INT4 path), so keep == the real
// D=64 keep set. Run:  make sim_kpath   (rtl/Makefile)

`timescale 1ns/1ps
`include "cq_fp_pkg.sv"

module tb_key_path;
  import cq_fp_pkg::*;

  localparam int MD = 128, DW = 16, G = 128;
  localparam string TVDIR = "tb/testvectors/channelquant/hex";

  reg clk = 0; always #5 clk = ~clk;
  reg rst_n;

  // ---- DUT (D=128) ----
  reg  [MD-1:0]     omask;
  reg               kiv, kgs, kgl;
  reg  [MD*DW-1:0]  kivec;
  wire              kgv, ktv;
  wire [MD*DW-1:0]  ksc_bus;
  wire [7:0]        kg_out;
  wire [6:0]        kti;
  wire [(MD/2)*8-1:0] ktpay;
  wire [MD*8-1:0]   ktcodes;
  reg  [MD*8-1:0]   kdc;
  reg  [MD*DW-1:0]  kds;
  reg  [$clog2(MD)-1:0] kdi;   // dequant channel index (one channel/beat)
  wire [31:0]       kdh;

  cq_key_path #(.D(MD), .DW(DW), .G(G)) dut (
    .clk(clk), .rst_n(rst_n), .outlier_mask(omask),
    .in_valid(kiv), .in_vec(kivec), .group_start(kgs), .group_last(kgl),
    .group_valid(kgv), .scales_bus(ksc_bus), .g_out(kg_out),
    .tok_valid(ktv), .tok_idx(kti), .tok_pay(ktpay), .tok_codes(ktcodes),
    .dec_codes(kdc), .dec_scales(kds), .dec_idx(kdi), .dec_hat(kdh)
  );

  // ---- vectors (per-channel tiers only) ----
  localparam int NVEC = 6;
  string vname [0:NVEC-1];
  int    vD[0:NVEC-1], vT[0:NVEC-1], vG[0:NVEC-1], vK[0:NVEC-1];
  initial begin
    vname[0]="d64_T128_G64__CQ4";      vD[0]=64;  vT[0]=128; vG[0]=64;  vK[0]=0;
    vname[1]="d64_T128_G64__CQ4plus";  vD[1]=64;  vT[1]=128; vG[1]=64;  vK[1]=2;
    vname[2]="d64_T70_G64__CQ4";       vD[2]=64;  vT[2]=70;  vG[2]=64;  vK[2]=0;
    vname[3]="d64_T70_G64__CQ4plus";   vD[3]=64;  vT[3]=70;  vG[3]=64;  vK[3]=2;
    vname[4]="d128_T100_G128__CQ4";    vD[4]=128; vT[4]=100; vG[4]=128; vK[4]=0;
    vname[5]="d128_T100_G128__CQ4plus";vD[5]=128; vT[5]=100; vG[5]=128; vK[5]=2;
  end

  localparam int MAXN = 128*128;
  logic [15:0] in_k  [0:MAXN-1];
  logic [15:0] g_ksc [0:MAXN-1];
  logic [7:0]  g_kpay[0:MAXN-1];
  logic [31:0] g_khat[0:MAXN-1];
  logic [7:0]  mask  [0:MD-1];
  logic [15:0] g_side[0:MAXN-1];
  int keep [0:MD-1];
  logic [7:0]  kc_arr [0:MAXN-1];   // this-group compacted keep codes [token*MD + keepidx]
  logic [7:0]  paybytes [0:MAXN-1];

  int total_fail;

  task automatic run(input int vi, output int fails);
    int D,T,GG,K,nk,c,a,b,g,t,ci,et,sc_base,pidx,pb;
    logic [15:0] sc; logic [31:0] hat; logic [7:0] by;
    begin
      D=vD[vi]; T=vT[vi]; GG=vG[vi]; K=vK[vi]; fails=0; sc_base=0; pidx=0;
      // outlier mask bus: real mask on low D, upper channels forced outlier
      omask = '0;
      for (c=0;c<MD;c=c+1) omask[c] = (c<D) ? mask[c][0] : 1'b1;
      // real keep list
      nk=0; for (c=0;c<D;c=c+1) if(!mask[c][0]) begin keep[nk]=c; nk=nk+1; end

      for (a=0; a<T; a=a+GG) begin
        b=a+GG; if (b>T) b=T; g=b-a;
        // ---- drive the group's tokens ----
        for (t=a; t<b; t=t+1) begin
          @(negedge clk);
          kivec = '0;
          for (c=0;c<D;c=c+1) kivec[c*DW +: DW] = in_k[t*D+c];
          kiv=1'b1; kgs=(t==a); kgl=(t==b-1);
        end
        @(negedge clk); kiv=1'b0; kgs=1'b0; kgl=1'b0;
        // ---- collect emitted tokens until group_valid ----
        et=0;
        while (!kgv) begin
          @(negedge clk);
          if (ktv) begin
            for (ci=0; ci<nk; ci=ci+1) kc_arr[et*MD+ci] = ktcodes[ci*8 +: 8];
            for (pb=0; pb<nk/2; pb=pb+1) paybytes[pidx++] = ktpay[pb*8 +: 8];
            et=et+1;
          end
        end
        // ---- scales (keep channels) ----
        for (ci=0; ci<nk; ci=ci+1) begin
          sc = ksc_bus[keep[ci]*DW +: DW];
          if (sc !== g_ksc[sc_base+ci]) begin fails++; if (fails<=6)
            $display("  [%s] key scale grp@%0d ch %0d: got %04h exp %04h",
                     vname[vi], a, keep[ci], sc, g_ksc[sc_base+ci]); end
        end
        // ---- K_hat per token (keep via dequant, outlier via fp16 identity) ----
        for (t=a; t<b; t=t+1) begin
          kdc='0; kds='0;
          for (c=0;c<D;c=c+1) kds[c*DW +: DW] = ksc_bus[c*DW +: DW];
          for (ci=0; ci<nk; ci=ci+1) kdc[keep[ci]*8 +: 8] = kc_arr[(t-a)*MD+ci];
          #1;
          for (ci=0; ci<nk; ci=ci+1) begin
            c = keep[ci]; kdi = c[$clog2(MD)-1:0]; #1; hat = kdh;   // index one channel
            if (hat !== g_khat[t*D+c]) begin fails++; if (fails<=6)
              $display("  [%s] k_hat (t%0d,c%0d): got %08h exp %08h",
                       vname[vi], t, c, hat, g_khat[t*D+c]); end
          end
          // outlier channels: golden k_hat must be the fp16 value widened to fp32
          for (c=0;c<D;c=c+1) if (mask[c][0])
            if (f32_to_real(g_khat[t*D+c]) != f16_to_real(in_k[t*D+c])) begin fails++;
              if (fails<=6) $display("  [%s] outlier k_hat (t%0d,c%0d) mismatch", vname[vi], t, c); end
        end
        sc_base += nk;
      end

      // ---- payload byte stream ----
      for (pb=0; pb<pidx; pb=pb+1)
        if (paybytes[pb] !== g_kpay[pb]) begin fails++; if (fails<=6)
          $display("  [%s] key payload byte %0d: got %02h exp %02h",
                   vname[vi], pb, paybytes[pb], g_kpay[pb]); end

      // ---- sidecar (CQ-4+) ----
      if (K>0)
        for (t=0;t<T;t=t+1) begin
          int oc; oc=0;
          for (c=0;c<D;c=c+1) if (mask[c][0]) begin
            if (in_k[t*D+c] !== g_side[t*K+oc]) begin fails++; if (fails<=6)
              $display("  [%s] sidecar (t%0d,c%0d): got %04h exp %04h",
                       vname[vi], t, c, in_k[t*D+c], g_side[t*K+oc]); end
            oc=oc+1;
          end
        end
    end
  endtask

  int vi, fk, nk_c, c_i, slen, plen, ga, gb;
  initial begin
    rst_n=0; kiv=0; kgs=0; kgl=0; kivec=0; kdc=0; kds=0; omask=0;
    repeat(3) @(posedge clk); rst_n=1; @(posedge clk);
    total_fail=0;

    for (vi=0; vi<NVEC; vi=vi+1) begin
      $readmemh($sformatf("%s/%s/input_k.f16.hex",       TVDIR, vname[vi]), in_k,  0, vD[vi]*vT[vi]-1);
      $readmemh($sformatf("%s/%s/expected_k_hat.f32.hex", TVDIR, vname[vi]), g_khat,0, vD[vi]*vT[vi]-1);
      $readmemh($sformatf("%s/%s/outlier_mask.u8.hex",    TVDIR, vname[vi]), mask,  0, vD[vi]-1);
      if (vK[vi]>0) $readmemh($sformatf("%s/%s/sidecar.f16.hex", TVDIR, vname[vi]), g_side, 0, vT[vi]*vK[vi]-1);
      // variable lengths: nk = D - k; groups = ceil(T/G)
      nk_c=0; for (c_i=0;c_i<vD[vi];c_i=c_i+1) if(!mask[c_i][0]) nk_c++;
      slen=0; plen=0;
      for (ga=0; ga<vT[vi]; ga=ga+vG[vi]) begin
        gb=ga+vG[vi]; if(gb>vT[vi]) gb=vT[vi];
        slen += nk_c; plen += ((gb-ga)*nk_c+1)/2;
      end
      $readmemh($sformatf("%s/%s/key_scales.f16.hex",  TVDIR, vname[vi]), g_ksc,  0, slen-1);
      $readmemh($sformatf("%s/%s/key_payload.u8.hex",  TVDIR, vname[vi]), g_kpay, 0, plen-1);

      run(vi, fk);
      $display("%-26s D=%0d T=%0d G=%0d k=%0d : K %s  (%0d)",
               vname[vi], vD[vi], vT[vi], vG[vi], vK[vi], (fk==0)?"PASS":"FAIL", fk);
      total_fail += fk;
    end

    $display("============================================================");
    if (total_fail==0) $display("KEY-PATH PARITY: ALL %0d VECTORS BIT-EXACT (scale+payload+K_hat+sidecar)", NVEC);
    else               $display("KEY-PATH PARITY: %0d TOTAL MISMATCHES", total_fail);
    $display("============================================================");
    $finish;
  end

endmodule
