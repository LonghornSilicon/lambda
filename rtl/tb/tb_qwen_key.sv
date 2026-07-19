// tb_qwen_key.sv — KEY-path RTL vs real Qwen keys (bit-exact K̂ reconstruction).
//
// Companion to tb_qwen_multi (value path). Streams real Qwen2 key slices (dumped by
// channelquant_hw.py --mode dump-multi: key_i.hex / keymask_i.hex / keyhat_i.hex)
// through cq_key_path and checks the reconstructed K̂ bit-for-bit against the fp16-exact
// codec (keep channels via the DUT's per-channel INT4 dequant, the k=2 outlier channels
// via the FP16 identity lane). Reuses the K̂-check logic proven in tb_key_path; it drops
// the scale/payload/sidecar golden comparisons (those goldens are C++-reference-only).
// One D=128 DUT covers both head dims (D=64 slices mask the upper 64 channels as outliers).
//   iverilog ... ; vvp a.out +TVDIR=tb/testvectors/qwen/g15b/multi   (or g05b)
`timescale 1ns/1ps
`include "cq_fp_pkg.sv"

module tb_qwen_key;
  import cq_fp_pkg::*;
  localparam int MD = 128, DW = 16, G = 128;

  reg clk = 0; always #5 clk = ~clk;
  reg rst_n;

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
  reg  [$clog2(MD)-1:0] kdi;
  wire [31:0]       kdh;

  cq_key_path #(.D(MD), .DW(DW), .G(G)) dut (
    .clk(clk), .rst_n(rst_n), .outlier_mask(omask),
    .in_valid(kiv), .in_vec(kivec), .group_start(kgs), .group_last(kgl),
    .group_valid(kgv), .scales_bus(ksc_bus), .g_out(kg_out),
    .tok_valid(ktv), .tok_idx(kti), .tok_pay(ktpay), .tok_codes(ktcodes),
    .dec_codes(kdc), .dec_scales(kds), .dec_idx(kdi), .dec_hat(kdh)
  );

  localparam int MAXN = 128*128;
  logic [15:0] in_k  [0:MAXN-1];
  logic [31:0] g_khat[0:MAXN-1];
  logic [7:0]  mask  [0:MD-1];
  int   keep [0:MD-1];
  logic [7:0] kc_arr [0:MAXN-1];

  integer checks, pass;

  // Reconstruct-check one slice (single group since T<=G): returns via globals checks/pass.
  task automatic run_slice(input int D, input int T);
    int nk,c,t,ci,et;
    logic [31:0] hat;
    begin
      omask = '0;
      for (c=0;c<MD;c=c+1) omask[c] = (c<D) ? mask[c][0] : 1'b1;
      nk=0; for (c=0;c<D;c=c+1) if(!mask[c][0]) begin keep[nk]=c; nk=nk+1; end
      // drive the single group
      for (t=0; t<T; t=t+1) begin
        @(negedge clk);
        kivec = '0;
        for (c=0;c<D;c=c+1) kivec[c*DW +: DW] = in_k[t*D+c];
        kiv=1'b1; kgs=(t==0); kgl=(t==T-1);
      end
      @(negedge clk); kiv=1'b0; kgs=1'b0; kgl=1'b0;
      // collect emitted tokens' compacted keep codes
      et=0;
      while (!kgv) begin
        @(negedge clk);
        if (ktv) begin
          for (ci=0; ci<nk; ci=ci+1) kc_arr[et*MD+ci] = ktcodes[ci*8 +: 8];
          et=et+1;
        end
      end
      // K_hat per token: keep via DUT dequant (bit-exact), outlier via fp16 identity
      for (t=0; t<T; t=t+1) begin
        kdc='0; kds='0;
        for (c=0;c<D;c=c+1) kds[c*DW +: DW] = ksc_bus[c*DW +: DW];
        for (ci=0; ci<nk; ci=ci+1) kdc[keep[ci]*8 +: 8] = kc_arr[t*MD+ci];
        #1;
        for (ci=0; ci<nk; ci=ci+1) begin
          c = keep[ci]; kdi = c[$clog2(MD)-1:0]; #1; hat = kdh;
          checks = checks + 1;
          if (hat === g_khat[t*D+c]) pass = pass + 1;
          else if (checks - pass <= 5)
            $display("  KEEP mismatch (t%0d,c%0d): got %08h exp %08h", t, c, hat, g_khat[t*D+c]);
        end
        for (c=0;c<D;c=c+1) if (mask[c][0]) begin
          checks = checks + 1;
          if (f32_to_real(g_khat[t*D+c]) == f16_to_real(in_k[t*D+c])) pass = pass + 1;
          else if (checks - pass <= 5) $display("  OUTLIER mismatch (t%0d,c%0d)", t, c);
        end
      end
    end
  endtask

  string TVDIR;
  integer fman, fk, code, nsl, si, Dn, Tn, Ln, Hn, idx, t, c;
  logic [15:0] tmp16; logic [31:0] tmp32; logic [7:0] tmp8;
  int slices;

  initial begin
    rst_n=0; kiv=0; kgs=0; kgl=0; kivec=0; kdc=0; kds=0; omask=0;
    if (!$value$plusargs("TVDIR=%s", TVDIR)) TVDIR = "tb/testvectors/qwen/g15b/multi";
    repeat(3) @(posedge clk); rst_n=1; @(posedge clk);
    checks=0; pass=0; slices=0;

    fman = $fopen($sformatf("%s/manifest.txt", TVDIR), "r");
    if (fman==0) begin $display("ERROR: no manifest at %s", TVDIR); $finish; end
    code = $fscanf(fman, "%d\n", nsl);
    $display("KEY path — manifest: %0d slices from %s", nsl, TVDIR);

    for (si=0; si<nsl; si=si+1) begin
      code = $fscanf(fman, "%d %d %d %d %d\n", idx, Dn, Tn, Ln, Hn);
      fk = $fopen($sformatf("%s/key_%0d.hex", TVDIR, idx), "r");
      code = $fscanf(fk, "%d %d\n", Dn, Tn);
      for (t=0;t<Tn;t=t+1) for (c=0;c<Dn;c=c+1) begin code=$fscanf(fk,"%h",tmp16); in_k[t*Dn+c]=tmp16; end
      $fclose(fk);
      fk = $fopen($sformatf("%s/keyhat_%0d.hex", TVDIR, idx), "r");
      for (t=0;t<Tn;t=t+1) for (c=0;c<Dn;c=c+1) begin code=$fscanf(fk,"%h",tmp32); g_khat[t*Dn+c]=tmp32; end
      $fclose(fk);
      fk = $fopen($sformatf("%s/keymask_%0d.hex", TVDIR, idx), "r");
      for (c=0;c<Dn;c=c+1) begin code=$fscanf(fk,"%h",tmp8); mask[c]=tmp8; end
      $fclose(fk);
      run_slice(Dn, Tn);
      slices = slices + 1;
    end
    $fclose(fman);

    $display("");
    $display("Real-Qwen KEY-path grid: %0d/%0d elements bit-exact across %0d slices",
             pass, checks, slices);
    if (checks>0 && pass==checks) $display("ALL TESTS PASSED");
    else $display("FAILED (%0d/%0d)", pass, checks);
    $finish;
  end
endmodule
