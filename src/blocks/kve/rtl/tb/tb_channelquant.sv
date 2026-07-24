// tb_channelquant.sv — bit-exact parity vs the ChannelQuant golden vectors,
// driving the actual datapath modules in cq_units.sv.
//
// Contract §8 acceptance: for all 9 vendored golden vectors (CQ-8/CQ-4/CQ-4+,
// D in {64,128}, full g=G and partial g<G key groups) the recomputed
//   - per-axis fp16 scales (cq_scale_unit)
//   - signed integer codes      (cq_quant_unit)
//   - packed byte stream        (cq_pack2 for int4, raw byte for int8)
//   - reconstructed K/V_hat f32  (cq_dequant_unit)
// must equal the golden hex byte-for-byte. The TB owns only the amax reduction,
// grouping/flush control, and the static outlier mask; all fp arithmetic flows
// through the cq_units modules.
//
// Run:  make sim_cq   (rtl/Makefile)

`timescale 1ns/1ps
`include "cq_fp_pkg.sv"

module tb_channelquant;
  import cq_fp_pkg::*;

  localparam real   EPS   = 2.0 ** -14;
  localparam string TVDIR = "tb/testvectors/channelquant/hex";

  // ===== DUT instances (combinational compute cores) =======================
  reg  [15:0]      u_amax;     reg [3:0] u_sbits_bits;   wire [15:0] u_scale;
  cq_scale_unit   u_scale_i (.amax_f16(u_amax), .bits(u_sbits_bits), .scale_f16(u_scale));

  reg  [15:0]      u_x;        reg [15:0] u_qscale;  reg [3:0] u_qbits; wire signed [7:0] u_code;
  cq_quant_unit   u_quant_i (.x_f16(u_x), .scale_f16(u_qscale), .bits(u_qbits), .code(u_code));

  reg  signed [7:0] u_dcode;   reg [15:0] u_dscale;  wire [31:0] u_xhat;
  cq_dequant_unit u_dequant_i (.code(u_dcode), .scale_f16(u_dscale), .xhat_f32(u_xhat));

  reg  signed [7:0] u_plo, u_phi;  wire [7:0] u_pbyte;
  cq_pack2        u_pack_i (.c_lo(u_plo), .c_hi(u_phi), .byte_out(u_pbyte));

  // settle helper: drive then let `always @*` propagate across instance boundary
  localparam SETTLE = 1;

  // ===== vector table ======================================================
  localparam int NVEC = 9;
  string  vname [0:NVEC-1];
  int     vD    [0:NVEC-1];
  int     vT    [0:NVEC-1];
  int     vBits [0:NVEC-1];   // value path bits: 8 for CQ-8, else 4
  int     vG    [0:NVEC-1];   // key group size (0 => per-token keys, CQ-8)
  int     vTier [0:NVEC-1];   // 0=CQ-8, 1=CQ-4, 2=CQ-4+
  int     vK    [0:NVEC-1];   // # outlier channels (CQ-4+); else 0

  initial begin
    vname[0]="d64_T128_G64__CQ8";     vD[0]=64;  vT[0]=128; vBits[0]=8; vG[0]=0;   vTier[0]=0; vK[0]=0;
    vname[1]="d64_T128_G64__CQ4";     vD[1]=64;  vT[1]=128; vBits[1]=4; vG[1]=64;  vTier[1]=1; vK[1]=0;
    vname[2]="d64_T128_G64__CQ4plus"; vD[2]=64;  vT[2]=128; vBits[2]=4; vG[2]=64;  vTier[2]=2; vK[2]=2;
    vname[3]="d64_T70_G64__CQ8";      vD[3]=64;  vT[3]=70;  vBits[3]=8; vG[3]=0;   vTier[3]=0; vK[3]=0;
    vname[4]="d64_T70_G64__CQ4";      vD[4]=64;  vT[4]=70;  vBits[4]=4; vG[4]=64;  vTier[4]=1; vK[4]=0;
    vname[5]="d64_T70_G64__CQ4plus";  vD[5]=64;  vT[5]=70;  vBits[5]=4; vG[5]=64;  vTier[5]=2; vK[5]=2;
    vname[6]="d128_T100_G128__CQ8";   vD[6]=128; vT[6]=100; vBits[6]=8; vG[6]=0;   vTier[6]=0; vK[6]=0;
    vname[7]="d128_T100_G128__CQ4";   vD[7]=128; vT[7]=100; vBits[7]=4; vG[7]=128; vTier[7]=1; vK[7]=0;
    vname[8]="d128_T100_G128__CQ4plus";vD[8]=128;vT[8]=100; vBits[8]=4; vG[8]=128; vTier[8]=2; vK[8]=2;
  end

  // ===== buffers (max D*T = 128*128) =======================================
  localparam int MAXN = 128*128;
  logic [15:0] in_v   [0:MAXN-1];
  logic [15:0] g_scal [0:MAXN-1];
  logic [7:0]  g_pay  [0:MAXN-1];
  logic [31:0] g_vhat [0:MAXN-1];
  int          codes  [0:MAXN-1];

  logic [15:0] in_k    [0:MAXN-1];
  logic [15:0] gk_scal [0:MAXN-1];
  logic [7:0]  gk_pay  [0:MAXN-1];
  logic [31:0] gk_hat  [0:MAXN-1];
  logic [15:0] gk_side [0:MAXN-1];
  logic [7:0]  gk_mask [0:MAXN-1];

  int total_fail;

  // ---- amax helper: largest |fp16| over a strided run, as fp16 bits ---------
  // returns the magnitude (sign-cleared) bits of the max-abs element.
  function automatic logic [15:0] amax_bits_run(input int base, input int stride,
                                                input int count, input mem_sel_k);
    int i, idx;
    real best, m;
    logic [15:0] bestbits, w;
    begin
      best = -1.0; bestbits = 16'h0;
      for (i=0; i<count; i=i+1) begin
        idx = base + i*stride;
        w   = mem_sel_k ? in_k[idx] : in_v[idx];
        m   = f16_to_real({1'b0, w[14:0]});       // |value|
        if (m > best) begin best = m; bestbits = {1'b0, w[14:0]}; end
      end
      amax_bits_run = bestbits;
    end
  endfunction

  // ===== PER-TOKEN path (values all tiers; CQ-8 keys) ======================
  task automatic check_pertoken(input int vi, input string pfx, input int B, output int fails);
    int D, T, n, qmx, qmn, t, d, idx;
    real hat, exp_hat;
    logic [15:0] sbits;
    logic [7:0]  by;
    string inf, scf, paf, htf;
    begin
      D=vD[vi]; T=vT[vi]; n=D*T; qmx=(1<<(B-1))-1; qmn=-(1<<(B-1)); fails=0;
      inf = (pfx=="v") ? "input_v.f16.hex"       : "input_k.f16.hex";
      scf = (pfx=="v") ? "val_scales.f16.hex"    : "key_scales.f16.hex";
      paf = (pfx=="v") ? "val_payload.u8.hex"    : "key_payload.u8.hex";
      htf = (pfx=="v") ? "expected_v_hat.f32.hex": "expected_k_hat.f32.hex";

      $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], inf), in_v,  0, n-1);
      $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], scf), g_scal, 0, T-1);
      $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], htf), g_vhat, 0, n-1);
      if (B==8) $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], paf), g_pay, 0, n-1);
      else      $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], paf), g_pay, 0, (n+1)/2-1);

      for (t=0; t<T; t=t+1) begin
        // scale via cq_scale_unit, fed by the amax reduction
        u_amax = amax_bits_run(t*D, 1, D, 1'b0);  // per-token data always in in_v
        u_sbits_bits = B[3:0];
        #SETTLE; sbits = u_scale;
        if (sbits !== g_scal[t]) begin
          fails=fails+1;
          if (fails<=4) $display("  [%s] %s scale tok %0d: got %04h exp %04h",
                                 vname[vi], pfx, t, sbits, g_scal[t]);
        end
        // quant + dequant each dim through the units
        for (d=0; d<D; d=d+1) begin
          u_x=in_v[t*D+d]; u_qscale=sbits; u_qbits=B[3:0]; #SETTLE;
          codes[t*D+d] = u_code;
          u_dcode=u_code; u_dscale=sbits; #SETTLE;
          if (u_xhat !== g_vhat[t*D+d]) begin
            fails=fails+1;
            if (fails<=4) $display("  [%s] %s_hat (%0d,%0d): got %08h exp %08h",
                                   vname[vi], pfx, t, d, u_xhat, g_vhat[t*D+d]);
          end
        end
      end

      // pack + compare payload
      if (B==8) begin
        for (idx=0; idx<n; idx=idx+1) begin
          by = codes[idx][7:0];
          if (by !== g_pay[idx]) begin
            fails=fails+1;
            if (fails<=4) $display("  [%s] %s int8 byte %0d: got %02h exp %02h",
                                   vname[vi], pfx, idx, by, g_pay[idx]);
          end
        end
      end else begin
        for (idx=0; idx<(n+1)/2; idx=idx+1) begin
          u_plo = codes[2*idx];
          u_phi = (2*idx+1 < n) ? codes[2*idx+1] : 8'sh0;
          #SETTLE;
          if (u_pbyte !== g_pay[idx]) begin
            fails=fails+1;
            if (fails<=4) $display("  [%s] %s int4 byte %0d: got %02h exp %02h",
                                   vname[vi], pfx, idx, u_pbyte, g_pay[idx]);
          end
        end
      end
    end
  endtask

  // ===== PER-CHANNEL grouped KEY path (CQ-4 / CQ-4+) =======================
  task automatic check_keys_perchannel(input int vi, output int fails);
    int D, T, G, K, nk, noutl;
    int a, b, g, c, t, ci, cc, sc_base, nib_idx, gn, gi;
    int keep [0:127];
    int outl [0:7];
    int kcodes [0:MAXN-1];
    real hat, exp_hat;
    logic [15:0] sbits;
    begin
      D=vD[vi]; T=vT[vi]; G=vG[vi]; K=vK[vi]; fails=0;

      $readmemh($sformatf("%s/%s/input_k.f16.hex",       TVDIR, vname[vi]), in_k,   0, D*T-1);
      $readmemh($sformatf("%s/%s/expected_k_hat.f32.hex",TVDIR, vname[vi]), gk_hat, 0, D*T-1);
      $readmemh($sformatf("%s/%s/outlier_mask.u8.hex",   TVDIR, vname[vi]), gk_mask,0, D-1);
      if (K>0) $readmemh($sformatf("%s/%s/sidecar.f16.hex", TVDIR, vname[vi]), gk_side, 0, T*K-1);

      // keep[]/outl[] from static mask (mask=1 -> outlier)
      nk=0; noutl=0;
      for (c=0; c<D; c=c+1) begin
        if (gk_mask[c][0]) begin outl[noutl]=c; noutl=noutl+1; end
        else               begin keep[nk]=c;    nk=nk+1;       end
      end
      if (noutl !== K) $display("  [%s] WARN mask popcount %0d != K %0d", vname[vi], noutl, K);

      // exact lengths for the variable-length reads
      begin
        int slen, plen, ga, gb, gg;
        slen=0; plen=0;
        for (ga=0; ga<T; ga=ga+G) begin
          gb=ga+G; if (gb>T) gb=T; gg=gb-ga;
          slen=slen+nk; plen=plen+(gg*nk+1)/2;
        end
        $readmemh($sformatf("%s/%s/key_scales.f16.hex", TVDIR, vname[vi]), gk_scal, 0, slen-1);
        $readmemh($sformatf("%s/%s/key_payload.u8.hex", TVDIR, vname[vi]), gk_pay,  0, plen-1);
      end

      sc_base=0; nib_idx=0;
      for (a=0; a<T; a=a+G) begin
        b=a+G; if (b>T) b=T; g=b-a;
        for (ci=0; ci<nk; ci=ci+1) begin
          cc = keep[ci];
          // per-channel amax over the g tokens of this block (stride D), via unit
          u_amax = amax_bits_run(a*D+cc, D, g, 1'b1); u_sbits_bits = 4'd4;
          #SETTLE; sbits = u_scale;
          if (sbits !== gk_scal[sc_base+ci]) begin
            fails=fails+1;
            if (fails<=6) $display("  [%s] key scale grp@%0d ch %0d: got %04h exp %04h",
                                   vname[vi], a, cc, sbits, gk_scal[sc_base+ci]);
          end
          for (t=a; t<b; t=t+1) begin
            u_x=in_k[t*D+cc]; u_qscale=sbits; u_qbits=4'd4; #SETTLE;
            kcodes[(t-a)*nk + ci] = u_code;
            u_dcode=u_code; u_dscale=sbits; #SETTLE;
            if (u_xhat !== gk_hat[t*D+cc]) begin
              fails=fails+1;
              if (fails<=6) $display("  [%s] k_hat grp@%0d (t%0d,c%0d): got %08h exp %08h",
                                     vname[vi], a, t, cc, u_xhat, gk_hat[t*D+cc]);
            end
          end
        end
        // pack this group [g,nk] int4, C-order
        gn = g*nk;
        for (gi=0; gi<(gn+1)/2; gi=gi+1) begin
          u_plo = kcodes[2*gi];
          u_phi = (2*gi+1 < gn) ? kcodes[2*gi+1] : 8'sh0;
          #SETTLE;
          if (u_pbyte !== gk_pay[nib_idx+gi]) begin
            fails=fails+1;
            if (fails<=6) $display("  [%s] key payload byte %0d (grp@%0d): got %02h exp %02h",
                                   vname[vi], nib_idx+gi, a, u_pbyte, gk_pay[nib_idx+gi]);
          end
        end
        nib_idx = nib_idx + (gn+1)/2;
        sc_base = sc_base + nk;
      end

      // outlier sidecar (CQ-4+): fp16 identity of input K at outlier channels
      for (ci=0; ci<noutl; ci=ci+1) begin
        cc = outl[ci];
        for (t=0; t<T; t=t+1) begin
          if (in_k[t*D+cc] !== gk_side[t*noutl+ci]) begin
            fails=fails+1;
            if (fails<=6) $display("  [%s] sidecar (t%0d,c%0d): got %04h exp %04h",
                                   vname[vi], t, cc, in_k[t*D+cc], gk_side[t*noutl+ci]);
          end
          // expected_k_hat at outlier channel == fp16 value widened to f32
          if (f32_to_real(gk_hat[t*D+cc]) != f16_to_real(in_k[t*D+cc])) begin
            fails=fails+1;
            if (fails<=6) $display("  [%s] outlier k_hat (t%0d,c%0d) mismatch", vname[vi], t, cc);
          end
        end
      end
    end
  endtask

  int vi, fv, fk;
  initial begin
    #1;
    total_fail = 0;
    for (vi=0; vi<NVEC; vi=vi+1) begin
      check_pertoken(vi, "v", vBits[vi], fv);
      if (vTier[vi]==0) check_pertoken(vi, "k", 8, fk);
      else              check_keys_perchannel(vi, fk);
      $display("%-26s D=%0d T=%0d G=%0d tier=%0d : V %s / K %s  (%0d+%0d mism)",
               vname[vi], vD[vi], vT[vi], vG[vi], vTier[vi],
               (fv==0)?"PASS":"FAIL", (fk==0)?"PASS":"FAIL", fv, fk);
      total_fail = total_fail + fv + fk;
    end
    $display("============================================================");
    if (total_fail==0) $display("CHANNELQUANT PARITY: ALL %0d VECTORS BIT-EXACT (V+K, all tiers)", NVEC);
    else               $display("CHANNELQUANT PARITY: %0d TOTAL MISMATCHES", total_fail);
    $display("============================================================");
    $finish;
  end

endmodule
