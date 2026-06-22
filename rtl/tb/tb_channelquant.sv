// tb_channelquant.sv — bit-exact parity vs the ChannelQuant golden vectors.
//
// Stage 1 (this file, growing): VALUE path — per-token uniform INT4/INT8
// (contract §2). Recomputes scales + packed payload + reconstructed V_hat from
// input_v and asserts byte/bit-exact equality with the vendored golden hex
// (rtl/tb/testvectors/channelquant/hex/<vector>/...). Key path + CQ-4+ outlier
// lane are added in later stages.
//
// Run:  make -f Makefile sim_cq   (see rtl/Makefile)

`timescale 1ns/1ps
`include "cq_fp_pkg.sv"

module tb_channelquant;
  import cq_fp_pkg::*;

  localparam real EPS = 2.0 ** -14;
  localparam string TVDIR = "tb/testvectors/channelquant/hex";

  // ---- vector table -------------------------------------------------------
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

  // ---- generously sized buffers (max D*T = 128*128 = 16384) ----------------
  localparam int MAXN = 128*128;
  logic [15:0] in_v   [0:MAXN-1];      // input V, fp16
  logic [15:0] g_scal [0:MAXN-1];      // golden val_scales, fp16  (T entries)
  logic [7:0]  g_pay  [0:MAXN-1];      // golden val_payload, u8
  logic [31:0] g_vhat [0:MAXN-1];      // golden expected_v_hat, f32

  // computed
  logic [7:0]  c_pay  [0:MAXN-1];
  int          total_fail;

  // qmax/qmin per bits
  function automatic int qmax_of(input int b); qmax_of = (1<<(b-1))-1; endfunction
  function automatic int qmin_of(input int b); qmin_of = -(1<<(b-1));  endfunction

  // run a PER-TOKEN path (values for all tiers; keys for CQ-8) for one vector.
  // pfx is "v" (values) or "k" (CQ-8 per-token keys). Returns #mismatches.
  function automatic int check_pertoken(input int vi, input string pfx, input int B);
    int D, T, n, qmx, qmn, fails;
    int t, d, idx;
    real x, amax, sreal, sdbl, hat, exp_hat;
    longint q;
    logic [15:0] sbits;
    logic [7:0]  by;
    int          codes [0:MAXN-1];
    string       inf, scf, paf, htf;
    begin
      D=vD[vi]; T=vT[vi]; n=D*T;
      qmx=qmax_of(B); qmn=qmin_of(B);
      fails=0;
      inf = (pfx=="v") ? "input_v.f16.hex"       : "input_k.f16.hex";
      scf = (pfx=="v") ? "val_scales.f16.hex"    : "key_scales.f16.hex";
      paf = (pfx=="v") ? "val_payload.u8.hex"    : "key_payload.u8.hex";
      htf = (pfx=="v") ? "expected_v_hat.f32.hex": "expected_k_hat.f32.hex";

      // load golden hex for this vector
      $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], inf), in_v,   0, n-1);
      $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], scf), g_scal, 0, T-1);
      $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], htf), g_vhat, 0, n-1);
      // payload length: int8 -> n bytes ; int4 -> ceil(n/2) bytes
      if (B==8) $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], paf), g_pay, 0, n-1);
      else      $readmemh($sformatf("%s/%s/%s", TVDIR, vname[vi], paf), g_pay, 0, (n+1)/2-1);

      // per token
      for (t=0; t<T; t=t+1) begin
        // amax over D dims
        amax = 0.0;
        for (d=0; d<D; d=d+1) begin
          x = f16_to_real(in_v[t*D+d]);
          if (x < 0.0) x = -x;
          if (x > amax) amax = x;
        end
        sreal = amax / qmx;
        if (sreal < EPS) sreal = EPS;
        sbits = real_to_f16(sreal);
        if (sbits !== g_scal[t]) begin
          fails=fails+1;
          if (fails<=4) $display("  [%s] scale mismatch tok %0d: got %04h exp %04h",
                                 vname[vi], t, sbits, g_scal[t]);
        end
        sdbl = f16_to_real(sbits);
        // quant each dim
        for (d=0; d<D; d=d+1) begin
          x = f16_to_real(in_v[t*D+d]);
          q = srint_ll(x / sdbl);
          if (q > qmx) q = qmx;
          if (q < qmn) q = qmn;
          codes[t*D+d] = q;
          // dequant check
          hat     = q * sdbl;
          exp_hat = f32_to_real(g_vhat[t*D+d]);
          if (hat != exp_hat) begin
            fails=fails+1;
            if (fails<=4) $display("  [%s] vhat mismatch (%0d,%0d): got %f exp %f",
                                   vname[vi], t, d, hat, exp_hat);
          end
        end
      end

      // pack + compare payload
      if (B==8) begin
        for (idx=0; idx<n; idx=idx+1) begin
          by = codes[idx][7:0];            // two's complement int8
          if (by !== g_pay[idx]) begin
            fails=fails+1;
            if (fails<=4) $display("  [%s] int8 payload byte %0d: got %02h exp %02h",
                                   vname[vi], idx, by, g_pay[idx]);
          end
        end
      end else begin
        for (idx=0; idx<(n+1)/2; idx=idx+1) begin
          logic [3:0] lo, hi;
          lo = codes[2*idx][3:0];
          hi = (2*idx+1 < n) ? codes[2*idx+1][3:0] : 4'h0;
          by = {hi, lo};
          if (by !== g_pay[idx]) begin
            fails=fails+1;
            if (fails<=4) $display("  [%s] int4 payload byte %0d: got %02h exp %02h",
                                   vname[vi], idx, by, g_pay[idx]);
          end
        end
      end

      check_pertoken = fails;
    end
  endfunction

  // ---- per-channel grouped KEY path (CQ-4 / CQ-4+) ------------------------
  // Per contract §3 (grouped per-channel INT4) + §4 (top-k FP16 outlier lane).
  // Buffers for keys + outlier handling.
  logic [15:0] in_k    [0:MAXN-1];     // input K, fp16
  logic [15:0] gk_scal [0:MAXN-1];     // golden key_scales, fp16 (nk per group, concat)
  logic [7:0]  gk_pay  [0:MAXN-1];     // golden key_payload, u8
  logic [31:0] gk_hat  [0:MAXN-1];     // golden expected_k_hat, f32
  logic [15:0] gk_side [0:MAXN-1];     // golden sidecar, fp16 [T,k] row-major
  logic [7:0]  gk_mask [0:MAXN-1];     // golden outlier_mask, u8 (D entries, 1=outlier)

  function automatic int check_keys_perchannel(input int vi);
    int D, T, G, K, nk, fails;
    int a, b, g, c, t, ci, cc;
    int keep [0:127];                 // non-outlier channel indices
    int outl [0:7];                   // outlier channel indices (sorted)
    int sc_base;                      // running scale index across groups
    int pay_bit;                      // running nibble index across groups (per group resets)
    int nib_idx;                      // global nibble counter for packing compare
    real x, amax, sreal, sdbl, hat, exp_hat;
    longint q;
    logic [15:0] sbits;
    int   kcodes [0:MAXN-1];          // codes per (token,keep-channel) within group, flat
    int   noutl;
    begin
      D=vD[vi]; T=vT[vi]; G=vG[vi]; K=vK[vi];
      fails=0;

      // read fixed-size images + the mask first, so we can size the variable-
      // length scale/payload reads exactly (one scale set + packed block per group).
      $readmemh($sformatf("%s/%s/input_k.f16.hex",       TVDIR, vname[vi]), in_k,    0, D*T-1);
      $readmemh($sformatf("%s/%s/expected_k_hat.f32.hex",TVDIR, vname[vi]), gk_hat,  0, D*T-1);
      $readmemh($sformatf("%s/%s/outlier_mask.u8.hex",   TVDIR, vname[vi]), gk_mask, 0, D-1);
      if (K>0) $readmemh($sformatf("%s/%s/sidecar.f16.hex", TVDIR, vname[vi]), gk_side, 0, T*K-1);

      // build keep[] and outl[] from the static mask (mask=1 -> outlier)
      nk=0; noutl=0;
      for (c=0; c<D; c=c+1) begin
        if (gk_mask[c][0]) begin outl[noutl]=c; noutl=noutl+1; end
        else               begin keep[nk]=c;    nk=nk+1;       end
      end
      if (noutl !== K)
        $display("  [%s] WARN mask popcount %0d != K %0d", vname[vi], noutl, K);

      // exact lengths: scales = nGroups*nk ; payload = sum over groups ceil(g*nk/2)
      begin
        int slen, plen, ga, gb, gg;
        slen=0; plen=0;
        for (ga=0; ga<T; ga=ga+G) begin
          gb=ga+G; if (gb>T) gb=T; gg=gb-ga;
          slen = slen + nk;
          plen = plen + (gg*nk + 1)/2;
        end
        $readmemh($sformatf("%s/%s/key_scales.f16.hex", TVDIR, vname[vi]), gk_scal, 0, slen-1);
        $readmemh($sformatf("%s/%s/key_payload.u8.hex", TVDIR, vname[vi]), gk_pay,  0, plen-1);
      end

      // ---- per group: per-channel scale over g tokens, INT4 over keep chans ----
      sc_base = 0;
      nib_idx = 0;
      for (a=0; a<T; a=a+G) begin
        b = a+G; if (b>T) b=T; g = b-a;
        // per keep-channel scale + codes
        for (ci=0; ci<nk; ci=ci+1) begin
          cc = keep[ci];
          amax = 0.0;
          for (t=a; t<b; t=t+1) begin
            x = f16_to_real(in_k[t*D+cc]); if (x<0.0) x=-x;
            if (x>amax) amax=x;
          end
          sreal = amax / 7.0;            // qmax(4)=7
          if (sreal < EPS) sreal = EPS;
          sbits = real_to_f16(sreal);
          if (sbits !== gk_scal[sc_base+ci]) begin
            fails=fails+1;
            if (fails<=6) $display("  [%s] key scale grp@%0d ch %0d: got %04h exp %04h",
                                   vname[vi], a, cc, sbits, gk_scal[sc_base+ci]);
          end
          sdbl = f16_to_real(sbits);
          for (t=a; t<b; t=t+1) begin
            x = f16_to_real(in_k[t*D+cc]);
            q = srint_ll(x / sdbl);
            if (q>7)  q=7;
            if (q<-8) q=-8;
            // store in token-major,keep-channel-minor order for packing
            kcodes[(t-a)*nk + ci] = q;
            // dequant check at full (token,channel)
            hat     = q * sdbl;
            exp_hat = f32_to_real(gk_hat[t*D+cc]);
            if (hat != exp_hat) begin
              fails=fails+1;
              if (fails<=6) $display("  [%s] k_hat grp@%0d (t%0d,c%0d): got %f exp %f",
                                     vname[vi], a, t, cc, hat, exp_hat);
            end
          end
        end
        // pack this group's [g, nk] codes int4, C-order, compare against payload
        begin
          int gn, gi;
          logic [3:0] lo, hi;
          logic [7:0] by;
          gn = g*nk;
          for (gi=0; gi<(gn+1)/2; gi=gi+1) begin
            lo = kcodes[2*gi][3:0];
            hi = (2*gi+1 < gn) ? kcodes[2*gi+1][3:0] : 4'h0;
            by = {hi, lo};
            if (by !== gk_pay[nib_idx+gi]) begin
              fails=fails+1;
              if (fails<=6) $display("  [%s] key payload byte %0d (grp@%0d): got %02h exp %02h",
                                     vname[vi], nib_idx+gi, a, by, gk_pay[nib_idx+gi]);
            end
          end
          nib_idx = nib_idx + (gn+1)/2;
        end
        sc_base = sc_base + nk;
      end

      // ---- outlier sidecar (CQ-4+): fp16 = identity of input K at outlier chans -
      for (ci=0; ci<noutl; ci=ci+1) begin
        cc = outl[ci];
        for (t=0; t<T; t=t+1) begin
          // golden sidecar stored [T,k] row-major
          if (in_k[t*D+cc] !== gk_side[t*noutl+ci]) begin
            fails=fails+1;
            if (fails<=6) $display("  [%s] sidecar (t%0d,c%0d): got %04h exp %04h",
                                   vname[vi], t, cc, in_k[t*D+cc], gk_side[t*noutl+ci]);
          end
          // expected_k_hat at outlier channel == fp16 value as f32
          hat     = f16_to_real(in_k[t*D+cc]);
          exp_hat = f32_to_real(gk_hat[t*D+cc]);
          if (hat != exp_hat) begin
            fails=fails+1;
            if (fails<=6) $display("  [%s] outlier k_hat (t%0d,c%0d): got %f exp %f",
                                   vname[vi], t, cc, hat, exp_hat);
          end
        end
      end

      check_keys_perchannel = fails;
    end
  endfunction

  int vi, fv, fk;
  initial begin
    #1;
    total_fail = 0;
    for (vi=0; vi<NVEC; vi=vi+1) begin
      // value path (per-token) for every tier
      fv = check_pertoken(vi, "v", vBits[vi]);
      // key path: CQ-8 = per-token int8 keys; CQ-4/CQ-4+ = per-channel grouped
      if (vTier[vi]==0) fk = check_pertoken(vi, "k", 8);
      else              fk = check_keys_perchannel(vi);
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
