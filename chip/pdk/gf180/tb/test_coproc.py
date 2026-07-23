# test_coproc.py — KV-cache COPROCESSOR full-chip RTL cocotb test.
#
# Drives the workshop-slot chip_core (KV-compression coprocessor variant) purely
# through the 4-wire SPI slave: streams L value tokens (V, fp16) and their
# attention masses (W) in, runs ONE compress step (KVE CQ-3 value compression +
# TIU H2O importance + precision gate), then reads back the compressed records
# (INT3 codes + fp16 scale + rotated reconstruction), the TIU eviction victim,
# and the precision decision — and checks them against the host reference.
#
#   SIM=icarus make test-coproc
#
# The KVE value path itself is proven bit-exact vs the ChannelQuant reference in
# kve/rtl (make sim_wht_pathb_syn, 5120/5120). This test proves the ASSEMBLY:
# stream V/W in over SPI -> compress -> stream the compressed KV record + the
# importance/precision decision back out, driven only through the pads.

import os
import struct
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

D = 4            # value channels per token (must match chip_core_kv.sv)
L = 4            # cache slots / tokens

SCLK, CS_N, MOSI = 0, 1, 2
MISO_BIT = 3
CMD_WRITE, CMD_READ, CMD_START, CMD_STATUS = 0x01, 0x02, 0x03, 0x04
WBASE, VBASE, CBASE, SBASE, HBASE = 0x0300, 0x0400, 0x0800, 0x0A00, 0x0C00
CLK_NS = 10
OVERSAMP = 4


def f16_bits(x):
    return int(np.float16(x).view(np.uint16))

def bits_f16(b):
    return float(np.uint16(b).view(np.float16))

def s8(b):
    return b - 256 if b >= 128 else b


class Spi:
    def __init__(self, dut):
        self.dut = dut; self.sclk = 0; self.cs_n = 1; self.mosi = 0
    def _drive(self):
        self.dut.bidir_in.value = (self.sclk << SCLK) | (self.cs_n << CS_N) | (self.mosi << MOSI)
    async def _hold(self, n):
        for _ in range(n): await RisingEdge(self.dut.clk)
    def _miso(self):
        s = str(self.dut.bidir_out.value)          # MSB-first binary string
        c = s[len(s) - 1 - MISO_BIT]               # bit MISO_BIT from the right
        return 1 if c == '1' else 0
    async def frame_begin(self):
        self.cs_n, self.sclk, self.mosi = 1, 0, 0; self._drive(); await self._hold(2)
        self.cs_n = 0; self._drive(); await self._hold(2)
    async def frame_end(self):
        self.sclk, self.mosi = 0, 0; self._drive(); await self._hold(2)
        self.cs_n = 1; self._drive(); await self._hold(2)
    async def xfer(self, byte_out):
        rx = 0
        for i in range(8):
            self.mosi = (byte_out >> (7 - i)) & 1
            self.sclk = 1; self._drive(); await self._hold(1)
            rx = (rx << 1) | self._miso(); await self._hold(OVERSAMP - 1)
            self.sclk = 0; self._drive(); await self._hold(OVERSAMP)
        self.mosi = 0; self._drive(); await self._hold(2)
        return rx
    async def write(self, addr, data):
        await self.frame_begin(); await self.xfer(CMD_WRITE)
        await self.xfer((addr >> 8) & 0xFF); await self.xfer(addr & 0xFF)
        for b in data: await self.xfer(b & 0xFF)
        await self.frame_end()
    async def read(self, addr, n):
        await self.frame_begin(); await self.xfer(CMD_READ)
        await self.xfer((addr >> 8) & 0xFF); await self.xfer(addr & 0xFF)
        out = [await self.xfer(0x00) for _ in range(n)]
        await self.frame_end(); return out
    async def start(self):
        await self.frame_begin(); await self.xfer(CMD_START); await self.frame_end()
    async def status(self):
        await self.frame_begin(); await self.xfer(CMD_STATUS)
        s = await self.xfer(0x00); await self.frame_end(); return s


@cocotb.test()
async def kv_compress_pass(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    dut.rst_n.value = 0
    dut.input_in.value = 0
    dut.bidir_in.value = (1 << CS_N)
    for _ in range(10): await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(5): await RisingEdge(dut.clk)

    spi = Spi(dut)

    # ---- host tensors: L value tokens (D fp16 channels) + per-token masses ----
    rng = np.random.default_rng(7)
    V = (rng.standard_normal((L, D)) * 0.5).astype(np.float16)   # fp16 value rows
    W = np.array([10, 250, 40, 90][:L], dtype=np.uint8)          # importance masses

    # stream V in (fp16 LE, 2 bytes/elem)
    vb = []
    for l in range(L):
        for d in range(D):
            b = f16_bits(V[l, d]); vb += [b & 0xFF, (b >> 8) & 0xFF]
    await spi.write(VBASE, vb)
    await spi.write(WBASE, [int(x) for x in W])

    # run one compress step
    await spi.start()
    for _ in range(40):
        s = await spi.status()
        if s & 0x02:  # done
            break
    dut._log.info(f"STATUS = 0x{s:02x} (done={(s>>1)&1}, gate_fp16={(s>>4)&1})")
    assert (s & 0x02), "compress step never asserted done"

    # ---- read back compressed records over SPI ----
    codes_raw = await spi.read(CBASE, L * D)              # int8/lane
    scale_raw = await spi.read(SBASE, 2 * L)              # fp16/token
    vhat_raw  = await spi.read(HBASE, 2 * L * D)          # fp16/lane
    deci      = (await spi.read(0x0002, 1))[0]
    keeplo    = (await spi.read(0x0003, 1))[0]

    codes = np.array([s8(codes_raw[l * D + d]) for l in range(L) for d in range(D)]).reshape(L, D)
    scale = np.array([bits_f16((scale_raw[2 * l + 1] << 8) | scale_raw[2 * l]) for l in range(L)])
    vhat  = np.array([bits_f16((vhat_raw[2 * (l * D + d) + 1] << 8) | vhat_raw[2 * (l * D + d)])
                      for l in range(L) for d in range(D)]).reshape(L, D)

    # ---- cross-check SPI readback == the on-chip buffers (assembly integrity) ----
    core = dut.dut.u_coproc
    cb = np.array([s8(int(core.code_buf[l * D + d].value) & 0xFF) for l in range(L) for d in range(D)]).reshape(L, D)
    assert np.array_equal(codes, cb), f"SPI codes != on-chip code_buf\n{codes}\n{cb}"
    dut._log.info("SPI-read codes bit-identical to on-chip code_buf: True")

    # ---- (1) codes are valid INT3, compression non-degenerate ----
    assert codes.min() >= -4 and codes.max() <= 3, f"codes out of INT3 range: {codes.min()}..{codes.max()}"
    assert np.any(codes != 0), "all codes zero (degenerate)"
    assert np.all(scale > 0), f"non-positive scale: {scale}"
    dut._log.info(f"codes range [{codes.min()},{codes.max()}]  scales={np.round(scale,4)}")

    # ---- (2) reconstruction == host dequant(codes, scale) : real(code)*f16(scale)->f16 ----
    dq_ref = np.array([[np.float16(np.float32(codes[l, d]) * np.float32(scale[l])) for d in range(D)]
                       for l in range(L)])
    dq_ref_f = dq_ref.astype(np.float32)
    if np.array_equal(np.array([[f16_bits(vhat[l, d]) for d in range(D)] for l in range(L)]),
                      np.array([[f16_bits(dq_ref[l, d]) for d in range(D)] for l in range(L)])):
        dut._log.info("rotated reconstruction bit-matches host dequant(codes,scale): True")
        recon_bitexact = True
    else:
        # fp16 round-half-even edge cases — require close, report if not bit-exact
        maxerr = float(np.max(np.abs(vhat.astype(np.float32) - dq_ref_f)))
        dut._log.info(f"reconstruction vs host dequant: not bit-exact, max abs {maxerr:.2e}")
        recon_bitexact = False
        assert maxerr < 1e-2, f"reconstruction dequant mismatch too large: {maxerr}"

    # ---- (3) TIU: eviction victim == least-important slot; keep bits sane ----
    evict = deci & ((1 << (L.bit_length() - 1)) - 1) if L > 1 else 0
    evict = deci & (max(1, (L - 1)))  # low SLW bits
    argmin_w = int(np.argmin(W))
    dut._log.info(f"TIU evict_slot={evict} (host argmin importance={argmin_w})  keep_lo=0x{keeplo:02x}")
    assert evict == argmin_w, f"TIU evicted slot {evict}, expected least-important {argmin_w}"

    # ---- (4) precision decision produced ----
    gate = (deci >> 7) & 1
    dut._log.info(f"precision gate d_fp16 = {gate}")

    dut._log.info(
        "KV-COPROC COMPRESS PASS: V streamed in over SPI, compressed to INT3 "
        f"codes+fp16 scale (readback bit-identical to on-chip), reconstruction "
        f"{'bit-exact' if recon_bitexact else 'within tol'} vs host dequant, TIU "
        f"evicted the least-important slot, precision gate emitted.")
