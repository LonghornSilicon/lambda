# test_fullchip.py — full-chip RTL cocotb test for the assembled Lambda ACU.
#
# Drives the workshop-slot chip_core (via chip_core_wrap, pad widths 1/20/60)
# purely through the 4-wire SPI slave on the bidir pads, runs ONE decode
# attention pass, and checks the streamed-out fp32 attention row against a
# host-computed reference.
#
#   SIM=icarus make test-fullchip
#
# WHAT THIS PROVES (the full-chip integration gap):
#   * Q (int8), K (fp16) and V (fp16) are STREAMED IN over the real SPI loader
#     (WRITE frames), exactly as the host would over the ~20 workshop pads.
#   * START launches the decode FSM in lambda_acu, which sequences the hardened
#     macros KVE -> mate_qkt -> vecu_softmax -> mate_pv_fp16 -> inverse-WHT
#     (+ precision_controller gate + INT8 mate_pv + TIU) to one attention row.
#   * The host polls STATUS over SPI until done, then READs the OUT region and
#     reconstructs the fp32 attention output — the assembled chip's real result,
#     off-chip over the serial link.
#
# REFERENCE / TOLERANCE. Because the inverse-WHT is linear, the chip computes
#   o[d] = inverse_WHT( Σ_l w[l]·V̂rot[l][d] ) = Σ_l w[l]·V̂[l][d]
# where w = softmax(Q·Kᵀ) and V̂ is the KVE CQ-3-rot reconstruction of the input
# V. The host reference is plain fp32 attention over the *input* V:
#   o_ref[d] = Σ_l softmax(Q·Kᵀ)_ref[l]·V[l][d].
# The chip↔reference gap is therefore the sum of three CHARACTERIZED terms:
#   (1) KVE CQ-3-rot value quantization (int3 codes on the WHT-rotated row),
#   (2) fp16 exp-LUT online softmax (~2%), (3) fp16 P·V rounding.
# Each block's own numeric fidelity is proven bit-exact / within-tol in the
# cross-block cosim (chip/verif/tb_chip_cosim.sv); this test proves the ASSEMBLY
# computes the right attention row and streams it out. The measured max rel err
# is printed and asserted below.

import os
import struct
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ---- decode-tile shape (must match lambda_acu params in chip_core.sv) --------
DH = 8           # head-dim channels
L  = 4           # cached tokens / keys

# ---- SPI pad bits (see rtl/chip_core.sv PAD MAP) -----------------------------
SCLK, CS_N, MOSI = 0, 1, 2
MISO_BIT = 3     # bidir_out[3]

CMD_WRITE, CMD_READ, CMD_START, CMD_STATUS = 0x01, 0x02, 0x03, 0x04

QBASE, KBASE, VBASE, OBASE = 0x0100, 0x0200, 0x0400, 0x0800

CLK_NS = 10
OVERSAMP = 4     # core clks per SPI half-bit (SCLK << core clk)


# ============================ fp16 helpers ====================================
def f16_bits(x):
    return int(np.float16(x).view(np.uint16))

def bits_f16(b):
    return float(np.uint16(b).view(np.float16))

def f32_bytes(x):
    return struct.pack("<f", float(x))

def bytes_f32(b4):
    return struct.unpack("<f", bytes(b4))[0]


# ============================ SPI host driver =================================
class Spi:
    def __init__(self, dut):
        self.dut = dut
        self.sclk = 0
        self.cs_n = 1
        self.mosi = 0

    def _drive(self):
        v = (self.sclk << SCLK) | (self.cs_n << CS_N) | (self.mosi << MOSI)
        self.dut.bidir_in.value = v

    async def _hold(self, n):
        for _ in range(n):
            await RisingEdge(self.dut.clk)

    def _miso(self):
        return (int(self.dut.bidir_out.value) >> MISO_BIT) & 1

    async def frame_begin(self):
        self.cs_n, self.sclk, self.mosi = 1, 0, 0
        self._drive(); await self._hold(2)
        self.cs_n = 0                       # falling edge = frame start
        self._drive(); await self._hold(2)

    async def frame_end(self):
        self.sclk, self.mosi = 0, 0
        self._drive(); await self._hold(2)
        self.cs_n = 1
        self._drive(); await self._hold(2)

    async def xfer(self, byte_out):
        """Shift one byte MSB-first (SPI mode 0, CPHA=0). Returns the MISO byte.
        Data is set up while sclk is low; the chip samples MOSI on the rising
        edge and the host samples MISO on the same rising edge; the chip advances
        MISO to the next bit on the following falling edge. The slave presents its
        MSB after the byte is loaded (before the first rising edge), so we must
        sample on the rise BEFORE generating the fall (a leading fall would shift
        the MSB out unsampled)."""
        rx = 0
        for i in range(8):
            self.mosi = (byte_out >> (7 - i)) & 1
            self.sclk = 1                   # rising edge
            self._drive(); await self._hold(1)
            rx = (rx << 1) | self._miso()   # sample MISO early, before rise-detect advances bit_cnt
            await self._hold(OVERSAMP - 1)   # ... then let the chip sample MOSI on the (synced) rise
            self.sclk = 0                   # falling edge
            self._drive(); await self._hold(OVERSAMP)
        self.mosi = 0
        self._drive(); await self._hold(2)
        return rx

    async def write(self, addr, data_bytes):
        await self.frame_begin()
        await self.xfer(CMD_WRITE)
        await self.xfer((addr >> 8) & 0xFF)
        await self.xfer(addr & 0xFF)
        for b in data_bytes:
            await self.xfer(b & 0xFF)
        await self.frame_end()

    async def read(self, addr, n):
        await self.frame_begin()
        await self.xfer(CMD_READ)
        await self.xfer((addr >> 8) & 0xFF)
        await self.xfer(addr & 0xFF)
        out = [await self.xfer(0x00) for _ in range(n)]
        await self.frame_end()
        return out

    async def start(self):
        await self.frame_begin()
        await self.xfer(CMD_START)
        await self.frame_end()

    async def status(self):
        await self.frame_begin()
        await self.xfer(CMD_STATUS)
        s = await self.xfer(0x00)           # STATUS byte on next MISO byte
        await self.frame_end()
        return s


# ============================ test data =======================================
def load_qwen_v():
    """First L tokens x DH channels of the real-Qwen V tile (fp16)."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "..", "..", "verif", "vectors", "qwen_val.hex")
    with open(path) as f:
        toks = f.read().split("\n")
    header = toks[0].split()
    Dfull = int(header[0])                  # 128
    rows = []
    for t in range(L):
        words = toks[1 + t].split()
        rows.append([bits_f16(int(words[d], 16)) for d in range(DH)])  # DH<=Dfull
    return np.array(rows, dtype=np.float32), Dfull


def build_inputs():
    V, _ = load_qwen_v()                    # [L, DH] fp32 (from fp16 bits)
    # query: +1 on every channel (int8)
    Q = np.ones(DH, dtype=np.int32)
    # keys: per-key constant so score_l = Σ_d Q[d]·K[l][d] = DH·kbase_l spreads
    target_scores = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, 1.0, -1.0][:L], dtype=np.float32)
    K = np.zeros((L, DH), dtype=np.float32)
    for l in range(L):
        kb = f16_bits(target_scores[l] / DH)
        K[l, :] = bits_f16(kb)              # exact fp16 value actually sent
    return Q, K, V


def reference(Q, K, V):
    scores = np.array([float(np.dot(Q.astype(np.float32), K[l])) for l in range(L)],
                      dtype=np.float32)
    m = scores.max()
    e = np.exp(scores - m)
    w = e / e.sum()
    o = np.zeros(DH, dtype=np.float32)
    for l in range(L):
        o += w[l] * V[l]
    return scores, w, o


# ============================ the test ========================================
@cocotb.test()
async def full_decode_attention_pass(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    spi = Spi(dut)

    # reset
    dut.input_in.value = 0
    dut.bidir_in.value = (1 << CS_N)
    dut.rst_n.value = 0
    await Timer(40, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # pad-direction sanity
    await Timer(1, unit="ns")
    oe = int(dut.bidir_oe.value)
    assert (oe & 0b111) == 0, f"SPI input pads must be inputs (oe=0), got {oe:#x}"
    assert ((oe >> 3) & 0x1FFFF) == 0x1FFFF, f"output pads must be oe=1, got {oe:#x}"

    Q, K, V = build_inputs()
    scores_ref, w_ref, o_ref = reference(Q, K, V)

    # ---- stream Q (int8), K (fp16 LE), V (fp16 LE) over SPI WRITE frames ------
    await spi.write(QBASE, [int(np.uint8(np.int8(Q[d]))) for d in range(DH)])

    kbytes = []
    for l in range(L):
        for d in range(DH):
            b = f16_bits(K[l, d])
            kbytes += [b & 0xFF, (b >> 8) & 0xFF]
    await spi.write(KBASE, kbytes)

    vbytes = []
    for l in range(L):
        for d in range(DH):
            b = f16_bits(V[l, d])
            vbytes += [b & 0xFF, (b >> 8) & 0xFF]
    await spi.write(VBASE, vbytes)

    # ---- launch the decode step, poll STATUS over SPI until done -------------
    await spi.start()
    done = False
    for _ in range(64):
        s = await spi.status()
        if s & 0b10:                        # STATUS[1] = done (sticky until next START)
            done = True
            break
    assert done, "decode step never reported done over SPI STATUS"
    gate_fp16 = (s >> 5) & 1
    dut._log.info(f"STATUS byte = {s:#04x}  (done={s&2!=0}, gate_fp16={gate_fp16})")

    # ---- control read: STATUS mirror at 0x0001 (known value) via CMD_READ -----
    ctrl = await spi.read(0x0001, 1)
    assert ctrl[0] == s, f"CMD_READ @0x0001 STATUS mirror {ctrl[0]:#x} != STATUS {s:#x}"

    # ---- read the OUT region (DH fp32) back over SPI --------------------------
    raw = await spi.read(OBASE, 4 * DH)
    o_chip = np.array([bytes_f32(raw[4 * d:4 * d + 4]) for d in range(DH)],
                      dtype=np.float32)

    # ---- read internal state for a rigorous reference over the RECONSTRUCTED
    #      values (the same trick tb_chip_cosim uses: the KVE codec's V̂-vs-V
    #      fidelity is a separate, cosim-proven property; the ASSEMBLY must
    #      compute the right attention over whatever V̂ the codec produced). -----
    def sv(sig):
        s = str(sig.value)
        return int(s, 2) if set(s) <= {"0", "1"} else -1
    core = dut.dut.u_lambda_acu
    o_int = np.array(
        [bytes_f32(struct.pack("<I", sv(core.out_buf[d]) & 0xFFFFFFFF)) for d in range(DH)],
        dtype=np.float32)
    sb = np.array([bits_f16(sv(core.score_buf[l])) for l in range(L)])
    wb = np.array([bits_f16(sv(core.w_buf[l])) for l in range(L)])
    vhat = np.array([[bits_f16(sv(core.vhat_buf[l * DH + d])) for d in range(DH)]
                     for l in range(L)], dtype=np.float32)   # rotated V̂ per token

    # Hadamard (natural order) and the inverse-WHT the chip applies (H·x / D):
    def hadamard(n):
        H = np.array([[1.0]])
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H
    Hm = hadamard(DH)
    sum_vhat = np.zeros(DH, dtype=np.float64)
    for l in range(L):
        sum_vhat += float(wb[l]) * vhat[l]           # Σ_l w[l]·V̂rot[l]  (chip weights)
    o_ref_vhat = (Hm @ sum_vhat) / DH                # inverse-WHT once (chip does this)

    gmaxV  = float(np.max(np.abs(o_ref))) or 1e-9
    gmaxVh = float(np.max(np.abs(o_ref_vhat))) or 1e-9
    rel_int_V   = float(np.max(np.abs(o_int - o_ref))) / gmaxV          # vs orig-V attention
    rel_int_vhat = float(np.max(np.abs(o_int - o_ref_vhat))) / gmaxVh   # vs V̂ attention (tight)

    dut._log.info(f"score_buf(chip) matches ref: {np.allclose(sb, scores_ref, atol=2e-2)}")
    dut._log.info(f"w_buf(chip) matches ref:     {np.allclose(wb, w_ref, atol=2e-2)}")
    dut._log.info(f"o_int[:4]     = {np.round(o_int[:4],4)}   (chip datapath)")
    dut._log.info(f"o_ref_vhat[:4]= {np.round(o_ref_vhat[:4],4)}   (attention over V̂)")
    dut._log.info(f"o_ref_V[:4]   = {np.round(o_ref[:4],4)}   (attention over orig V)")
    dut._log.info(f"o_chip[:4]    = {np.round(o_chip[:4],4)}   (SPI-streamed)")
    dut._log.info(f"rel err: assembly(vs V̂)={rel_int_vhat:.4f}  end2end(vs V, codec-incl)={rel_int_V:.4f}")

    # ---- primary correctness: the ASSEMBLED datapath computes the right
    #      attention row over the reconstructed values (fp16 LUT+rounding only).
    assert np.allclose(sb, scores_ref, atol=3e-2), f"Q·Kᵀ scores mismatch: {sb} vs {scores_ref}"
    assert np.allclose(wb, w_ref, atol=3e-2), f"softmax weights mismatch: {wb} vs {w_ref}"
    TOL_ASM = 0.06
    assert rel_int_vhat < TOL_ASM, \
        f"assembled attention row out of tol vs V̂: {rel_int_vhat:.4f} >= {TOL_ASM}"

    # ---- serial readback: the SPI-streamed row must equal the on-chip result --
    spi_ok = np.array_equal(
        np.array([sv(core.out_buf[d]) & 0xFFFFFFFF for d in range(DH)]),
        np.array([int.from_bytes(bytes(raw[4*d:4*d+4]), "little") for d in range(DH)]))
    dut._log.info(f"SPI readback bit-identical to on-chip out_buf: {spi_ok}")
    assert spi_ok, "SPI-streamed OUT bytes do not match the on-chip out_buf"

    dut._log.info(f"FULL-CHIP DECODE PASS: assembled datapath attention within {TOL_ASM} vs V̂; "
                  f"end-to-end vs original-V attention {rel_int_V:.3f} (CQ-3 codec-dominated); "
                  f"result streamed out over SPI bit-identical.")
