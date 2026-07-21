# cocotb smoke test for the Lambda ACU workshop macro (chip_core).
#
# Elaborates chip_core (via chip_core_wrap, pinned to the SLOT_WORKSHOP pad
# widths 1/20/60) and exercises the bare minimum: reset behaviour, pad
# direction sanity, and one SPI START frame end to end through the serial
# loader. This is a HARNESS smoke test — real per-macro cocotb tests (kve,
# tiu, precision_controller, mate_pv/_fp16) land as each macro is wired in.
#
#   make test-smoke
#
# SPI on the bidir pads (see rtl/chip_core.sv PAD MAP):
#   bidir_in[0]=sclk  bidir_in[1]=cs_n  bidir_in[2]=mosi   bidir_out[3]=miso

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

CLK_PERIOD_NS = 10

SCLK = 0
CS_N = 1
MOSI = 2

CMD_START = 0x03


async def _reset(dut):
    dut.input_in.value = 0
    dut.bidir_in.value = (1 << CS_N)  # cs_n idle high, sclk/mosi low
    dut.rst_n.value = 0
    await Timer(25, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def _set_bidir(dut, sclk, cs_n, mosi):
    val = (sclk << SCLK) | (cs_n << CS_N) | (mosi << MOSI)
    dut.bidir_in.value = val


async def _spi_byte(dut, byte):
    """Shift one byte MSB-first, SPI mode 0, oversampled well below core clk."""
    for i in range(8):
        bit = (byte >> (7 - i)) & 1
        _set_bidir(dut, 0, 0, bit)          # sclk low, present bit
        for _ in range(4):
            await RisingEdge(dut.clk)
        _set_bidir(dut, 1, 0, bit)          # sclk rising -> loader samples
        for _ in range(4):
            await RisingEdge(dut.clk)
    _set_bidir(dut, 0, 0, 0)                 # sclk low
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_reset_tristates_and_pad_dirs(dut):
    """After reset the SPI input pads are inputs and the rest are outputs."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    await _reset(dut)
    await Timer(1, unit="ns")
    oe = int(dut.bidir_oe.value)
    # pads 0,1,2 must be inputs (oe=0), pads 3..19 outputs (oe=1)
    assert (oe & 0b111) == 0, f"SPI input pads must have oe=0, got oe={oe:#x}"
    assert ((oe >> 3) & 0x1FFFF) == 0x1FFFF, f"output pads must have oe=1, got oe={oe:#x}"


@cocotb.test()
async def test_spi_start_frame_runs(dut):
    """A START frame over SPI must pulse the datapath busy/done handshake,
    observable on the bidir observation bus."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    await _reset(dut)

    # frame: cs_n low, send CMD_START, cs_n high
    _set_bidir(dut, 0, 1, 0)
    await RisingEdge(dut.clk)
    _set_bidir(dut, 0, 0, 0)   # cs_n falling -> frame start
    await RisingEdge(dut.clk)

    await _spi_byte(dut, CMD_START)

    # let the placeholder sequencer complete; observe done on obs bus.
    saw_activity = False
    for _ in range(20):
        await RisingEdge(dut.clk)
        # obs_out is on bidir_out[19:4]; done/busy are the low obs bits
        # (obs_out = {..., done, busy, o_buf[0]}), i.e. bidir_out bits 12/13.
        out = int(dut.bidir_out.value)
        # obs_out = {.., done(bit9), busy(bit8), o_buf[0](bits7:0)} maps straight
        # onto bidir_out[19:4], so busy/done appear at bidir_out[8]/[9].
        if (out >> 8) & 0b11:
            saw_activity = True
            break

    _set_bidir(dut, 0, 1, 0)   # cs_n high -> frame end
    await RisingEdge(dut.clk)
    assert saw_activity, "START frame did not produce busy/done on the observation bus"
