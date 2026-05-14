# KV Cache Engine

**Block 2 of 4** in the LonghornSilicon LLM inference accelerator. Manages on-chip KV tensor storage with TurboQuant+ lossy compression for reduced DRAM bandwidth.

| Metric | Value |
|--------|-------|
| Algorithm | TurboQuant+ turbo4 (PolarQuant 3-bit + QJL 1-bit) |
| K compression | 3.56x (4.25 bpv) |
| V compression | 4.92x (~3.0 bpv) |
| Combined | >= 3x on canonical trace |
| Interface | AXI-Lite control + AXI-Stream data |
| Sky130 target | 80 MHz, behavioral SRAM |
| 16FFC target | 800 MHz, real SRAM macros |

## Architecture

```
                    AXI-Lite (control registers)
                            |
                    +-------v--------+
  s_axis_kv ------>| Input Buffer   |
  (16-bit coords)  | (collect DIM   |
                   |  coordinates)  |
                   +-------+--------+
                           |
              +------------+-------------+
              |                          |
        K path (tuser=0)           V path (tuser=1)
              |                          |
        +-----v------+            +------v-----+
        | Norm Unit   |            | Norm Unit   |
        +-----+------+            +------+-----+
              |                          |
        +-----v------+            +------v-----+
        | Rotation    |            | Rotation    |
        | (WHT+signs) |            | (WHT+signs) |
        +-----+------+            +------+-----+
              |                          |
        +-----v------+            +------v-----+
        | Quantizer   |            | Quantizer   |
        | (3-bit PQ)  |            | (3-bit PQ)  |
        +-----+------+            +------+-----+
              |                          |
        +-----v------+                   |
        | QJL Unit    |                   |
        | (1-bit sign)|                   |
        +-----+------+                   |
              |                          |
        +-----v------+            +------v-----+
        | Packer      |            | Packer      |
        | (288 bits)  |            | (208 bits)  |
        +-----+------+            +------+-----+
              |                          |
              +------------+-------------+
                           |
                    +------v-------+
                    | SRAM         |
                    | Controller   |-----> evict_needed
                    | (16 entries) |-----> evict_addr
                    +------+-------+
                           |
                    +------v-------+
                    | Decompressor |
                    +------+-------+
                           |
                    m_axis_kv ------> (decompressed 16-bit coords)
```

## Repository Structure

```
kv-cache-engine/
├── rtl/                        # SystemVerilog RTL
│   ├── kv_cache_engine.sv      # Top-level module
│   ├── norm_unit.sv            # L2 norm computation
│   ├── rotation_unit.sv        # WHT + random sign flips
│   ├── quantizer.sv            # 3-bit Lloyd-Max quantizer
│   ├── qjl_unit.sv             # QJL 1-bit projection (K only)
│   ├── packer.sv               # Bit-pack/unpack for SRAM
│   ├── sram_controller.sv      # Behavioral SRAM (reg array)
│   ├── decompressor.sv         # Reverse compression pipeline
│   ├── tb/                     # Testbenches + test vectors
│   ├── constraints/            # SDC timing constraints
│   ├── synth.ys                # Yosys synthesis script
│   ├── genus.tcl               # Cadence Genus synthesis
│   ├── innovus.tcl             # Cadence Innovus PnR
│   ├── mmmc.tcl                # Multi-mode multi-corner setup
│   └── Makefile
├── sw/reference_model/         # Bit-exact reference models
│   ├── kv_cache_engine_ref.py  # Python golden reference
│   ├── kv_cache_engine_ref.hpp # C++ header
│   ├── kv_cache_engine_ref.cpp # C++ implementation
│   ├── test_*.py / test_*.cpp  # Test suites
│   └── Makefile
├── analysis/                   # Algorithm exploration + test vector gen
├── openlane/                   # OpenLane Sky130 config
├── docs/isa/                   # Interface specification (LaTeX)
└── .github/workflows/ci.yml   # CI: sim, synth, Sky130, ref model
```

## Quick Start

```bash
# Run RTL testbenches
cd rtl && make sim && make sim_realdata

# Run reference model tests
cd sw/reference_model && make test-all

# Yosys synthesis sanity check
cd rtl && make synth
```

## Register Map

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x00 | CTRL | RW | bit[0]: soft_reset, bit[1]: enable |
| 0x04 | STATUS | R | bit[0]: idle, bit[3]: sram_full |
| 0x08 | INFO_DIM | R | VECTOR_DIM (64) |
| 0x0C | INFO_PQ_BITS | R | PQ quantization bits (3) |
| 0x10 | INFO_QJL_BITS | R | QJL projection bits (1) |
| 0x14 | INFO_SRAM_DEPTH | R | SRAM entries (16) |
| 0x18 | INFO_CR_K | R | Key compression ratio (8.8 fixed-point) |
| 0x1C | INFO_CR_V | R | Value compression ratio (8.8 fixed-point) |
| 0x20 | INFO_VERSION | R | ISA version (0x00010000) |
| 0x24 | OCCUPANCY | R | Valid SRAM entries |
| 0x28 | WRITE_ADDR | RW | Target write address |
| 0x2C | READ_ADDR | RW | Target read address |
| 0x30 | KV_SELECT | RW | 0=key, 1=value |
| 0x34 | IRQ_MASK | RW | Interrupt enable mask |
| 0x38 | IRQ_STATUS | R/W1C | Interrupt pending status |

## Verification Status

| Test Suite | Tests | Status |
|-----------|-------|--------|
| Python reference model | 120 | PASS |
| C++ reference model | 64 | PASS |
| RTL directed TB | 14 | PASS |
| RTL replay TB | 1 | PASS |
| Yosys synthesis | - | PASS |

## Related Blocks

- **Block 1**: [Precision Controller](https://github.com/LonghornSilicon/adaptive-precision-attention) — INT8/FP16 routing
- **Block 3**: Token Importance Unit — eviction priority scoring
- **Block 4**: Memory Hierarchy Controller — SRAM-DRAM spilling

## License

See repository root for license details.
