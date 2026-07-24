# HIF — Host Interface (PCIe Gen3 x1, M.2 2280)

**Status: spec-only stub — vendor IP + thin controller, no custom RTL yet.** Placeholder in the
canonical block taxonomy. Spec: [`../../../arch.yml`](../../../arch.yml) (block `HIF`); progress:
[`../../../docs/PROGRESS.md`](../../../docs/PROGRESS.md).

**Function (from arch.yml):** PCIe Gen3 x1 endpoint (~1 GB/s sustained) on M.2 2280 form factor —
boot-loads weights into LPDDR5X, receives prompts, streams tokens; CSR access + microcode load;
JTAG on dedicated pins. PHY is vendor IP (Synopsys DesignWare / Cadence, public 16nm datasheets).
Area ≈ 0.55 mm² (0.35 PHY + 0.20 controller/CSR/doorbell).

Mostly an IP-integration block; the controller/CSR glue fills out to the canonical template when
work begins (see [`../../../docs/REVISION_SYNC_SOP.md`](../../../docs/REVISION_SYNC_SOP.md) §5).
