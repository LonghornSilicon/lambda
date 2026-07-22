// Blackbox declaration for the GF180 SRAM hard macro. Lets Verilator lint and
// yosys elaborate the kv_sram wrapper without pulling in the timing-annotated
// simulation model; LibreLane hardens against the macro's lib/lef/gds abstract.
// (* blackbox *) tells yosys this is an opaque cell.
(* blackbox *)
module gf180mcu_fd_ip_sram__sram512x8m8wm1 (
    input  wire        CLK,
    input  wire        CEN,
    input  wire        GWEN,
    input  wire [7:0]  WEN,
    input  wire [8:0]  A,
    input  wire [7:0]  D,
    output wire [7:0]  Q,
    inout  wire        VDD,
    inout  wire        VSS
);
endmodule
