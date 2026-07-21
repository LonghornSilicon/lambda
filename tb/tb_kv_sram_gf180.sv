// tb_kv_sram_gf180.sv — KV store round-trip through the REAL GF180 SRAM macro.
//
// Drives the KVE's `sram_controller` store shell wired (via the GF180 `kv_sram`
// tiling wrapper) to real `gf180mcu_fd_ip_sram__sram512x8m8wm1` hard-macro
// SIMULATION MODELS (the sign-off sim view of the hard IP). Writes several KV
// records at a realistic depth (512-word store) and reads them back, checking
// the store returns each record BIT-EXACT through the real macro protocol
// (CEN/GWEN/WEN, registered Q) and that the rd_valid handshake still holds.
//
// This is the "KVE reconstruct reads through the real SRAM" gate-level proof:
// the KV storage is now a real 6T-bitcell SRAM macro, not a flip-flop array.

`timescale 1ns/1ps

module tb_kv_sram_gf180;
    localparam integer DEPTH = 512;   // real KV capacity (512 cached records)
    localparam integer WIDTH = 80;    // TIER-0 per-token record (SCALE_WIDTH + VECTOR_DIM*8)
    localparam integer AW    = 9;

    reg clk = 0, rst_n = 0; always #5 clk = ~clk;
    integer errors = 0;

    reg              wr_en, rd_en;
    reg  [AW-1:0]    wr_addr, rd_addr;
    reg  [WIDTH-1:0] wr_data;
    wire [WIDTH-1:0] rd_data;
    wire             rd_valid, full;
    wire [AW:0]      occupancy;

    // sram_controller instantiates the GF180 kv_sram (real macro banks)
    sram_controller #(
        .SRAM_DEPTH (DEPTH),
        .DATA_WIDTH (WIDTH),
        .ADDR_WIDTH (AW)
    ) u_store (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(rd_data), .rd_valid(rd_valid),
        .occupancy(occupancy), .full(full)
    );

    // reference model of what each address holds
    reg [WIDTH-1:0] ref_mem [0:DEPTH-1];

    integer k;
    reg [AW-1:0] addrs [0:3];
    reg [WIDTH-1:0] recs [0:3];

    task do_write(input [AW-1:0] a, input [WIDTH-1:0] d);
        begin
            @(negedge clk);
            wr_en = 1; wr_addr = a; wr_data = d;
            ref_mem[a] = d;
            @(negedge clk);
            wr_en = 0;
        end
    endtask

    // registered read: assert rd_en for one cycle, rd_valid + rd_data next cycle
    task do_read_check(input [AW-1:0] a);
        reg [WIDTH-1:0] got;
        integer w;
        begin
            @(negedge clk);
            rd_en = 1; rd_addr = a;
            @(negedge clk);
            rd_en = 0;
            // rd_valid pulses the cycle after rd_en; sample it
            w = 0;
            while (rd_valid !== 1'b1 && w < 4) begin @(negedge clk); w = w + 1; end
            got = rd_data;
            if (rd_valid !== 1'b1) begin
                errors = errors + 1;
                $display("  addr %0d: rd_valid never pulsed", a);
            end else if (got !== ref_mem[a]) begin
                errors = errors + 1;
                $display("  addr %0d: got %h exp %h", a, got, ref_mem[a]);
            end else begin
                $display("  addr %0d: read %h  MATCH", a, got);
            end
        end
    endtask

    initial begin
        wr_en = 0; rd_en = 0; wr_addr = 0; rd_addr = 0; wr_data = 0;
        rst_n = 0; repeat(4) @(negedge clk); rst_n = 1; @(negedge clk);

        $display("=== GF180 SRAM-macro KV store round-trip (gf180mcu_fd_ip_sram__sram512x8m8wm1) ===");
        $display("    %0d banks x 512x8 = %0d-bit x %0d-word real SRAM store", (WIDTH+7)/8, WIDTH, DEPTH);

        // realistic KV records at spread-out addresses (incl. first/last of bank)
        addrs[0] = 9'd5;   recs[0] = 80'hDEAD_BEEF_1234_5678_9ABC;
        addrs[1] = 9'd200; recs[1] = 80'h0F1E_2D3C_4B5A_6978_8796;
        addrs[2] = 9'd511; recs[2] = 80'hFFFF_0000_AAAA_5555_C3C3;
        addrs[3] = 9'd42;  recs[3] = 80'h1122_3344_5566_7788_99AA;

        for (k = 0; k < 4; k = k + 1) do_write(addrs[k], recs[k]);
        $display("  wrote 4 records; occupancy=%0d", occupancy);

        // read them back (out of order) and check bit-exact through the macro
        do_read_check(addrs[2]);
        do_read_check(addrs[0]);
        do_read_check(addrs[3]);
        do_read_check(addrs[1]);

        // overwrite one and re-read
        do_write(addrs[0], 80'hCAFE_F00D_0BAD_C0DE_5A5A);
        do_read_check(addrs[0]);

        $display("");
        $display("GF180 SRAM-macro KV store round-trip: %s", (errors==0)?"ALL BIT-EXACT (real SRAM)":"FAILED");
        $finish;
    end
endmodule
