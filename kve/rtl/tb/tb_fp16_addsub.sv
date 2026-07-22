// tb_fp16_addsub.sv — sweep fp16_addsub_syn vs the behavioral oracle (cq_fp_pkg).
`timescale 1ns/1ps
`include "cq_fp_pkg.sv"
module tb_fp16_addsub;
    import cq_fp_pkg::*;
    reg  [15:0] a, b; reg sub; wire [15:0] y;
    fp16_addsub_syn dut (.a(a), .b(b), .sub(sub), .y(y));
    integer i, checks = 0, pass = 0; reg [15:0] exp;
    task chk; begin
        if (a[14:10] != 5'h1f && b[14:10] != 5'h1f) begin   // skip inf/nan inputs
            #1; exp = sub ? fp16_sub(a, b) : fp16_add(a, b);
            checks = checks + 1;
            if (y === exp) pass = pass + 1;
            else if (checks - pass <= 12)
                $display("  MISMATCH a=%04h b=%04h sub=%0d: syn=%04h oracle=%04h", a, b, sub, y, exp);
        end
    end endtask
    initial begin
        // directed edge cases
        a=16'h0000; b=16'h0000; sub=0; chk;
        a=16'h3c00; b=16'hbc00; sub=0; chk;   // 1 + (-1) = 0
        a=16'h3c00; b=16'h3c00; sub=1; chk;   // 1 - 1 = 0
        a=16'h0001; b=16'h0001; sub=0; chk;   // subnormal + subnormal
        a=16'h7bff; b=16'h0001; sub=0; chk;   // max-normal + tiny
        a=16'h3c00; b=16'h0001; sub=1; chk;   // 1 - subnormal
        // random sweep (values + all exponents)
        for (i = 0; i < 400000; i = i + 1) begin a = $random; b = $random; sub = $random & 1; chk; end
        $display("");
        $display("fp16_addsub_syn vs oracle: %0d/%0d bit-exact", pass, checks);
        if (checks > 0 && pass == checks) $display("ALL TESTS PASSED"); else $display("FAILED (%0d/%0d)", pass, checks);
        $finish;
    end
endmodule
