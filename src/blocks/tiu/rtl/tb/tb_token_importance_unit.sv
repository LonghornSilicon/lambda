// tb_token_importance_unit.sv — directed + randomized self-checking testbench.
//
// Maintains a shadow (score,valid) model that mirrors the DUT's exact eviction
// semantics (serialized argmin, strict-less, seeded at slot 0) and checks every
// EVICT victim plus saturation and free-on-evict behaviour.
`timescale 1ns/1ps

module tb_token_importance_unit;
    localparam integer N_SLOTS      = 8;
    localparam integer SCORE_WIDTH  = 8;
    localparam integer WEIGHT_WIDTH = 8;
    localparam integer SLOT_WIDTH   = 3;
    localparam [SCORE_WIDTH-1:0] SCORE_MAX = {SCORE_WIDTH{1'b1}};

    reg clk = 0, rst_n = 0;
    reg acc_valid = 0; reg [SLOT_WIDTH-1:0] acc_slot = 0; reg [WEIGHT_WIDTH-1:0] acc_weight = 0;
    reg ld_valid = 0;  reg [SLOT_WIDTH-1:0] ld_slot = 0;
    reg evict_req = 0;
    reg [SCORE_WIDTH-1:0] tier_threshold = 0;
    wire [N_SLOTS-1:0] tier_keep;
    wire evict_valid; wire [SLOT_WIDTH-1:0] evict_slot; wire busy;

    token_importance_unit #(
        .N_SLOTS(N_SLOTS), .SCORE_WIDTH(SCORE_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .acc_valid(acc_valid), .acc_slot(acc_slot), .acc_weight(acc_weight),
        .ld_valid(ld_valid), .ld_slot(ld_slot),
        .evict_req(evict_req), .evict_valid(evict_valid), .evict_slot(evict_slot),
        .tier_threshold(tier_threshold), .tier_keep(tier_keep),
        .busy(busy)
    );

    always #5 clk = ~clk;

    // Shadow model
    integer sh_score [0:N_SLOTS-1];
    integer sh_valid [0:N_SLOTS-1];
    integer tests = 0, passed = 0;
    integer i;

    task do_reset;
        begin
            rst_n = 0; acc_valid=0; ld_valid=0; evict_req=0;
            @(posedge clk); @(posedge clk);
            rst_n = 1; @(posedge clk);
            for (i=0;i<N_SLOTS;i=i+1) begin sh_score[i]=0; sh_valid[i]=0; end
        end
    endtask

    task do_load(input [SLOT_WIDTH-1:0] s);
        begin
            @(negedge clk); ld_valid=1; ld_slot=s;
            @(posedge clk); #1 ld_valid=0;
            sh_score[s]=0; sh_valid[s]=1;
        end
    endtask

    task do_acc(input [SLOT_WIDTH-1:0] s, input [WEIGHT_WIDTH-1:0] w);
        integer nv;
        begin
            @(negedge clk); acc_valid=1; acc_slot=s; acc_weight=w;
            @(posedge clk); #1 acc_valid=0;
            nv = sh_score[s] + w;
            if (nv > SCORE_MAX) nv = SCORE_MAX;
            sh_score[s] = nv;
        end
    endtask

    // expected victim: mirror DUT exactly (seed slot 0, strict-less scan 0..N-1)
    function integer expected_victim;
        integer em_score, em_idx, j;
        begin
            em_score = sh_valid[0] ? sh_score[0] : SCORE_MAX;
            em_idx = 0;
            for (j=0;j<N_SLOTS;j=j+1)
                if (sh_valid[j] && sh_score[j] < em_score) begin
                    em_score = sh_score[j]; em_idx = j;
                end
            expected_victim = em_idx;
        end
    endfunction

    task do_evict;
        integer exp_idx;
        begin
            exp_idx = expected_victim();
            @(negedge clk); evict_req=1;
            @(posedge clk); #1 evict_req=0;
            wait (evict_valid == 1'b1);
            #1;
            tests = tests + 1;
            if (evict_slot === exp_idx[SLOT_WIDTH-1:0]) passed = passed + 1;
            else $display("  MISMATCH evict: got %0d expected %0d", evict_slot, exp_idx);
            sh_valid[exp_idx] = 0;   // DUT frees the victim
            @(posedge clk);
        end
    endtask

    integer r, s, w;
    initial begin
        do_reset();

        // Directed: load 4 slots, give them distinct masses, evict min
        do_load(0); do_load(1); do_load(2); do_load(3);
        do_acc(0, 100); do_acc(1, 30); do_acc(2, 200); do_acc(3, 30);
        do_evict();  // min mass = slot1 or slot3 (tie 30) -> first = slot1

        // After eviction slot1 freed; accumulate more, evict again
        do_acc(2, 50); do_acc(0, 10);
        do_evict();  // among {0:110,2:250,3:30} -> slot3

        // Saturation: hammer slot0 past SCORE_MAX
        for (r=0;r<600;r=r+1) do_acc(0, 8'hFF);
        tests = tests + 1;
        if (dut.score[0] === SCORE_MAX) begin passed = passed + 1;
            $display("  saturation OK: score[0] = 0x%0h", dut.score[0]); end
        else $display("  MISMATCH saturation: score[0]=0x%0h", dut.score[0]);

        // Reload the freed slots and evict again
        do_load(1); do_load(3);
        do_acc(1, 5);
        do_evict();  // slot1 has mass 5, the new min

        // Tier handshake: set a threshold, check tier_keep matches valid && score>=thr
        tier_threshold = 8'd40;
        #1;
        for (i = 0; i < N_SLOTS; i = i + 1) begin
            tests = tests + 1;
            if (tier_keep[i] === (sh_valid[i] && (sh_score[i] >= tier_threshold)))
                passed = passed + 1;
            else $display("  MISMATCH tier[%0d]: got %b exp %b (valid=%0d score=%0d thr=%0d)",
                          i, tier_keep[i], (sh_valid[i] && (sh_score[i]>=tier_threshold)),
                          sh_valid[i], sh_score[i], tier_threshold);
        end

        // Randomized soak
        for (r=0;r<200;r=r+1) begin
            s = $random % N_SLOTS; if (s<0) s=-s;
            w = $random % 256; if (w<0) w=-w;
            if (($random % 4) == 0) do_load(s[SLOT_WIDTH-1:0]);
            else if (($random % 8) == 0) begin
                // ensure at least one valid slot before evicting
                for (i=0;i<N_SLOTS;i=i+1) if (sh_valid[i]) begin do_evict(); i=N_SLOTS; end
            end else do_acc(s[SLOT_WIDTH-1:0], w[WEIGHT_WIDTH-1:0]);
        end

        $display("");
        $display("Tests: %0d  Pass: %0d", tests, passed);
        if (tests == passed) $display("ALL TESTS PASSED");
        else $display("FAILED (%0d/%0d)", passed, tests);
        $finish;
    end
endmodule
