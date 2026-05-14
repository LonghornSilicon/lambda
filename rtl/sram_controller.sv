// sram_controller.sv — Behavioral SRAM model with read/write control
//
// For Sky130: synthesized as a reg array (flip-flops). Keep SRAM_DEPTH
// small (e.g., 16) to avoid FF explosion. Real SRAM macros replace this
// at 16FFC time.
//
// FF count: SRAM_DEPTH * DATA_WIDTH (memory array) +
//           ADDR_WIDTH (address regs) + 4 (control)

module sram_controller #(
    parameter integer SRAM_DEPTH  = 16,
    parameter integer DATA_WIDTH  = 288,
    parameter integer ADDR_WIDTH  = $clog2(SRAM_DEPTH)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Write port
    input  wire                    wr_en,
    input  wire [ADDR_WIDTH-1:0]   wr_addr,
    input  wire [DATA_WIDTH-1:0]   wr_data,

    // Read port
    input  wire                    rd_en,
    input  wire [ADDR_WIDTH-1:0]   rd_addr,
    output reg  [DATA_WIDTH-1:0]   rd_data,
    output reg                     rd_valid,

    // Status
    output reg  [ADDR_WIDTH:0]     occupancy,
    output wire                    full
);

    // Behavioral memory array
    reg [DATA_WIDTH-1:0] mem [0:SRAM_DEPTH-1];
    reg [SRAM_DEPTH-1:0] valid_bits;

    assign full = (occupancy >= SRAM_DEPTH);

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data    <= '0;
            rd_valid   <= 1'b0;
            occupancy  <= '0;
            valid_bits <= '0;
            for (i = 0; i < SRAM_DEPTH; i++)
                mem[i] <= '0;
        end else begin
            rd_valid <= 1'b0;

            if (wr_en && wr_addr < SRAM_DEPTH) begin
                mem[wr_addr] <= wr_data;
                if (!valid_bits[wr_addr]) begin
                    valid_bits[wr_addr] <= 1'b1;
                    occupancy <= occupancy + 1;
                end
            end

            if (rd_en && rd_addr < SRAM_DEPTH && valid_bits[rd_addr]) begin
                rd_data  <= mem[rd_addr];
                rd_valid <= 1'b1;
            end
        end
    end

endmodule
