// residual_buffer.sv — in-flight FP16 hold for the streaming per-channel KEY path
//
// The defining new mechanism of ChannelQuant (contract §3.1): per-channel key
// scaling needs the per-channel max over a GROUP of tokens, unknown until the
// group fills. Incoming key vectors accumulate here in FP16; when the group
// completes (full G or a partial final group) they are read back token-by-token
// and quantized against the just-computed per-channel scales.
//
// Simple indexed store: write appends at `cnt`; `clear` starts a new group
// (asserted with the group's first token); `rd_vec` is an async read by index.
// Sized for G tokens x D fp16. Instantiated by cq_key_path.

`default_nettype none

module residual_buffer #(
    parameter int DIM = 64,    // head_dim D
    parameter int DW  = 16,    // FP16 element width
    parameter int G   = 128    // key group size (tokens)
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // write side (compress): append a key token; clear starts a new group
    input  wire                    wr_valid,
    input  wire [DIM*DW-1:0]       wr_vec,
    input  wire                    clear,
    output wire [$clog2(G+1)-1:0]  fill,     // tokens currently buffered

    // read side (flush): async read a buffered token by index
    input  wire [$clog2(G)-1:0]    rd_idx,
    output wire [DIM*DW-1:0]       rd_vec
);

    reg [DIM*DW-1:0]      mem [0:G-1];
    reg [$clog2(G+1)-1:0] cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= '0;
        end else if (clear) begin
            if (wr_valid) begin mem[0] <= wr_vec; cnt <= 'd1; end
            else                cnt <= '0;
        end else if (wr_valid) begin
            mem[cnt] <= wr_vec;
            cnt      <= cnt + 'd1;
        end
    end

    assign fill   = cnt;
    assign rd_vec = mem[rd_idx];

endmodule

`default_nettype wire
