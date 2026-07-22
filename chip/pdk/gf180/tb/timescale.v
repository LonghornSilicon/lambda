// Default timescale for GL simulation. Without this, iverilog falls back to
// 1s/1s precision because neither the gf180 stdcell models nor the synthesised
// .nl.v carry an explicit `timescale directive; a cocotb Clock in ns then
// trips on the coarse precision. Compiled ahead of any gf180 stdcell model in
// the GL flow (see the template's tb/Makefile test-*-gl targets).

`timescale 1ns/1ps
