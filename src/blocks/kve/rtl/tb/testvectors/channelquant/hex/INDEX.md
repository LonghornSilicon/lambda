# Golden-vector .hex images (`$readmemh`-loadable)

One value per line, MSB-first, fixed width per word (`u8`=2 hex, `f16`=4 hex raw half, `f32`=8 hex raw single). Canonical byte stream per `docs/HW_CONTRACT.md` §5 — reframe to your bus width. Source of truth is the sibling `*.npz`; regenerate with `reference/export_hex.py`.

## d128_T100_G128__CQ4
| file | kind | hex/word | lines | src shape |
|---|---|---|---|---|
| `input_k.f16.hex` | f16 | 4 | 12800 | (100, 128) |
| `input_v.f16.hex` | f16 | 4 | 12800 | (100, 128) |
| `key_payload.u8.hex` | u8 | 2 | 6400 | (6400,) |
| `key_scales.f16.hex` | f16 | 4 | 128 | (128,) |
| `val_payload.u8.hex` | u8 | 2 | 6400 | (6400,) |
| `val_scales.f16.hex` | f16 | 4 | 100 | (100,) |
| `sidecar.f16.hex` | f16 | 4 | 0 | (100, 0) |
| `outlier_mask.u8.hex` | u8 | 2 | 128 | (128,) |
| `expected_k_hat.f32.hex` | f32 | 8 | 12800 | (100, 128) |
| `expected_v_hat.f32.hex` | f32 | 8 | 12800 | (100, 128) |

## d128_T100_G128__CQ4plus
| file | kind | hex/word | lines | src shape |
|---|---|---|---|---|
| `input_k.f16.hex` | f16 | 4 | 12800 | (100, 128) |
| `input_v.f16.hex` | f16 | 4 | 12800 | (100, 128) |
| `key_payload.u8.hex` | u8 | 2 | 6300 | (6300,) |
| `key_scales.f16.hex` | f16 | 4 | 126 | (126,) |
| `val_payload.u8.hex` | u8 | 2 | 6400 | (6400,) |
| `val_scales.f16.hex` | f16 | 4 | 100 | (100,) |
| `sidecar.f16.hex` | f16 | 4 | 200 | (100, 2) |
| `outlier_mask.u8.hex` | u8 | 2 | 128 | (128,) |
| `expected_k_hat.f32.hex` | f32 | 8 | 12800 | (100, 128) |
| `expected_v_hat.f32.hex` | f32 | 8 | 12800 | (100, 128) |

## d128_T100_G128__CQ8
| file | kind | hex/word | lines | src shape |
|---|---|---|---|---|
| `input_k.f16.hex` | f16 | 4 | 12800 | (100, 128) |
| `input_v.f16.hex` | f16 | 4 | 12800 | (100, 128) |
| `key_payload.u8.hex` | u8 | 2 | 12800 | (12800,) |
| `key_scales.f16.hex` | f16 | 4 | 100 | (100,) |
| `val_payload.u8.hex` | u8 | 2 | 12800 | (12800,) |
| `val_scales.f16.hex` | f16 | 4 | 100 | (100,) |
| `sidecar.f16.hex` | f16 | 4 | 0 | (100, 0) |
| `outlier_mask.u8.hex` | u8 | 2 | 128 | (128,) |
| `expected_k_hat.f32.hex` | f32 | 8 | 12800 | (100, 128) |
| `expected_v_hat.f32.hex` | f32 | 8 | 12800 | (100, 128) |

## d64_T128_G64__CQ4
| file | kind | hex/word | lines | src shape |
|---|---|---|---|---|
| `input_k.f16.hex` | f16 | 4 | 8192 | (128, 64) |
| `input_v.f16.hex` | f16 | 4 | 8192 | (128, 64) |
| `key_payload.u8.hex` | u8 | 2 | 4096 | (4096,) |
| `key_scales.f16.hex` | f16 | 4 | 128 | (128,) |
| `val_payload.u8.hex` | u8 | 2 | 4096 | (4096,) |
| `val_scales.f16.hex` | f16 | 4 | 128 | (128,) |
| `sidecar.f16.hex` | f16 | 4 | 0 | (128, 0) |
| `outlier_mask.u8.hex` | u8 | 2 | 64 | (64,) |
| `expected_k_hat.f32.hex` | f32 | 8 | 8192 | (128, 64) |
| `expected_v_hat.f32.hex` | f32 | 8 | 8192 | (128, 64) |

## d64_T128_G64__CQ4plus
| file | kind | hex/word | lines | src shape |
|---|---|---|---|---|
| `input_k.f16.hex` | f16 | 4 | 8192 | (128, 64) |
| `input_v.f16.hex` | f16 | 4 | 8192 | (128, 64) |
| `key_payload.u8.hex` | u8 | 2 | 3968 | (3968,) |
| `key_scales.f16.hex` | f16 | 4 | 124 | (124,) |
| `val_payload.u8.hex` | u8 | 2 | 4096 | (4096,) |
| `val_scales.f16.hex` | f16 | 4 | 128 | (128,) |
| `sidecar.f16.hex` | f16 | 4 | 256 | (128, 2) |
| `outlier_mask.u8.hex` | u8 | 2 | 64 | (64,) |
| `expected_k_hat.f32.hex` | f32 | 8 | 8192 | (128, 64) |
| `expected_v_hat.f32.hex` | f32 | 8 | 8192 | (128, 64) |

## d64_T128_G64__CQ8
| file | kind | hex/word | lines | src shape |
|---|---|---|---|---|
| `input_k.f16.hex` | f16 | 4 | 8192 | (128, 64) |
| `input_v.f16.hex` | f16 | 4 | 8192 | (128, 64) |
| `key_payload.u8.hex` | u8 | 2 | 8192 | (8192,) |
| `key_scales.f16.hex` | f16 | 4 | 128 | (128,) |
| `val_payload.u8.hex` | u8 | 2 | 8192 | (8192,) |
| `val_scales.f16.hex` | f16 | 4 | 128 | (128,) |
| `sidecar.f16.hex` | f16 | 4 | 0 | (128, 0) |
| `outlier_mask.u8.hex` | u8 | 2 | 64 | (64,) |
| `expected_k_hat.f32.hex` | f32 | 8 | 8192 | (128, 64) |
| `expected_v_hat.f32.hex` | f32 | 8 | 8192 | (128, 64) |

## d64_T70_G64__CQ4
| file | kind | hex/word | lines | src shape |
|---|---|---|---|---|
| `input_k.f16.hex` | f16 | 4 | 4480 | (70, 64) |
| `input_v.f16.hex` | f16 | 4 | 4480 | (70, 64) |
| `key_payload.u8.hex` | u8 | 2 | 2240 | (2240,) |
| `key_scales.f16.hex` | f16 | 4 | 128 | (128,) |
| `val_payload.u8.hex` | u8 | 2 | 2240 | (2240,) |
| `val_scales.f16.hex` | f16 | 4 | 70 | (70,) |
| `sidecar.f16.hex` | f16 | 4 | 0 | (70, 0) |
| `outlier_mask.u8.hex` | u8 | 2 | 64 | (64,) |
| `expected_k_hat.f32.hex` | f32 | 8 | 4480 | (70, 64) |
| `expected_v_hat.f32.hex` | f32 | 8 | 4480 | (70, 64) |

## d64_T70_G64__CQ4plus
| file | kind | hex/word | lines | src shape |
|---|---|---|---|---|
| `input_k.f16.hex` | f16 | 4 | 4480 | (70, 64) |
| `input_v.f16.hex` | f16 | 4 | 4480 | (70, 64) |
| `key_payload.u8.hex` | u8 | 2 | 2170 | (2170,) |
| `key_scales.f16.hex` | f16 | 4 | 124 | (124,) |
| `val_payload.u8.hex` | u8 | 2 | 2240 | (2240,) |
| `val_scales.f16.hex` | f16 | 4 | 70 | (70,) |
| `sidecar.f16.hex` | f16 | 4 | 140 | (70, 2) |
| `outlier_mask.u8.hex` | u8 | 2 | 64 | (64,) |
| `expected_k_hat.f32.hex` | f32 | 8 | 4480 | (70, 64) |
| `expected_v_hat.f32.hex` | f32 | 8 | 4480 | (70, 64) |

## d64_T70_G64__CQ8
| file | kind | hex/word | lines | src shape |
|---|---|---|---|---|
| `input_k.f16.hex` | f16 | 4 | 4480 | (70, 64) |
| `input_v.f16.hex` | f16 | 4 | 4480 | (70, 64) |
| `key_payload.u8.hex` | u8 | 2 | 4480 | (4480,) |
| `key_scales.f16.hex` | f16 | 4 | 70 | (70,) |
| `val_payload.u8.hex` | u8 | 2 | 4480 | (4480,) |
| `val_scales.f16.hex` | f16 | 4 | 70 | (70,) |
| `sidecar.f16.hex` | f16 | 4 | 0 | (70, 0) |
| `outlier_mask.u8.hex` | u8 | 2 | 64 | (64,) |
| `expected_k_hat.f32.hex` | f32 | 8 | 4480 | (70, 64) |
| `expected_v_hat.f32.hex` | f32 | 8 | 4480 | (70, 64) |
