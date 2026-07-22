"""KV-cache quantization with entropy-based precision selection.

Hardware blocks:
  PrecisionController  -- comparator circuit: entropy + outlier -> precision tag
  QuantizedKVCache     -- integrates controller, quantizer, SRAM, dequantizer
  select_precision     -- the raw comparator function (for direct use)
"""

from kv_cache.precision_controller import (
    PrecisionController,
    select_precision,
    compute_block_entropy,
    detect_outliers,
)
from kv_cache.quantized_cache import (
    QuantizedKVCache,
    CachedBlock,
    quantize_block_int8,
    quantize_block_fp16,
    dequantize_block,
)
