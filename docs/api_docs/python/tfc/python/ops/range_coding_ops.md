
# Module: tfc.python.ops.range_coding_ops



Defined in [`python/ops/range_coding_ops.py`](https://github.com/tensorflow/compression/tree/master/python/ops/range_coding_ops.py).

<!-- Placeholder for "Used in" -->

Range coding operations.

## Functions

[`pmf_to_quantized_cdf(...)`](../../../tfc/pmf_to_quantized_cdf.md): Converts PMF to quantized CDF. This op uses floating-point operations

[`range_decode(...)`](../../../tfc/range_decode.md): Decodes a range-coded `code` into an int32 tensor of shape `shape`.

[`range_encode(...)`](../../../tfc/range_encode.md): Using the provided cumulative distribution functions (CDF) inside `cdf`, returns

[`unbounded_index_range_decode(...)`](../../../tfc/unbounded_index_range_decode.md): This is the reverse op of `UnboundedIndexRangeEncode`, and decodes the range

[`unbounded_index_range_encode(...)`](../../../tfc/unbounded_index_range_encode.md): Range encodes unbounded integer `data` using an indexed probability table.

[`check_array_fingerprint(...)`](../../../tfc/check_array_fingerprint.md): Computes the fingerprint of `input` and checks the computed value against

[`array_fingerprint(...)`](../../../tfc/array_fingerprint.md): Produces fingerprint of the input data.

