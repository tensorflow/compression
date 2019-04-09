
# Module: tfc.python.ops.range_coding_ops



Defined in [`python/ops/range_coding_ops.py`](https://github.com/tensorflow/compression/tree/master/python/ops/range_coding_ops.py).

<!-- Placeholder for "Used in" -->

Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: range_coding_ops.cc

## Functions

[`array_fingerprint(...)`](../../../tfc/array_fingerprint.md): Produces fingerprint of the input data.

[`array_fingerprint_eager_fallback(...)`](../../../tfc/array_fingerprint_eager_fallback.md): This is the slowpath function for Eager mode.

[`check_array_fingerprint(...)`](../../../tfc/check_array_fingerprint.md): Computes the fingerprint of `input` and checks the computed value against

[`check_array_fingerprint_eager_fallback(...)`](../../../tfc/check_array_fingerprint_eager_fallback.md): This is the slowpath function for Eager mode.

[`deprecated_endpoints(...)`](../../../tfc/deprecated_endpoints.md): Decorator for marking endpoints deprecated.

[`pmf_to_quantized_cdf(...)`](../../../tfc/pmf_to_quantized_cdf.md): Converts PMF to quantized CDF. This op uses floating-point operations

[`pmf_to_quantized_cdf_eager_fallback(...)`](../../../tfc/pmf_to_quantized_cdf_eager_fallback.md): This is the slowpath function for Eager mode.

[`range_decode(...)`](../../../tfc/range_decode.md): Decodes a range-coded `code` into an int32 tensor of shape `shape`.

[`range_decode_eager_fallback(...)`](../../../tfc/range_decode_eager_fallback.md): This is the slowpath function for Eager mode.

[`range_encode(...)`](../../../tfc/range_encode.md): Using the provided cumulative distribution functions (CDF) inside `cdf`, returns

[`range_encode_eager_fallback(...)`](../../../tfc/range_encode_eager_fallback.md): This is the slowpath function for Eager mode.

[`unbounded_index_range_decode(...)`](../../../tfc/unbounded_index_range_decode.md): This is the reverse op of `UnboundedIndexRangeEncode`, and decodes the range

[`unbounded_index_range_decode_eager_fallback(...)`](../../../tfc/unbounded_index_range_decode_eager_fallback.md): This is the slowpath function for Eager mode.

[`unbounded_index_range_encode(...)`](../../../tfc/unbounded_index_range_encode.md): Range encodes unbounded integer `data` using an indexed probability table.

[`unbounded_index_range_encode_eager_fallback(...)`](../../../tfc/unbounded_index_range_encode_eager_fallback.md): This is the slowpath function for Eager mode.

## Other Members

<h3 id="tf_export"><code>tf_export</code></h3>

