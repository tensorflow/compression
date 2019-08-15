<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.python.ops.range_coding_ops" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfc.python.ops.range_coding_ops


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/range_coding_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Range coding operations.

<!-- Placeholder for "Used in" -->


## Functions

[`pmf_to_quantized_cdf(...)`](../../../tfc/pmf_to_quantized_cdf.md): Converts PMF to quantized CDF. This op uses floating-point operations

[`range_decode(...)`](../../../tfc/range_decode.md): Decodes a range-coded `code` into an int32 tensor of shape `shape`.

[`range_encode(...)`](../../../tfc/range_encode.md): Using the provided cumulative distribution functions (CDF) inside `cdf`, returns

[`unbounded_index_range_decode(...)`](../../../tfc/unbounded_index_range_decode.md): This is the reverse op of `UnboundedIndexRangeEncode`, and decodes the range

[`unbounded_index_range_encode(...)`](../../../tfc/unbounded_index_range_encode.md): Range encodes unbounded integer `data` using an indexed probability table.

