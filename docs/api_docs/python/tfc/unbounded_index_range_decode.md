description: Range decodes encoded using an indexed probability table.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.unbounded_index_range_decode" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.unbounded_index_range_decode

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Range decodes `encoded` using an indexed probability table.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.unbounded_index_range_decode(
    encoded, index, cdf, cdf_size, offset, precision, overflow_width, debug_level=1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is the reverse op of `UnboundedIndexRangeEncode`, and decodes the range
encoded stream `encoded` into an int32 tensor `decoded`. The other inputs
`index`, `cdf`, `cdf_size`, and `offset` should be the identical tensors passed
to the `UnboundedIndexRangeEncode` op that generated the `decoded` tensor.

#### Implementation notes:



- If a wrong input was given (e.g., a corrupt `encoded` string, or `cdf` or
`precision` not matching the encoder), the decode is unsuccessful. Because of
potential performance issues, the decoder does not return an error status.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`encoded`
</td>
<td>
A `Tensor` of type `string`.
A scalar string tensor from `UnboundedIndexRangeEncode`.
</td>
</tr><tr>
<td>
`index`
</td>
<td>
A `Tensor` of type `int32`.
An int32 tensor of the same shape as `data`.
</td>
</tr><tr>
<td>
`cdf`
</td>
<td>
A `Tensor` of type `int32`.
An int32 tensor representing the CDF's of `data`. Each integer is divided
by `2^precision` to represent a fraction.
</td>
</tr><tr>
<td>
`cdf_size`
</td>
<td>
A `Tensor` of type `int32`. An int32 tensor.
</td>
</tr><tr>
<td>
`offset`
</td>
<td>
A `Tensor` of type `int32`. An int32 tensor.
</td>
</tr><tr>
<td>
`precision`
</td>
<td>
An `int` that is `>= 1`.
The number of bits for probability quantization. Must be <= 16, and
must match the precision used by `UnboundedIndexRangeEncode` that produced
`encoded`.
</td>
</tr><tr>
<td>
`overflow_width`
</td>
<td>
An `int` that is `>= 1`.
The bit width of the variable-length overflow code. Must be <=
precision, and must match the width used by `UnboundedIndexRangeEncode` that
produced `encoded`.
</td>
</tr><tr>
<td>
`debug_level`
</td>
<td>
An optional `int`. Defaults to `1`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `int32`.
An int32 tensor with the same shape as `index`.
</td>
</tr>

</table>

