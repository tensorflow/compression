description: Range-decodes code into an int32 tensor of shape shape.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.range_decode" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.range_decode

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Range-decodes `code` into an int32 tensor of shape `shape`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.range_decode(
    encoded, shape, cdf, precision, debug_level=1, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is the reverse op of `RangeEncode`. The shape of the tensor that was
encoded should be known by the caller.

#### Implementation notes:



- If wrong input was given (e.g., corrupt `encoded` string, or `cdf` or
`precision` do not match encoder), the decode is unsuccessful. Because of
potential performance issues, the decoder does not return error status.

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
A scalar string tensor from RangeEncode.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
A `Tensor` of type `int32`.
An int32 1-D tensor representing the shape of the data encoded by
RangeEncode.
</td>
</tr><tr>
<td>
`cdf`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`precision`
</td>
<td>
An `int` that is `>= 1`.
The number of bits for probability quantization. Must be <= 16, and
must match the precision used by RangeEncode that produced `encoded`.
</td>
</tr><tr>
<td>
`debug_level`
</td>
<td>
An optional `int`. Defaults to `1`. Either 0 or 1.
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
A `Tensor` of type `int16`. An int16 tensor with shape equal to `shape`.
</td>
</tr>

</table>

