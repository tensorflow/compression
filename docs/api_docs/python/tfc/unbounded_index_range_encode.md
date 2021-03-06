description: Range encodes unbounded integer data using an indexed probability table.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.unbounded_index_range_encode" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.unbounded_index_range_encode

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Range encodes unbounded integer `data` using an indexed probability table.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.unbounded_index_range_encode(
    data, index, cdf, cdf_size, offset, precision, overflow_width, debug_level=1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Arguments `data` and `index` should have the same shape. `data` contains the
values to be encoded. For each value in `data`, the corresponding value in
`index` determines which row in `cdf` should be used to encode the value in
`data`. `index` also determines which element in `offset` vector determines the
integer interval the cdf applies to. Naturally, the elements of `index` should
be in the half-open interval `[0, cdf.shape[0])`.

The argument `cdf` is a 2-D tensor and each of its rows contains a CDF. The
argument `cdf_size` is a 1-D tensor, and its length should be the same as the
number of rows of `cdf`. The values in `cdf_size` denote the length of CDF
vector in the corresponding row of `cdf`.

For i = 0,1,..., let `m = cdf_size[i] - 1`, i.e., all the "regular" data values
associated with `index == i` should be in the half-open interval
`[offset[i], offset[i] + m)`. (More details below about regular and non-regular
values.) Then

```
   cdf[..., 0] / 2^precision = Pr(0 <= X - offset[i] < 0) = 0
   cdf[..., 1] / 2^precision = Pr(0 <= X - offset[i] < 1)
   cdf[..., 2] / 2^precision = Pr(0 <= X - offset[i] < 2)
   ...
   cdf[..., m-1] / 2^precision = Pr(0 <= X - offset[i] < m-1).
   cdf[..., m] / 2^precision = 1.
```

We require that `1 < m < cdf.shape[-1]` and that all elements of `cdf` be in the
closed interval `[0, 2^precision]`.

Note that the last CDF entry is the probability that `X - offset[i]` is any
value, including the events `X - offset[i] < 0` and `m - 1 <= X - offset[i]`.
When a value from `data` is regular and is in the interval
`[offset[i], offset[i] + m - 1)`, then the value minus `offset[i]` is range
encoded using the CDF values. The maximum value in each CDF (`m - 1`) is an
overflow code. When a value from `data` is outside of the previous interval, the
overflow code is range encoded, followed by a variable-length encoding of the
actual data value.

The encoded output contains neither the shape information of the encoded data
nor a termination symbol. Therefore the shape of the encoded data must be
explicitly provided to the decoder.

#### Implementation notes:



- Because of potential performance issues, the op does not check if `cdf`
satisfies monotonic increase property.

- For the range coder to decode the encoded string correctly, the decoder should
be able to reproduce the internal states of the encoder precisely. Otherwise,
the decoding would fail and once an error occur, all subsequent decoded values
are incorrect. For this reason, the range coder uses integer arithmetics and
avoids using any floating point operations internally, and `cdf` should contain
integers representing quantized probability mass rather than floating points.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
A `Tensor` of type `int32`. An int32 tensor.
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
The number of bits for probability quantization. Must be <= 16.
</td>
</tr><tr>
<td>
`overflow_width`
</td>
<td>
An `int` that is `>= 1`.
The bit width of the variable-length overflow code. Must be <=
precision.
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
A `Tensor` of type `string`.
A range-coded scalar string and a prefix varint string.
</td>
</tr>

</table>

