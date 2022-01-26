description: Encodes each input in value.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.entropy_encode_channel" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.entropy_encode_channel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Encodes each input in `value`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.entropy_encode_channel(
    handle, value, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

In general, entropy encoders in `handle` reference multiple distributions.
The last (innermost) dimension of `value` determines which distribution is used
to encode `value`. For example, if `value` is a 3-D array, then `value(i,j,k)`
is encoded using the `k`-th distribution.

`handle` controls the number of coding streams. Suppose that `value` has the
shape `[2, 3, 4]` and that `handle` has shape `[2]`. Then the first slice
`[0, :, :]` of shape `[3, 4]` is encoded into `handle[0]` and the second
slice `[1, :, :]` is encoded into `handle[1]`. If `handle` has shape `[]`, then
there is only one handle, and the entire input is encoded into a single stream.

Values must be in the provided ranges specified when the input `handle` was
originally created, unless overflow functionality was enabled. The `handle` may
be produced by the `CreateRangeEncoder` op, or may be passed through from a
different `EntropyEncodeChannel/EntropyEncodeIndex` op.

Because the op modifies `handle`, the corresponding input edge to the op nodes
of this type should not have other consumers in the graph.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`handle`
</td>
<td>
A `Tensor` of type `variant`.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`.
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
A `Tensor` of type `variant`.
</td>
</tr>

</table>

