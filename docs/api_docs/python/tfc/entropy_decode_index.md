description: Decodes the encoded stream inside handle.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.entropy_decode_index" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.entropy_decode_index

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Decodes the encoded stream inside `handle`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.entropy_decode_index(
    handle, index, shape, Tdecoded, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The output shape is defined as `handle.shape + MakeShape(shape)`, and therefore
both `handle` and `shape` arguments determine how many symbols are decoded.

Like encoders, decoders in `handle` reference multiple distributions. `index`
indicates which distribution should be used to decode each value in the output.
For example, if `decoded` is a 3-D array, then `output(i,j,k)` is decoded using
the `index(i,j,k)`-th distribution. In general, `index` should match the `index`
of the corresponding `EntropyEncodeIndex` op. `index` should have the same shape
as output `decoded`: `handle.shape + MakeShape(shape)`.

`handle` controls the number of coding streams. Suppose that `index` has the
shape `[2, 3, 4]` and that `handle` has shape `[2]`. Then the first output slice
`[0, :, :]` of shape `[3, 4]` is decoded from `handle[0]` and the second output
slice `[1, :, :]` is decoded from `handle[1]`. If `handle` has shape `[]`, then
there is only one handle, and the entire output is decoded from a single stream.

The input handle may be produced by the `CreateRangeDecoder` op, or may be
passed through from a different `EntropyDecode*` op.

This op modifies the input `handle`. The handle input edge to the op nodes of
this type should not have other consumers in the graph.

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
`index`
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`Tdecoded`
</td>
<td>
A `tf.DType` from: `tf.int32`.
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
A tuple of `Tensor` objects (aliased_handle, decoded).
</td>
</tr>
<tr>
<td>
`aliased_handle`
</td>
<td>
A `Tensor` of type `variant`.
</td>
</tr><tr>
<td>
`decoded`
</td>
<td>
A `Tensor` of type `Tdecoded`.
</td>
</tr>
</table>

