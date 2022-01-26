description: Creates range decoder objects to be used by EntropyDecode* ops.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.create_range_decoder" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.create_range_decoder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates range decoder objects to be used by `EntropyDecode*` ops.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.create_range_decoder(
    encoded, lookup, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The input `encoded` is referenced by `handle`. No op should modify the strings
contained in `encoded` while `handle` is alive.

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
A string tensor which contains the code stream. Typically produced by
`EntropyEncodeFinalize`.
</td>
</tr><tr>
<td>
`lookup`
</td>
<td>
A `Tensor` of type `int32`.
An int32 1-D or 2-D tensor. This should match the `lookup` argument of
the corresponding `CreateRangeEncoder` op.
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

