description: Decodes data using run-length and Elias gamma coding.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.run_length_gamma_decode" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.run_length_gamma_decode

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Decodes `data` using run-length and Elias gamma coding.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.run_length_gamma_decode(
    code, shape, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is the inverse operation to `RunLengthGammaEncode`. The shape of the tensor
that was encoded must be known by the caller.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`code`
</td>
<td>
A `Tensor` of type `string`.
An encoded scalar string as returned by `RunLengthGammaEncode`.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
A `Tensor` of type `int32`.
An int32 vector giving the shape of the encoded data.
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
An int32 tensor of decoded values, with shape `shape`.
</td>
</tr>

</table>

