description: Finalizes the decoding process. This op performs a *weak* sanity check, and the

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.entropy_decode_finalize" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.entropy_decode_finalize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Finalizes the decoding process. This op performs a *weak* sanity check, and the

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.entropy_decode_finalize(
    handle, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

return value may be false if some catastrophic error has happened. This is a
quite weak safety device, and one should not rely on this for error detection.

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
A `Tensor` of type `bool`.
</td>
</tr>

</table>

