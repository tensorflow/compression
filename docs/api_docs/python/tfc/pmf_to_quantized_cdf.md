description: Converts a PMF into a quantized CDF for range coding.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.pmf_to_quantized_cdf" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.pmf_to_quantized_cdf

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Converts a PMF into a quantized CDF for range coding.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.pmf_to_quantized_cdf(
    pmf, precision, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This op uses floating-point operations internally. Therefore the quantized
output may not be consistent across multiple platforms. For entropy encoders and
decoders to have the same quantized CDF on different platforms, the quantized
CDF should be produced once and saved, then the saved quantized CDF should be
used everywhere.

After quantization, if PMF does not sum to 2^precision, then some values of PMF
are increased or decreased to adjust the sum to equal to 2^precision.

Note that the input PMF is pre-quantization. The input PMF is not normalized
by this op prior to quantization. Therefore the user is responsible for
normalizing PMF if necessary.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`pmf`
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`precision`
</td>
<td>
An `int` that is `>= 1`.
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
</td>
</tr>

</table>

