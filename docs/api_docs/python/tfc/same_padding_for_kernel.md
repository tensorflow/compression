description: Determine correct amount of padding for same convolution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.same_padding_for_kernel" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.same_padding_for_kernel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/padding_ops.py#L22-L51">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determine correct amount of padding for `same` convolution.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.same_padding_for_kernel(
    shape, corr, strides_up=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

To implement `'same'` convolutions, we first pad the image, and then perform a
`'valid'` convolution or correlation. Given the kernel shape, this function
determines the correct amount of padding so that the output of the convolution
or correlation is the same size as the pre-padded input.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`
</td>
<td>
Shape of the convolution kernel (without the channel dimensions).
</td>
</tr><tr>
<td>
`corr`
</td>
<td>
Boolean. If `True`, assume cross correlation, if `False`, convolution.
</td>
</tr><tr>
<td>
`strides_up`
</td>
<td>
If this is used for an upsampled convolution, specify the
strides here. (For downsampled convolutions, specify `(1, 1)`: in that
case, the strides don't matter.)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The amount of padding at the beginning and end for each dimension.
</td>
</tr>

</table>

