description: Computes distribution-dependent quantization offset.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.quantization_offset" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.quantization_offset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/distributions/helpers.py#L88-L131">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes distribution-dependent quantization offset.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.quantization_offset(
    distribution
)
</code></pre>



<!-- Placeholder for "Used in" -->

For range coding of continuous random variables, the values need to be
quantized first. Typically, it is beneficial for compression performance to
align the centers of the quantization bins such that one of them coincides
with the mode of the distribution. With `offset` being the mode of the
distribution, for instance, this can be achieved simply by computing:
```
x_hat = tf.round(x - offset) + offset
```

This method tries to determine the offset in a best-effort fashion, based on
which statistics the `Distribution` implements. First, a method
`self._quantization_offset()` is tried. If that isn't defined, it tries in
turn: `self.mode()`, `self.quantile(.5)`, then `self.mean()`. If none of
these are implemented, it falls back on quantizing to integer values (i.e.,
an offset of zero).

Note the offset is always in the range [-.5, .5] as it is assumed to be
combined with a round quantizer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`distribution`
</td>
<td>
A `tfp.distributions.Distribution` object.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `tf.Tensor` broadcastable to shape `self.batch_shape`, containing
the determined quantization offsets. No gradients are allowed to flow
through the return value.
</td>
</tr>

</table>

