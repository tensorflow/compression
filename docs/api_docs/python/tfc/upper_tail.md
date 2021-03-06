description: Approximates upper tail quantile for range coding.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.upper_tail" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.upper_tail

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/distributions/helpers.py#L170-L203">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Approximates upper tail quantile for range coding.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.upper_tail(
    distribution, tail_mass
)
</code></pre>



<!-- Placeholder for "Used in" -->

For range coding of random variables, the distribution tails need special
handling, because range coding can only handle alphabets with a finite
number of symbols. This method returns a cut-off location for the upper
tail, such that approximately `tail_mass` probability mass is contained in
the tails (together). The tails are then handled by using the 'overflow'
functionality of the range coder implementation (using a Golomb-like
universal code).

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
</tr><tr>
<td>
`tail_mass`
</td>
<td>
Float between 0 and 1. Desired probability mass for the tails.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `tf.Tensor` broadcastable to shape `self.batch_shape` containing the
approximate upper tail quantiles for each scalar distribution.
</td>
</tr>

</table>

