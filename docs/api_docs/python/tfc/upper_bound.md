description: Same as tf.minimum, but with helpful gradient for inputs > bound.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.upper_bound" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.upper_bound

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/math_ops.py#L27-L89">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Same as `tf.minimum`, but with helpful gradient for `inputs > bound`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.upper_bound(
    inputs, bound, gradient=&#x27;identity_if_towards&#x27;,
    name=&#x27;upper_bound&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function behaves just like `tf.minimum`, but the behavior of the gradient
with respect to `inputs` for input values that hit the bound depends on
`gradient`:

If set to `'disconnected'`, the returned gradient is zero for values that hit
the bound. This is identical to the behavior of `tf.minimum`.

If set to `'identity'`, the gradient is unconditionally replaced with the
identity function (i.e., pretending this function does not exist).

If set to `'identity_if_towards'`, the gradient is replaced with the identity
function, but only if applying gradient descent would push the values of
`inputs` towards the bound. For gradient values that push away from the bound,
the returned gradient is still zero.

Note: In the latter two cases, no gradient is returned for `bound`.
Also, the implementation of `gradient == 'identity_if_towards'` currently
assumes that the shape of `inputs` is the same as the shape of the output. It
won't work reliably for all possible broadcasting scenarios.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Input tensor.
</td>
</tr><tr>
<td>
`bound`
</td>
<td>
Upper bound for the input tensor.
</td>
</tr><tr>
<td>
`gradient`
</td>
<td>
'disconnected', 'identity', or 'identity_if_towards' (default).
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Name for this op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
`tf.minimum(inputs, bound)`
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
for invalid value of `gradient`.
</td>
</tr>
</table>

