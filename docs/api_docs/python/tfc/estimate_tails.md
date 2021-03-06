description: Estimates approximate tail quantiles.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.estimate_tails" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.estimate_tails

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/distributions/helpers.py#L29-L85">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Estimates approximate tail quantiles.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.estimate_tails(
    func, target, shape, dtype
)
</code></pre>



<!-- Placeholder for "Used in" -->

This runs a simple Adam iteration to determine tail quantiles. The
objective is to find an `x` such that:
```
func(x) == target
```
For instance, if `func` is a CDF and the target is a quantile value, this
would find the approximate location of that quantile. Note that `func` is
assumed to be monotonic. When each tail estimate has passed the optimal value
of `x`, the algorithm does 100 additional iterations and then stops.

This operation is vectorized. The tensor shape of `x` is given by `shape`, and
`target` must have a shape that is broadcastable to the output of `func(x)`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`func`
</td>
<td>
A callable that computes cumulative distribution function, survival
function, or similar.
</td>
</tr><tr>
<td>
`target`
</td>
<td>
The desired target value.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
The shape of the `tf.Tensor` representing `x`.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The `tf.dtypes.Dtype` of the computation (and the return value).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `tf.Tensor` representing the solution (`x`).
</td>
</tr>

</table>

