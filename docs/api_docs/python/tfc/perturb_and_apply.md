description: Perturbs the inputs of a pointwise function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.perturb_and_apply" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.perturb_and_apply

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/math_ops.py#L157-L216">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Perturbs the inputs of a pointwise function.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.perturb_and_apply(
    f, x, *args, u=None, x_plus_u=None, expected_grads=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function adds uniform noise in the range -0.5 to 0.5 to the first
argument of the given function.
It further replaces derivatives of the function with (analytically computed)
expected derivatives w.r.t. the noise.

This is described in Sec. 4.2. in the paper
> "Universally Quantized Neural Compression"<br />
> Eirikur Agustsson & Lucas Theis<br />
> https://arxiv.org/abs/2006.09952

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`f`
</td>
<td>
Callable. Pointwise function applied after perturbation.
</td>
</tr><tr>
<td>
`x`
</td>
<td>
The inputs.
</td>
</tr><tr>
<td>
`*args`
</td>
<td>
Other arguments to f.
</td>
</tr><tr>
<td>
`u`
</td>
<td>
The noise to perturb x with. If not set and x_plus_u is not provided,
it will be sampled.
</td>
</tr><tr>
<td>
`x_plus_u`
</td>
<td>
Alternative way to provide the noise, as `x+u`.
</td>
</tr><tr>
<td>
`expected_grads`
</td>
<td>
If True, will compute expected gradients.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple (y, x+u) where y=f(x+u, *args) and u is uniform noise, and the
gradient of `y` w.r.t. `x` uses expected derivatives w.r.t. the distribution
of u.
</td>
</tr>

</table>

