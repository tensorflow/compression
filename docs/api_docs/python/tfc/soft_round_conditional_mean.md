description: Conditional mean of inputs given noisy soft rounded values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.soft_round_conditional_mean" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.soft_round_conditional_mean

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/soft_round_ops.py#L92-L111">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Conditional mean of inputs given noisy soft rounded values.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.soft_round_conditional_mean(
    inputs, alpha
)
</code></pre>



<!-- Placeholder for "Used in" -->

Computes g(z) = E[Y | s(Y) + U = z] where s is the soft-rounding function,
U is uniform between -0.5 and 0.5 and `Y` is considered uniform when truncated
to the interval [z-0.5, z+0.5].

This is described in Sec. 4.1. in the paper
> "Universally Quantized Neural Compression"<br />
> Eirikur Agustsson & Lucas Theis<br />
> https://arxiv.org/abs/2006.09952

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`
</td>
<td>
The input tensor.
</td>
</tr><tr>
<td>
`alpha`
</td>
<td>
The softround alpha.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The conditional mean, of same shape as `inputs`.
</td>
</tr>

</table>

