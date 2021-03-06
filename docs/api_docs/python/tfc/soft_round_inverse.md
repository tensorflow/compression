description: Inverse of soft_round().

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.soft_round_inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.soft_round_inverse

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/soft_round_ops.py#L59-L89">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Inverse of soft_round().

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.soft_round_inverse(
    y, alpha, eps=0.001
)
</code></pre>



<!-- Placeholder for "Used in" -->

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
`y`
</td>
<td>
tf.Tensor. Inputs to this function.
</td>
</tr><tr>
<td>
`alpha`
</td>
<td>
Float or tf.Tensor. Controls smoothness of the approximation.
</td>
</tr><tr>
<td>
`eps`
</td>
<td>
Float. Threshold below which soft_round() is assumed to equal the
identity function.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
tf.Tensor
</td>
</tr>

</table>

