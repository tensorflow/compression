description: Matrix for implementing kernel reparameterization with tf.matmul.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.irdft_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.irdft_matrix

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/spectral_ops.py#L27-L70">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Matrix for implementing kernel reparameterization with `tf.matmul`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.irdft_matrix(
    shape, dtype=tf.float32
)
</code></pre>



<!-- Placeholder for "Used in" -->

This can be used to represent a kernel with the provided shape in the RDFT
domain.

Example code for kernel creation, assuming 2D kernels:

```
def create_kernel(init):
  shape = init.shape
  matrix = irdft_matrix(shape[:2])
  init = tf.reshape(init, (shape[0] * shape[1], shape[2] * shape[3]))
  init = tf.matmul(tf.transpose(matrix), init)
  kernel = tf.Variable(init)
  kernel = tf.matmul(matrix, kernel)
  kernel = tf.reshape(kernel, shape)
  return kernel
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`
</td>
<td>
Iterable of integers. Shape of kernel to apply this matrix to.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
`dtype` of returned matrix.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
`Tensor` of shape `(prod(shape), prod(shape))` and dtype `dtype`.
</td>
</tr>

</table>

