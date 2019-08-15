<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.irdft_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.irdft_matrix


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/spectral_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Matrix for implementing kernel reparameterization with `tf.matmul`.

### Aliases:

* `tfc.python.ops.spectral_ops.irdft_matrix`


``` python
tfc.irdft_matrix(
    shape,
    dtype=tf.float32
)
```



<!-- Placeholder for "Used in" -->

This can be used to represent a kernel with the provided shape in the RDFT
domain.

Example code for kernel creation, assuming 2D kernels:

```
def create_kernel(init):
  shape = init.shape.as_list()
  matrix = irdft_matrix(shape[:2])
  init = tf.reshape(init, (shape[0] * shape[1], shape[2] * shape[3]))
  init = tf.matmul(tf.transpose(matrix), init)
  kernel = tf.Variable(init)
  kernel = tf.matmul(matrix, kernel)
  kernel = tf.reshape(kernel, shape)
  return kernel
```

#### Args:


* <b>`shape`</b>: Iterable of integers. Shape of kernel to apply this matrix to.
* <b>`dtype`</b>: `dtype` of returned matrix.


#### Returns:

`Tensor` of shape `(prod(shape), prod(shape))` and dtype `dtype`.
