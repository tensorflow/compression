
# tfc.irdft_matrix

### Aliases:

* `tfc.irdft_matrix`
* `tfc.python.ops.spectral_ops.irdft_matrix`

``` python
tfc.irdft_matrix(
    shape,
    dtype=tf.float32
)
```



Defined in [`python/ops/spectral_ops.py`](https://github.com/tensorflow/compression/tree/master/python/ops/spectral_ops.py).

<!-- Placeholder for "Used in" -->

Matrix for implementing kernel reparameterization with `tf.matmul`.

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