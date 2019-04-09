
# tfc.IdentityInitializer

## Class `IdentityInitializer`



### Aliases:

* Class `tfc.IdentityInitializer`
* Class `tfc.python.layers.initializers.IdentityInitializer`



Defined in [`python/layers/initializers.py`](https://github.com/tensorflow/compression/tree/master/python/layers/initializers.py).

<!-- Placeholder for "Used in" -->

Initialize to the identity kernel with the given shape.

This creates an n-D kernel suitable for `SignalConv*` with the requested
support that produces an output identical to its input (except possibly at the
signal boundaries).

Note: The identity initializer in `tf.initializers` is only suitable for
matrices, not for n-D convolution kernels (i.e., no spatial support).

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(gain=1)
```





## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    shape,
    dtype=None,
    partition_info=None
)
```





