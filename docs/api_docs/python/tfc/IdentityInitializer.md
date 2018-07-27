<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.IdentityInitializer" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfc.IdentityInitializer

## Class `IdentityInitializer`



Initialize to the identity kernel with the given shape.

This creates an n-D kernel suitable for `SignalConv*` with the requested
support that produces an output identical to its input (except possibly at the
signal boundaries).

Note: The identity initializer in `tf.initializers` is only suitable for
matrices, not for n-D convolution kernels (i.e., no spatial support).

## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(gain=1)
```



<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    shape,
    dtype=None,
    partition_info=None
)
```





