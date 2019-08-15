<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.IdentityInitializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfc.IdentityInitializer


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/initializers.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `IdentityInitializer`

Initialize to the identity kernel with the given shape.



### Aliases:

* Class `tfc.python.layers.initializers.IdentityInitializer`


<!-- Placeholder for "Used in" -->

This creates an n-D kernel suitable for `SignalConv*` with the requested
support that produces an output identical to its input (except possibly at the
signal boundaries).

Note: The identity initializer in `tf.initializers` is only suitable for
matrices, not for n-D convolution kernels (i.e., no spatial support).

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/initializers.py">View source</a>

``` python
__init__(gain=1)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/initializers.py">View source</a>

``` python
__call__(
    shape,
    dtype=None,
    partition_info=None
)
```

Call self as a function.




