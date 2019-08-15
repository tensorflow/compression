<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.StaticParameterizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfc.StaticParameterizer


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameterizers.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `StaticParameterizer`

A parameterizer that returns a non-variable.

Inherits From: [`Parameterizer`](../tfc/Parameterizer.md)

### Aliases:

* Class `tfc.python.layers.parameterizers.StaticParameterizer`


<!-- Placeholder for "Used in" -->

No variables are created, and `getter` is ignored. If `value` is a `Tensor`,
the parameter can depend on some other computation. Otherwise, it never
changes.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameterizers.py">View source</a>

``` python
__init__(value)
```

Initializer.


#### Arguments:


* <b>`value`</b>: Either a constant or `Tensor` value, or a callable which returns
  such a thing given a shape and dtype argument (for example, an
  initializer).



## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameterizers.py">View source</a>

``` python
__call__(
    getter,
    name,
    shape,
    dtype,
    initializer,
    regularizer=None
)
```

Call self as a function.




