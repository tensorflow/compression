<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.RDFTParameterizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfc.RDFTParameterizer


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameterizers.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `RDFTParameterizer`

Object encapsulating RDFT reparameterization.

Inherits From: [`Parameterizer`](../tfc/Parameterizer.md)

### Aliases:

* Class `tfc.python.layers.parameterizers.RDFTParameterizer`


<!-- Placeholder for "Used in" -->

This uses the real-input discrete Fourier transform (RDFT) of a kernel as
its parameterization. The inverse RDFT is applied to the variable to produce
the parameter.

(see https://en.wikipedia.org/wiki/Discrete_Fourier_transform)

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameterizers.py">View source</a>

``` python
__init__(dc=True)
```

Initializer.


#### Arguments:


* <b>`dc`</b>: Boolean. If `False`, the DC component of the kernel RDFTs is not
  represented, forcing the filters to be highpass. Defaults to `True`.



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




