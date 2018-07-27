<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.RDFTParameterizer" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfc.RDFTParameterizer

## Class `RDFTParameterizer`

Inherits From: [`Parameterizer`](../tfc/Parameterizer.md)

Object encapsulating RDFT reparameterization.

This uses the real-input discrete Fourier transform (RDFT) of a kernel as
its parameterization. The inverse RDFT is applied to the variable to produce
the parameter.

(see https://en.wikipedia.org/wiki/Discrete_Fourier_transform)

#### Args:

* <b>`dc`</b>: Boolean. If `False`, the DC component of the kernel RDFTs is not
    represented, forcing the filters to be highpass. Defaults to `True`.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(dc=True)
```



<h3 id="__call__"><code>__call__</code></h3>

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





