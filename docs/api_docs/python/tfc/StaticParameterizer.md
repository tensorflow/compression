<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.StaticParameterizer" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfc.StaticParameterizer

## Class `StaticParameterizer`

Inherits From: [`Parameterizer`](../tfc/Parameterizer.md)

A parameterization object that always returns a constant tensor.

No variables are created, hence the parameter never changes.

#### Args:

* <b>`initializer`</b>: An initializer object which will be called to produce the
    static parameter.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(initializer)
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





