
# tfc.StaticParameterizer

## Class `StaticParameterizer`

Inherits From: [`Parameterizer`](../tfc/Parameterizer.md)

### Aliases:

* Class `tfc.StaticParameterizer`
* Class `tfc.python.layers.parameterizers.StaticParameterizer`



Defined in [`python/layers/parameterizers.py`](https://github.com/tensorflow/compression/tree/master/python/layers/parameterizers.py).

<!-- Placeholder for "Used in" -->

A parameterizer that always returns a constant tensor.

No variables are created, hence the parameter never changes.

#### Args:

* <b>`initializer`</b>: An initializer object which will be called to produce the
    static parameter.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(initializer)
```





## Methods

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





