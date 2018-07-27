<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.NonnegativeParameterizer" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfc.NonnegativeParameterizer

## Class `NonnegativeParameterizer`

Inherits From: [`Parameterizer`](../tfc/Parameterizer.md)

Object encapsulating nonnegative parameterization as needed for GDN.

The variable is subjected to an invertible transformation that slows down the
learning rate for small values.

#### Args:

* <b>`minimum`</b>: Float. Lower bound for parameters (defaults to zero).
* <b>`reparam_offset`</b>: Float. Offset added to the reparameterization of beta and
    gamma. The reparameterization of beta and gamma as their square roots lets
    the training slow down when their values are close to zero, which is
    desirable as small values in the denominator can lead to a situation where
    gradient noise on beta/gamma leads to extreme amounts of noise in the GDN
    activations. However, without the offset, we would get zero gradients if
    any elements of beta or gamma were exactly zero, and thus the training
    could get stuck. To prevent this, we add this small constant. The default
    value was empirically determined as a good starting point. Making it
    bigger potentially leads to more gradient noise on the activations, making
    it too small may lead to numerical precision issues.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(
    minimum=0,
    reparam_offset=(2 ** -18)
)
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





