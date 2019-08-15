<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.NonnegativeParameterizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfc.NonnegativeParameterizer


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameterizers.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `NonnegativeParameterizer`

Object encapsulating nonnegative parameterization as needed for GDN.

Inherits From: [`Parameterizer`](../tfc/Parameterizer.md)

### Aliases:

* Class `tfc.python.layers.parameterizers.NonnegativeParameterizer`


<!-- Placeholder for "Used in" -->

The variable is subjected to an invertible transformation that slows down the
learning rate for small values.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameterizers.py">View source</a>

``` python
__init__(
    minimum=0,
    reparam_offset=(2 ** -18)
)
```

Initializer.


#### Arguments:


* <b>`minimum`</b>: Float. Lower bound for parameters (defaults to zero).
* <b>`reparam_offset`</b>: Float. Offset added to the reparameterization of beta and
  gamma. The parameterization of beta and gamma as their square roots lets
  the training slow down when their values are close to zero, which is
  desirable as small values in the denominator can lead to a situation
  where gradient noise on beta/gamma leads to extreme amounts of noise in
  the GDN activations. However, without the offset, we would get zero
  gradients if any elements of beta or gamma were exactly zero, and thus
  the training could get stuck. To prevent this, we add this small
  constant. The default value was empirically determined as a good
  starting point. Making it bigger potentially leads to more gradient
  noise on the activations, making it too small may lead to numerical
  precision issues.



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




