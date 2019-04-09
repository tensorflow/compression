
# tfc.Parameterizer

## Class `Parameterizer`



### Aliases:

* Class `tfc.Parameterizer`
* Class `tfc.python.layers.parameterizers.Parameterizer`



Defined in [`python/layers/parameterizers.py`](https://github.com/tensorflow/compression/tree/master/python/layers/parameterizers.py).

<!-- Placeholder for "Used in" -->

Parameterization object (abstract base class).

`Parameterizer`s are immutable objects designed to facilitate
reparameterization of model parameters (tensor variables). They are called
just like `tf.get_variable` with an additional argument `getter` specifying
the actual function call to generate a variable (in many cases, `getter` would
be `tf.get_variable`).

To achieve reparameterization, a `Parameterizer` wraps the provided
initializer, regularizer, and the returned variable in its own TensorFlow
code.

