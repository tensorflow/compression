<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.Parameterizer" />
</div>

# tfc.Parameterizer

## Class `Parameterizer`



Parameterizer object (abstract base class).

Parameterizer objects are immutable objects designed to facilitate
reparameterization of model parameters (tensor variables). They are called
just like `tf.get_variable` with an additional argument `getter` specifying
the actual function call to generate a variable (in many cases, `getter` would
be `tf.get_variable`).

To achieve reparameterization, a parameterizer object wraps the provided
initializer, regularizer, and the returned variable in its own Tensorflow
code.

