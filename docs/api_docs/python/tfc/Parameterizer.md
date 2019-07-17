<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.Parameterizer" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.Parameterizer

## Class `Parameterizer`

Parameterization object (abstract base class).



### Aliases:

* Class `tfc.Parameterizer`
* Class `tfc.python.layers.parameterizers.Parameterizer`




<table class="tfo-github-link" align="left">
<a target="_blank" href=https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameterizers.py>
  <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
  View source on GitHub
</a>
</table>

<!-- Placeholder for "Used in" -->

`Parameterizer`s are immutable objects designed to facilitate
reparameterization of model parameters (tensor variables). They are called
just like `tf.get_variable` with an additional argument `getter` specifying
the actual function call to generate a variable (in many cases, `getter` would
be `tf.get_variable`).

To achieve reparameterization, a `Parameterizer` wraps the provided
initializer, regularizer, and the returned variable in its own TensorFlow
code.

