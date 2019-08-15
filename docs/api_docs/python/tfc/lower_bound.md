<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.lower_bound" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.lower_bound


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/math_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Same as `tf.maximum`, but with helpful gradient for `inputs < bound`.

### Aliases:

* `tfc.python.ops.math_ops.lower_bound`


``` python
tfc.lower_bound(
    inputs,
    bound,
    gradient='identity_if_towards',
    name=None
)
```



<!-- Placeholder for "Used in" -->

This function behaves just like `tf.maximum`, but the behavior of the gradient
with respect to `inputs` for input values that hit the bound depends on
`gradient`:

If set to `'disconnected'`, the returned gradient is zero for values that hit
the bound. This is identical to the behavior of `tf.maximum`.

If set to `'identity'`, the gradient is unconditionally replaced with the
identity function (i.e., pretending this function does not exist).

If set to `'identity_if_towards'`, the gradient is replaced with the identity
function, but only if applying gradient descent would push the values of
`inputs` towards the bound. For gradient values that push away from the bound,
the returned gradient is still zero.

Note: In the latter two cases, no gradient is returned for `bound`.
Also, the implementation of `gradient == 'identity_if_towards'` currently
assumes that the shape of `inputs` is the same as the shape of the output. It
won't work reliably for all possible broadcasting scenarios.

#### Args:


* <b>`inputs`</b>: Input tensor.
* <b>`bound`</b>: Lower bound for the input tensor.
* <b>`gradient`</b>: 'disconnected', 'identity', or 'identity_if_towards' (default).
* <b>`name`</b>: Name for this op.


#### Returns:

`tf.maximum(inputs, bound)`



#### Raises:


* <b>`ValueError`</b>: for invalid value of `gradient`.