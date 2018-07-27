<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.upper_bound" />
</div>

# tfc.upper_bound

``` python
tfc.upper_bound(
    inputs,
    bound,
    gradient='identity_if_towards',
    name=None
)
```

Same as `tf.minimum`, but with helpful gradient for `inputs > bound`.

This function behaves just like `tf.minimum`, but the behavior of the gradient
with respect to `inputs` for input values that hit the bound depends on
`gradient`:

If set to `'disconnected'`, the returned gradient is zero for values that hit
the bound. This is identical to the behavior of `tf.minimum`.

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
* <b>`bound`</b>: Upper bound for the input tensor.
* <b>`gradient`</b>: 'disconnected', 'identity', or 'identity_if_towards' (default).
* <b>`name`</b>: Name for this op.


#### Returns:

`tf.minimum(inputs, bound)`


#### Raises:

* <b>`ValueError`</b>: for invalid value of `gradient`.