description: Batched entropy model for continuous random variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.ContinuousBatchedEntropyModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="compress"/>
<meta itemprop="property" content="decompress"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="quantize"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfc.ContinuousBatchedEntropyModel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/entropy_models/continuous_batched.py#L31-L391">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Batched entropy model for continuous random variables.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.ContinuousBatchedEntropyModel(
    prior=None, coding_rank=None, compression=False, stateless=False,
    expected_grads=False, tail_mass=(2 ** -8), range_coder_precision=12, dtype=None,
    prior_shape=None, cdf=None, cdf_offset=None, cdf_length=None,
    cdf_max_length=None, non_integer_offsets=True, quantization_offset=None,
    laplace_tail_mass=0
)
</code></pre>



<!-- Placeholder for "Used in" -->

This entropy model handles quantization of a bottleneck tensor and helps with
training of the parameters of the probability distribution modeling the
tensor (a shared "prior" between sender and receiver). It also pre-computes
integer probability tables, which can then be used to compress and decompress
bottleneck tensors reliably across different platforms.

A typical workflow looks like this:

- Train a model using an instance of this entropy model as a bottleneck,
  passing the bottleneck tensor through it. With `training=True`, the model
  computes a differentiable upper bound on the number of bits needed to
  compress the bottleneck tensor.
- For evaluation, get a closer estimate of the number of compressed bits
  using `training=False`.
- Instantiate an entropy model with `compression=True` (and the same
  parameters as during training), and share the model between a sender and a
  receiver.
- On the sender side, compute the bottleneck tensor and call `compress()` on
  it. The output is a compressed string representation of the tensor. Transmit
  the string to the receiver, and call `decompress()` there. The output is the
  quantized bottleneck tensor. Continue processing the tensor on the receiving
  side.

Entropy models which contain range coding tables (i.e. with
`compression=True`) can be instantiated in three ways:

- By providing a continuous "prior" distribution object. The range coding
  tables are then derived from that continuous distribution.
- From a config as returned by `get_config`, followed by a call to
  `set_weights`. This implements the Keras serialization protocol. In this
  case, the initializer creates empty state variables for the range coding
  tables, which are then filled by `set_weights`. As a consequence, this
  method requires `stateless=False`.
- In a more low-level way, by directly providing the range coding tables to
  `__init__`, for use cases where the Keras protocol can't be used (e.g., when
  the entropy model must not create variables).

This class assumes that all scalar elements of the encoded tensor are
statistically independent, and that the parameters of their scalar
distributions do not depend on data. The innermost dimensions of the
bottleneck tensor must be broadcastable to the batch shape of `prior`. Any
dimensions to the left of the batch shape are assumed to be i.i.d., i.e. the
likelihoods are broadcast to the bottleneck tensor accordingly.

A more detailed description (and motivation) of this way of performing
quantization and range coding can be found in the following paper. Please cite
the paper when using this code for derivative work.

> "End-to-end Optimized Image Compression"<br />
> J. Ballé, V. Laparra, E.P. Simoncelli<br />
> https://openreview.net/forum?id=rJxdQ3jeg

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`prior`
</td>
<td>
A `tfp.distributions.Distribution` object. A density model fitting
the marginal distribution of the bottleneck data with additive uniform
noise, which is shared a priori between the sender and the receiver. For
best results, the distribution should be flexible enough to have a
unit-width uniform distribution as a special case, since this is the
marginal distribution for bottleneck dimensions that are constant. The
distribution parameters may not depend on data (they must be either
variables or constants).
</td>
</tr><tr>
<td>
`coding_rank`
</td>
<td>
Integer. Number of innermost dimensions considered a coding
unit. Each coding unit is compressed to its own bit string, and the
bits in the __call__ method are summed over each coding unit.
</td>
</tr><tr>
<td>
`compression`
</td>
<td>
Boolean. If set to `True`, the range coding tables used by
`compress()` and `decompress()` will be built on instantiation. If set
to `False`, these two methods will not be accessible.
</td>
</tr><tr>
<td>
`stateless`
</td>
<td>
Boolean. If `False`, range coding tables are created as
`Variable`s. This allows the entropy model to be serialized using the
`SavedModel` protocol, so that both the encoder and the decoder use
identical tables when loading the stored model. If `True`, creates range
coding tables as `Tensor`s. This makes the entropy model stateless and
allows it to be constructed within a `tf.function` body, for when the
range coding tables are provided manually. If `compression=False`, then
`stateless=True` is implied and the provided value is ignored.
</td>
</tr><tr>
<td>
`expected_grads`
</td>
<td>
If True, will use analytical expected gradients during
backpropagation w.r.t. additive uniform noise.
</td>
</tr><tr>
<td>
`tail_mass`
</td>
<td>
Float. Approximate probability mass which is range encoded with
less precision, by using a Golomb-like code.
</td>
</tr><tr>
<td>
`range_coder_precision`
</td>
<td>
Integer. Precision passed to the range coding op.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Data type of prior. Must be provided when `prior` is omitted.
</td>
</tr><tr>
<td>
`prior_shape`
</td>
<td>
Batch shape of the prior (dimensions which are not assumed
i.i.d.). Must be provided when `prior` is omitted.
</td>
</tr><tr>
<td>
`cdf`
</td>
<td>
`tf.Tensor` or `None`. When provided, is used for range coding rather
than tables built from the prior.
</td>
</tr><tr>
<td>
`cdf_offset`
</td>
<td>
`tf.Tensor` or `None`. Must be provided along with `cdf`.
</td>
</tr><tr>
<td>
`cdf_length`
</td>
<td>
`tf.Tensor` or `None`. Must be provided along with `cdf`.
</td>
</tr><tr>
<td>
`cdf_max_length`
</td>
<td>
Maximum `cdf_length`. When provided, an empty range coding
table is created, which can then be restored using `set_weights`.
Requires `compression=True` and `stateless=False`.
</td>
</tr><tr>
<td>
`non_integer_offsets`
</td>
<td>
Boolean. Whether to quantize to non-integer offsets
heuristically determined from mode/median of prior. Set to `False` when
using soft quantization during training.
</td>
</tr><tr>
<td>
`quantization_offset`
</td>
<td>
`tf.Tensor` or `None`. If `cdf` is provided and
`non_integer_offsets=True`, must be provided.
</td>
</tr><tr>
<td>
`laplace_tail_mass`
</td>
<td>
Float. If positive, will augment the prior with a
Laplace mixture for training stability. (experimental)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
when attempting to instantiate an entropy model with
`compression=True` and not in eager execution mode.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`cdf`
</td>
<td>

</td>
</tr><tr>
<td>
`cdf_length`
</td>
<td>

</td>
</tr><tr>
<td>
`cdf_offset`
</td>
<td>

</td>
</tr><tr>
<td>
`coding_rank`
</td>
<td>
Number of innermost dimensions considered a coding unit.
</td>
</tr><tr>
<td>
`compression`
</td>
<td>
Whether this entropy model is prepared for compression.
</td>
</tr><tr>
<td>
`context_shape`
</td>
<td>
The shape of the non-flattened PDF/CDF tables for range coding.

This is typically the same as the prior shape, but can differ e.g. in
universal entropy models. In any case, the context_shape contains the prior
shape (in the trailing dimensions).
</td>
</tr><tr>
<td>
`context_shape_tensor`
</td>
<td>
The context shape as a `Tensor`.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Data type of this entropy model.
</td>
</tr><tr>
<td>
`expected_grads`
</td>
<td>
Whether to use analytical expected gradients during backpropagation.
</td>
</tr><tr>
<td>
`laplace_tail_mass`
</td>
<td>
Whether to augment the prior with a Laplace mixture.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.
</td>
</tr><tr>
<td>
`name_scope`
</td>
<td>
Returns a `tf.name_scope` instance for this class.
</td>
</tr><tr>
<td>
`non_integer_offsets`
</td>
<td>

</td>
</tr><tr>
<td>
`non_trainable_variables`
</td>
<td>
Sequence of non-trainable variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr><tr>
<td>
`prior`
</td>
<td>
Prior distribution, used for deriving range coding tables.
</td>
</tr><tr>
<td>
`prior_shape`
</td>
<td>
Batch shape of `prior` (dimensions which are not assumed i.i.d.).
</td>
</tr><tr>
<td>
`prior_shape_tensor`
</td>
<td>
Batch shape of `prior` as a `Tensor`.
</td>
</tr><tr>
<td>
`quantization_offset`
</td>
<td>

</td>
</tr><tr>
<td>
`range_coder_precision`
</td>
<td>
Precision passed to range coding op.
</td>
</tr><tr>
<td>
`stateless`
</td>
<td>
Whether range coding tables are created as `Tensor`s or `Variable`s.
</td>
</tr><tr>
<td>
`submodules`
</td>
<td>
Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> list(a.submodules) == [b, c]
True
>>> list(b.submodules) == [c]
True
>>> list(c.submodules) == []
True
```
</td>
</tr><tr>
<td>
`tail_mass`
</td>
<td>
Approximate probability mass which is range encoded with overflow.
</td>
</tr><tr>
<td>
`trainable_variables`
</td>
<td>
Sequence of trainable variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr><tr>
<td>
`variables`
</td>
<td>
Sequence of variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr>
</table>



## Methods

<h3 id="compress"><code>compress</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/entropy_models/continuous_batched.py#L283-L334">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compress(
    bottleneck
)
</code></pre>

Compresses a floating-point tensor.

Compresses the tensor to bit strings. `bottleneck` is first quantized
as in `quantize()`, and then compressed using the probability tables in
`self.cdf` derived from `self.prior`. The quantized tensor can later be
recovered by calling `decompress()`.

The innermost `self.coding_rank` dimensions are treated as one coding unit,
i.e. are compressed into one string each. Any additional dimensions to the
left are treated as batch dimensions.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`bottleneck`
</td>
<td>
`tf.Tensor` containing the data to be compressed. Must have at
least `self.coding_rank` dimensions, and the innermost dimensions must
be broadcastable to `self.prior_shape`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.Tensor` having the same shape as `bottleneck` without the
`self.coding_rank` innermost dimensions, containing a string for each
coding unit.
</td>
</tr>

</table>



<h3 id="decompress"><code>decompress</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/entropy_models/continuous_batched.py#L336-L379">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompress(
    strings, broadcast_shape
)
</code></pre>

Decompresses a tensor.

Reconstructs the quantized tensor from bit strings produced by `compress()`.
It is necessary to provide a part of the output shape in `broadcast_shape`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`strings`
</td>
<td>
`tf.Tensor` containing the compressed bit strings.
</td>
</tr><tr>
<td>
`broadcast_shape`
</td>
<td>
Iterable of ints. The part of the output tensor shape
between the shape of `strings` on the left and `self.prior_shape` on the
right. This must match the shape of the input to `compress()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.Tensor` of shape `strings.shape + broadcast_shape +
self.prior_shape`.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/entropy_models/continuous_batched.py#L381-L391">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the configuration of the entropy model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A JSON-serializable Python dict.
</td>
</tr>

</table>



<h3 id="get_weights"><code>get_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/entropy_models/continuous_base.py#L410-L411">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_weights()
</code></pre>




<h3 id="quantize"><code>quantize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/entropy_models/continuous_batched.py#L262-L281">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>quantize(
    bottleneck
)
</code></pre>

Quantizes a floating-point bottleneck tensor.

The tensor is rounded to integer values potentially shifted by offsets (if
`self.non_integer_offsets==True`). These offsets depend on `self.prior`. For
instance, for a Gaussian distribution, the returned values would be rounded
to the location of the mode of the distribution plus or minus an integer.

The gradient of this rounding operation is overridden with the identity
(straight-through gradient estimator).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`bottleneck`
</td>
<td>
`tf.Tensor` containing the data to be quantized. The innermost
dimensions must be broadcastable to `self.prior_shape`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.Tensor` containing the quantized values.
</td>
</tr>

</table>



<h3 id="set_weights"><code>set_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/entropy_models/continuous_base.py#L413-L418">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_weights(
    weights
)
</code></pre>




<h3 id="with_name_scope"><code>with_name_scope</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>with_name_scope(
    method
)
</code></pre>

Decorator to automatically enter the module name scope.

```
>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
...     return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
>>> mod = MyModule()
>>> mod(tf.ones([1, 2]))
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
>>> mod.w
<tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
numpy=..., dtype=float32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`method`
</td>
<td>
The method to wrap.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The original method wrapped such that it enters the module's name scope.
</td>
</tr>

</table>



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/entropy_models/continuous_batched.py#L229-L260">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    bottleneck, training=True
)
</code></pre>

Perturbs a tensor with (quantization) noise and estimates rate.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`bottleneck`
</td>
<td>
`tf.Tensor` containing the data to be compressed. Must have at
least `self.coding_rank` dimensions, and the innermost dimensions must
be broadcastable to `self.prior_shape`.
</td>
</tr><tr>
<td>
`training`
</td>
<td>
Boolean. If `False`, computes the Shannon information of
`bottleneck` under the distribution `self.prior`, which is a
non-differentiable, tight *lower* bound on the number of bits needed to
compress `bottleneck` using `compress()`. If `True`, returns a somewhat
looser, but differentiable *upper* bound on this quantity.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple (bottleneck_perturbed, bits) where `bottleneck_perturbed` is
`bottleneck` perturbed with (quantization) noise, and `bits` is the rate.
`bits` has the same shape as `bottleneck` without the `self.coding_rank`
innermost dimensions.
</td>
</tr>

</table>





