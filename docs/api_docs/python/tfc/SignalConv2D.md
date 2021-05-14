description: 2D convolution layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.SignalConv2D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="add_loss"/>
<meta itemprop="property" content="add_metric"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfc.SignalConv2D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/signal_conv.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



2D convolution layer.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.SignalConv2D(
    filters, kernel_support, corr=False, strides_down=1, strides_up=1,
    padding=&#x27;valid&#x27;, extra_pad_end=None, channel_separable=False,
    data_format=&#x27;channels_last&#x27;, activation=None, use_bias=False,
    use_explicit=True, kernel_parameter=&#x27;rdft&#x27;,
    bias_parameter=&#x27;variable&#x27;,
    kernel_initializer=&#x27;variance_scaling&#x27;,
    bias_initializer=&#x27;zeros&#x27;, kernel_regularizer=None,
    bias_regularizer=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer creates a filter kernel that is convolved or cross correlated with
the layer input to produce an output tensor. The main difference of this class
to `tf.layers.Conv2D` is how padding, up- and downsampling, and alignment
is handled. It supports much more flexible options for structuring the linear
transform.

In general, the outputs are equivalent to a composition of:
1. an upsampling step (if `strides_up > 1`)
2. a convolution or cross correlation
3. a downsampling step (if `strides_down > 1`)
4. addition of a bias vector (if `use_bias == True`)
5. a pointwise nonlinearity (if `activation is not None`)

For more information on what the difference between convolution and cross
correlation is, see [this](https://en.wikipedia.org/wiki/Convolution) and
[this](https://en.wikipedia.org/wiki/Cross-correlation) Wikipedia article,
respectively. Note that the distinction between convolution and cross
correlation is occasionally blurred (one may use convolution as an umbrella
term for both). For a discussion of up-/downsampling, refer to the articles
about [upsampling](https://en.wikipedia.org/wiki/Upsampling) and
[decimation](https://en.wikipedia.org/wiki/Decimation_(signal_processing)). A
more in-depth treatment of all of these operations can be found in:

> "Discrete-Time Signal Processing"<br />
> Oppenheim, Schafer, Buck (Prentice Hall)

For purposes of this class, the center position of a kernel is always
considered to be at `K // 2`, where `K` is the support length of the kernel.
This implies that in the `'same_*'` padding modes, all of the following
operations will produce the same result if applied to the same inputs, which
is not generally true for convolution operations as implemented by
`tf.nn.convolution` or `tf.layers.Conv?D` (numbers represent kernel
coefficient values):

- convolve with `[1, 2, 3]`
- convolve with `[0, 1, 2, 3, 0]`
- convolve with `[0, 1, 2, 3]`
- correlate with `[3, 2, 1]`
- correlate with `[0, 3, 2, 1, 0]`
- correlate with `[0, 3, 2, 1]`

Available padding (boundary handling) modes:

- `'valid'`: This always yields the maximum number of output samples that can
  be computed without making any assumptions about the values outside of the
  support of the input tensor. The padding semantics are always applied to the
  inputs. In contrast, even though `tf.nn.conv2d_transpose` implements
  upsampling, in `'VALID'` mode it will produce an output tensor with *larger*
  support than the input tensor (because it is the transpose of a `'VALID'`
  downsampled convolution).

  Examples (numbers represent indexes into the respective tensors, periods
  represent skipped spatial positions):

  `kernel_support = 5` and `strides_down = 2`:
  ```
  inputs:  |0 1 2 3 4 5 6 7 8|
  outputs: |    0 . 1 . 2    |
  ```
  ```
  inputs:  |0 1 2 3 4 5 6 7|
  outputs: |    0 . 1 .    |
  ```

  `kernel_support = 3`, `strides_up = 2`, and `extra_pad_end = True`:
  ```
  inputs:   |0 . 1 . 2 . 3 . 4 .|
  outputs:  |  0 1 2 3 4 5 6 7  |
  ```

  `kernel_support = 3`, `strides_up = 2`, and `extra_pad_end = False`:
  ```
  inputs:   |0 . 1 . 2 . 3 . 4|
  outputs:  |  0 1 2 3 4 5 6  |
  ```

- `'same_zeros'`: Values outside of the input tensor support are assumed to be
  zero. Similar to `'SAME'` in `tf.nn.convolution`, but with different
  padding. In `'SAME'`, the spatial alignment of the output depends on the
  input shape. Here, the output alignment depends only on the kernel support
  and the strides, making alignment more predictable. The first sample in the
  output is always spatially aligned with the first sample in the input.

  Examples (numbers represent indexes into the respective tensors, periods
  represent skipped spatial positions):

  `kernel_support = 5` and `strides_down = 2`:
  ```
  inputs:  |0 1 2 3 4 5 6 7 8|
  outputs: |0 . 1 . 2 . 3 . 4|
  ```
  ```
  inputs:  |0 1 2 3 4 5 6 7|
  outputs: |0 . 1 . 2 . 3 .|
  ```

  `kernel_support = 3`, `strides_up = 2`, and `extra_pad_end = True`:
  ```
  inputs:   |0 . 1 . 2 . 3 . 4 .|
  outputs:  |0 1 2 3 4 5 6 7 8 9|
  ```

  `kernel_support = 3`, `strides_up = 2`, and `extra_pad_end = False`:
  ```
  inputs:   |0 . 1 . 2 . 3 . 4|
  outputs:  |0 1 2 3 4 5 6 7 8|
  ```

- `'same_reflect'`: Values outside of the input tensor support are assumed to
  be reflections of the samples inside. Note that this is the same padding as
  implemented by `tf.pad` in the `'REFLECT'` mode (i.e. with the symmetry axis
  on the samples rather than between). The output alignment is identical to
  the `'same_zeros'` mode.

  Examples: see `'same_zeros'`.

  When applying several convolutions with down- or upsampling in a sequence,
  it can be helpful to keep the axis of symmetry for the reflections
  consistent. To do this, set `extra_pad_end = False` and make sure that the
  input has length `M`, such that `M % S == 1`, where `S` is the product of
  stride lengths of all subsequent convolutions. Example for subsequent
  downsampling (here, `M = 9`, `S = 4`, and `^` indicate the symmetry axes
  for reflection):

  ```
  inputs:       |0 1 2 3 4 5 6 7 8|
  intermediate: |0 . 1 . 2 . 3 . 4|
  outputs:      |0 . . . 1 . . . 2|
                 ^               ^
  ```

Note that due to limitations of the underlying operations, not all
combinations of arguments are currently implemented. In this case, this class
will throw a `NotImplementedError` exception.

#### Speed tips:



- Prefer combining correlations with downsampling, and convolutions with
  upsampling, as the underlying ops implement these combinations directly.
- If that isn't desirable, prefer using odd-length kernel supports, since
  odd-length kernels can be flipped if necessary, to use the fastest
  implementation available.
- Combining upsampling and downsampling (for rational resampling ratios)
  is relatively slow, because no underlying ops exist for that use case.
  Downsampling in this case is implemented by discarding computed output
  values.
- Note that `channel_separable` is only implemented for 1D and 2D. Also,
  upsampled channel-separable convolutions are currently only implemented for
  `filters == 1`. When using `channel_separable`, prefer using identical
  strides in all dimensions to maximize performance.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`filters`
</td>
<td>
Integer. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`kernel_support`
</td>
<td>
Integer or iterable of integers. Initial value of
eponymous attribute.
</td>
</tr><tr>
<td>
`corr`
</td>
<td>
Boolean. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`strides_down`
</td>
<td>
Integer or iterable of integers. Initial value of eponymous
attribute.
</td>
</tr><tr>
<td>
`strides_up`
</td>
<td>
Integer or iterable of integers. Initial value of eponymous
attribute.
</td>
</tr><tr>
<td>
`padding`
</td>
<td>
String. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`extra_pad_end`
</td>
<td>
Boolean or `None`. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`channel_separable`
</td>
<td>
Boolean. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`data_format`
</td>
<td>
String. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`activation`
</td>
<td>
Callable or `None`. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`use_bias`
</td>
<td>
Boolean. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`use_explicit`
</td>
<td>
Boolean. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`kernel_parameter`
</td>
<td>
Tensor, `tf.Variable`, callable, `'rdft'`, or
`'variable'`. Initial value of eponymous attribute.
</td>
</tr><tr>
<td>
`bias_parameter`
</td>
<td>
Tensor, `tf.Variable`, callable, or `'variable'`. Initial
value of eponymous attribute.
</td>
</tr><tr>
<td>
`kernel_initializer`
</td>
<td>
`Initializer` object. Initial value of eponymous
attribute.
</td>
</tr><tr>
<td>
`bias_initializer`
</td>
<td>
`Initializer` object. Initial value of eponymous
attribute.
</td>
</tr><tr>
<td>
`kernel_regularizer`
</td>
<td>
`Regularizer` object or `None`. Initial value of
eponymous attribute.
</td>
</tr><tr>
<td>
`bias_regularizer`
</td>
<td>
`Regularizer` object or `None`. Initial value of
eponymous attribute.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments passed to superclass (`Layer`).
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`filters`
</td>
<td>
Integer. If `not channel_separable`, specifies the total number of
filters, which is equal to the number of output channels. Otherwise,
specifies the number of filters per channel, which makes the number of
output channels equal to `filters` times the number of input channels.
</td>
</tr><tr>
<td>
`kernel_support`
</td>
<td>
An integer or iterable of 2 integers, specifying the
length of the convolution/correlation window in each dimension.
</td>
</tr><tr>
<td>
`corr`
</td>
<td>
Boolean. If True, compute cross correlation. If False, convolution.
</td>
</tr><tr>
<td>
`strides_down`
</td>
<td>
An integer or iterable of 2 integers, specifying an
optional downsampling stride after the convolution/correlation.
</td>
</tr><tr>
<td>
`strides_up`
</td>
<td>
An integer or iterable of 2 integers, specifying an
optional upsampling stride before the convolution/correlation.
</td>
</tr><tr>
<td>
`padding`
</td>
<td>
String. One of the supported padding modes (see above).
</td>
</tr><tr>
<td>
`extra_pad_end`
</td>
<td>
Boolean or `None`. When upsampling, use extra skipped samples
at the end of each dimension. `None` implies `True` for `same_*` padding
modes, and `False` for `valid`. For examples, refer to the discussion of
padding modes above.
</td>
</tr><tr>
<td>
`channel_separable`
</td>
<td>
Boolean. If `False`, each output channel is computed by
summing over all filtered input channels. If `True`, each output channel
is computed from only one input channel, and `filters` specifies the
number of filters per channel. The output channels are ordered such that
the first block of `filters` channels is computed from the first input
channel, the second block from the second input channel, etc.
</td>
</tr><tr>
<td>
`data_format`
</td>
<td>
String, one of `'channels_last'` or `'channels_first'`. The
ordering of the input dimensions. `'channels_last'` corresponds to input
tensors with shape `(batch, ..., channels)`, while `'channels_first'`
corresponds to input tensors with shape `(batch, channels, ...)`.
</td>
</tr><tr>
<td>
`activation`
</td>
<td>
Activation function or `None`.
</td>
</tr><tr>
<td>
`use_bias`
</td>
<td>
Boolean, whether an additive constant will be applied to each
output channel.
</td>
</tr><tr>
<td>
`use_explicit`
</td>
<td>
Boolean, whether to use `EXPLICIT` padding mode (supported in
TensorFlow >1.14).
</td>
</tr><tr>
<td>
`kernel_parameter`
</td>
<td>
Tensor, `tf.Variable`, callable, or one of the strings
`'rdft'`, `'variable'`. A `tf.Tensor` means that the kernel is fixed, a
`tf.Variable` that it is trained. A callable can be used to determine the
value of the kernel as a function of some other variable or tensor. This
can be a `Parameter` object. `'rdft'` means that when the layer is built,
a `RDFTParameter` object is created to train the kernel. `'variable'`
means that when the layer is built, a `tf.Variable` is created to train
the kernel. Note that certain choices here such as `tf.Tensor`s or lambda
functions may prevent JSON-style serialization (`Parameter` objects and
`tf.Variable`s work).
</td>
</tr><tr>
<td>
`bias_parameter`
</td>
<td>
Tensor, `tf.Variable`, callable, or the string `'variable'`.
A `tf.Tensor` means that the bias is fixed, a `tf.Variable` that it is
trained. A callable can be used to determine the value of the bias as a
function of some other variable or tensor. This can be a `Parameter`
object. `'variable'` means that when the layer is built, a `tf.Variable`
is created to train the bias. Note that certain choices here such as
`tf.Tensor`s or lambda functions may prevent JSON-style serialization
(`Parameter` objects and `tf.Variable`s work).
</td>
</tr><tr>
<td>
`kernel_initializer`
</td>
<td>
`Initializer` object for the filter kernel.
</td>
</tr><tr>
<td>
`bias_initializer`
</td>
<td>
`Initializer` object for the bias vector.
</td>
</tr><tr>
<td>
`kernel_regularizer`
</td>
<td>
`Regularizer` object or `None`. Optional regularizer for
the filter kernel.
</td>
</tr><tr>
<td>
`bias_regularizer`
</td>
<td>
`Regularizer` object or `None`. Optional regularizer for
the bias vector.
</td>
</tr><tr>
<td>
`kernel`
</td>
<td>
`tf.Tensor`. Read-only property always returning the current kernel
tensor.
</td>
</tr><tr>
<td>
`bias`
</td>
<td>
`tf.Tensor`. Read-only property always returning the current bias
tensor.
</td>
</tr><tr>
<td>
`activity_regularizer`
</td>
<td>
Optional regularizer function for the output of this layer.
</td>
</tr><tr>
<td>
`compute_dtype`
</td>
<td>
The dtype of the layer's computations.

This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
mixed precision is used, this is the same as `Layer.dtype`, the dtype of
the weights.

Layers automatically cast their inputs to the compute dtype, which causes
computations and the output to be in the compute dtype as well. This is done
by the base Layer class in `Layer.__call__`, so you do not have to insert
these casts if implementing your own layer.

Layers often perform certain internal computations in higher precision when
`compute_dtype` is float16 or bfloat16 for numeric stability. The output
will still typically be float16 or bfloat16 in such cases.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The dtype of the layer weights.

This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
mixed precision is used, this is the same as `Layer.compute_dtype`, the
dtype of the layer's computations.
</td>
</tr><tr>
<td>
`dtype_policy`
</td>
<td>
The dtype policy associated with this layer.

This is an instance of a `tf.keras.mixed_precision.Policy`.
</td>
</tr><tr>
<td>
`dynamic`
</td>
<td>
Whether the layer is dynamic (eager-only); set in the constructor.
</td>
</tr><tr>
<td>
`input`
</td>
<td>
Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.
</td>
</tr><tr>
<td>
`input_spec`
</td>
<td>
`InputSpec` instance(s) describing the input format for this layer.

When you create a layer subclass, you can set `self.input_spec` to enable
the layer to run input compatibility checks when it is called.
Consider a `Conv2D` layer: it can only be called on a single input tensor
of rank 4. As such, you can set, in `__init__()`:

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
```

Now, if you try to call the layer on an input that isn't rank 4
(for instance, an input of shape `(2,)`, it will raise a nicely-formatted
error:

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
```

Input checks that can be specified via `input_spec` include:
- Structure (e.g. a single input, a list of 2 inputs, etc)
- Shape
- Rank (ndim)
- Dtype

For more information, see `tf.keras.layers.InputSpec`.
</td>
</tr><tr>
<td>
`losses`
</td>
<td>
List of losses added using the `add_loss()` API.

Variable regularization tensors are created when this property is accessed,
so it is eager safe: accessing `losses` under a `tf.GradientTape` will
propagate gradients back to the corresponding variables.

```
>>> class MyLayer(tf.keras.layers.Layer):
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
>>> l = MyLayer()
>>> l(np.ones((10, 1)))
>>> l.losses
[1.0]
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> x = tf.keras.layers.Dense(10)(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Activity regularization.
>>> len(model.losses)
0
>>> model.add_loss(tf.abs(tf.reduce_mean(x)))
>>> len(model.losses)
1
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
>>> x = d(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Weight regularization.
>>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
>>> model.losses
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]
```
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
List of metrics added using the `add_metric()` API.


```
>>> input = tf.keras.layers.Input(shape=(3,))
>>> d = tf.keras.layers.Dense(2)
>>> output = d(input)
>>> d.add_metric(tf.reduce_max(output), name='max')
>>> d.add_metric(tf.reduce_min(output), name='min')
>>> [m.name for m in d.metrics]
['max', 'min']
```
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Name of the layer (string), set in the constructor.
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
`non_trainable_weights`
</td>
<td>
List of all non-trainable weights tracked by this layer.

Non-trainable weights are *not* updated during training. They are expected
to be updated manually in `call()`.
</td>
</tr><tr>
<td>
`output`
</td>
<td>
Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output,
i.e. if it is connected to one incoming layer.
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
`supports_masking`
</td>
<td>
Whether this layer supports computing a mask using `compute_mask`.
</td>
</tr><tr>
<td>
`trainable`
</td>
<td>

</td>
</tr><tr>
<td>
`trainable_weights`
</td>
<td>
List of all trainable weights tracked by this layer.

Trainable weights are updated via gradient descent during training.
</td>
</tr><tr>
<td>
`variable_dtype`
</td>
<td>
Alias of `Layer.dtype`, the dtype of the weights.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
Returns the list of all layer variables/weights.
</td>
</tr>
</table>



## Methods

<h3 id="add_loss"><code>add_loss</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_loss(
    losses, **kwargs
)
</code></pre>

Add loss tensor(s), potentially dependent on layer inputs.

Some losses (for instance, activity regularization losses) may be dependent
on the inputs passed when calling a layer. Hence, when reusing the same
layer on different inputs `a` and `b`, some entries in `layer.losses` may
be dependent on `a` and some on `b`. This method automatically keeps track
of dependencies.

This method can be used inside a subclassed layer or model's `call`
function, in which case `losses` should be a Tensor or list of Tensors.

#### Example:



```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any loss Tensors passed to this Model must
be symbolic and be able to be traced back to the model's `Input`s. These
losses become part of the model's topology and are tracked in `get_config`.

#### Example:



```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Activity regularization.
model.add_loss(tf.abs(tf.reduce_mean(x)))
```

If this is not the case for your loss (if, for example, your loss references
a `Variable` of one of the model's layers), you can wrap your loss in a
zero-argument lambda. These losses are not tracked as part of the model's
topology since they can't be serialized.

#### Example:



```python
inputs = tf.keras.Input(shape=(10,))
d = tf.keras.layers.Dense(10)
x = d(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Weight regularization.
model.add_loss(lambda: tf.reduce_mean(d.kernel))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`losses`
</td>
<td>
Loss tensor, or list/tuple of tensors. Rather than tensors, losses
may also be zero-argument callables which create a loss tensor.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments for backward compatibility.
Accepted values:
inputs - Deprecated, will be automatically inferred.
</td>
</tr>
</table>



<h3 id="add_metric"><code>add_metric</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_metric(
    value, name=None, **kwargs
)
</code></pre>

Adds metric tensor to the layer.

This method can be used inside the `call()` method of a subclassed layer
or model.

```python
class MyMetricLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

  def call(self, inputs):
    self.add_metric(self.mean(inputs))
    self.add_metric(tf.reduce_sum(inputs), name='metric_2')
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any tensor passed to this Model must
be symbolic and be able to be traced back to the model's `Input`s. These
metrics become part of the model's topology and are tracked when you
save the model via `save()`.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(math_ops.reduce_sum(x), name='metric_1')
```

Note: Calling `add_metric()` with the result of a metric object on a
Functional Model, as shown in the example below, is not supported. This is
because we cannot trace the metric result tensor back to the model's inputs.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
Metric tensor.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String metric name.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments for backward compatibility.
Accepted values:
`aggregation` - When the `value` tensor provided is not the result of
calling a `keras.Metric` instance, it will be aggregated by default
using a `keras.Metric.Mean`.
</td>
</tr>
</table>



<h3 id="build"><code>build</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/signal_conv.py#L574-L611">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build(
    input_shape
)
</code></pre>

Creates the variables of the layer (optional, for subclass implementers).

This is a method that implementers of subclasses of `Layer` or `Model`
can override if they need a state-creation step in-between
layer instantiation and layer call.

This is typically used to create the weights of `Layer` subclasses.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_shape`
</td>
<td>
Instance of `TensorShape`, or list of instances of
`TensorShape` if the layer expects a list of inputs
(one instance per input).
</td>
</tr>
</table>



<h3 id="compute_mask"><code>compute_mask</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_mask(
    inputs, mask=None
)
</code></pre>

Computes an output mask tensor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Tensor or list of tensors.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
Tensor or list of tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None or a tensor (or list of tensors,
one per output tensor of the layer).
</td>
</tr>

</table>



<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/signal_conv.py#L945-L974">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_output_shape(
    input_shape
) -> tf.TensorShape
</code></pre>

Computes the output shape of the layer.

If the layer has not been built, this method will call `build` on the
layer. This assumes that the layer will later be used with inputs that
match the input shape provided here.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_shape`
</td>
<td>
Shape tuple (tuple of integers)
or list of shape tuples (one per output tensor of the layer).
Shape tuples can include None for free dimensions,
instead of an integer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An input shape tuple.
</td>
</tr>

</table>



<h3 id="count_params"><code>count_params</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>count_params()
</code></pre>

Count the total number of scalars composing the weights.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An integer count.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the layer isn't yet built
(in which case its weights aren't yet defined).
</td>
</tr>
</table>



<h3 id="from_config"><code>from_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Creates a layer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same layer from the config
dictionary. It does not handle layer connectivity
(handled by Network), nor weights (handled by `set_weights`).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
A Python dictionary, typically the
output of get_config.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A layer instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/signal_conv.py#L976-L1019">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>

Returns the config of the layer.

A layer config is a Python dictionary (serializable)
containing the configuration of a layer.
The same layer can be reinstantiated later
(without its trained weights) from this configuration.

The config of a layer does not include connectivity
information, nor the layer class name. These are handled
by `Network` (one layer of abstraction above).

Note that `get_config()` does not guarantee to return a fresh copy of dict
every time it is called. The callers should make a copy of the returned dict
if they want to modify it.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Python dictionary.
</td>
</tr>

</table>



<h3 id="get_weights"><code>get_weights</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_weights()
</code></pre>

Returns the current weights of the layer, as NumPy arrays.

The weights of a layer represent the state of the layer. This function
returns both trainable and non-trainable weight values associated with this
layer as a list of NumPy arrays, which can in turn be used to load state
into similarly parameterized layers.

For example, a `Dense` layer returns a list of two values: the kernel matrix
and the bias vector. These can be used to set the weights of another
`Dense` layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Weights values as a list of NumPy arrays.
</td>
</tr>

</table>



<h3 id="set_weights"><code>set_weights</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_weights(
    weights
)
</code></pre>

Sets the weights of the layer, from NumPy arrays.

The weights of a layer represent the state of the layer. This function
sets the weight values from numpy arrays. The weight values should be
passed in the order they are created by the layer. Note that the layer's
weights must be instantiated before calling this function, by calling
the layer.

For example, a `Dense` layer returns a list of two values: the kernel matrix
and the bias vector. These can be used to set the weights of another
`Dense` layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`weights`
</td>
<td>
a list of NumPy arrays. The number
of arrays and their shape must match
number of the dimensions of the weights
of the layer (i.e. it should match the
output of `get_weights`).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the provided weights list does not match the
layer's specifications.
</td>
</tr>
</table>



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

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>

Wraps `call`, applying pre- and post-processing steps.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
Positional arguments to be passed to `self.call`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to be passed to `self.call`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Output tensor(s).
</td>
</tr>

</table>



#### Note:

- The following optional keyword arguments are reserved for specific uses:
  * `training`: Boolean scalar tensor of Python boolean indicating
    whether the `call` is meant for training or inference.
  * `mask`: Boolean input mask.
- If the layer's `call` method takes a `mask` argument (as some Keras
  layers do), its default value will be set to the mask generated
  for `inputs` by the previous layer (if `input` did come from
  a layer that generated a corresponding mask, i.e. if it came from
  a Keras layer with masking support.
- If the layer is not built, the method will call `build`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the layer's `call` method returns None (an invalid value).
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
if `super().__init__()` was not called in the constructor.
</td>
</tr>
</table>





