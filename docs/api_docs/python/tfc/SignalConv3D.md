<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.SignalConv3D" />
<meta itemprop="property" content="activation"/>
<meta itemprop="property" content="activity_regularizer"/>
<meta itemprop="property" content="bias"/>
<meta itemprop="property" content="bias_initializer"/>
<meta itemprop="property" content="bias_parameterizer"/>
<meta itemprop="property" content="bias_regularizer"/>
<meta itemprop="property" content="channel_separable"/>
<meta itemprop="property" content="corr"/>
<meta itemprop="property" content="data_format"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="extra_pad_end"/>
<meta itemprop="property" content="filters"/>
<meta itemprop="property" content="graph"/>
<meta itemprop="property" content="inbound_nodes"/>
<meta itemprop="property" content="input"/>
<meta itemprop="property" content="input_mask"/>
<meta itemprop="property" content="input_shape"/>
<meta itemprop="property" content="kernel"/>
<meta itemprop="property" content="kernel_initializer"/>
<meta itemprop="property" content="kernel_parameterizer"/>
<meta itemprop="property" content="kernel_regularizer"/>
<meta itemprop="property" content="kernel_support"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="outbound_nodes"/>
<meta itemprop="property" content="output"/>
<meta itemprop="property" content="output_mask"/>
<meta itemprop="property" content="output_shape"/>
<meta itemprop="property" content="padding"/>
<meta itemprop="property" content="scope_name"/>
<meta itemprop="property" content="strides_down"/>
<meta itemprop="property" content="strides_up"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="updates"/>
<meta itemprop="property" content="use_bias"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__deepcopy__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_loss"/>
<meta itemprop="property" content="add_update"/>
<meta itemprop="property" content="add_variable"/>
<meta itemprop="property" content="add_weight"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_input_at"/>
<meta itemprop="property" content="get_input_mask_at"/>
<meta itemprop="property" content="get_input_shape_at"/>
<meta itemprop="property" content="get_losses_for"/>
<meta itemprop="property" content="get_output_at"/>
<meta itemprop="property" content="get_output_mask_at"/>
<meta itemprop="property" content="get_output_shape_at"/>
<meta itemprop="property" content="get_updates_for"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="set_weights"/>
</div>

# tfc.SignalConv3D

## Class `SignalConv3D`



3D convolution layer.

This layer creates a filter kernel that is convolved or cross correlated with
the layer input to produce an output tensor. The main difference of this class
to `tf.layers.Conv?D` is how padding, up- and downsampling, and alignment is
handled.

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
will throw an exception.

#### Arguments:

* <b>`filters`</b>: Integer. If `not channel_separable`, specifies the total number of
    filters, which is equal to the number of output channels. Otherwise,
    specifies the number of filters per channel, which makes the number of
    output channels equal to `filters` times the number of input channels.
* <b>`kernel_support`</b>: An integer or iterable of 3 integers, specifying the
    length of the convolution/correlation window in each dimension.
* <b>`corr`</b>: Boolean. If True, compute cross correlation. If False, convolution.
* <b>`strides_down`</b>: An integer or iterable of 3 integers, specifying an
    optional downsampling stride after the convolution/correlation.
* <b>`strides_up`</b>: An integer or iterable of 3 integers, specifying an
    optional upsampling stride before the convolution/correlation.
* <b>`padding`</b>: String. One of the supported padding modes (see above).
* <b>`extra_pad_end`</b>: Boolean. When upsampling, use extra skipped samples at the
    end of each dimension (default). For examples, refer to the discussion
    of padding modes above.
* <b>`channel_separable`</b>: Boolean. If `False` (default), each output channel is
    computed by summing over all filtered input channels. If `True`, each
    output channel is computed from only one input channel, and `filters`
    specifies the number of filters per channel. The output channels are
    ordered such that the first block of `filters` channels is computed from
    the first input channel, the second block from the second input channel,
    etc.
* <b>`data_format`</b>: String, one of `channels_last` (default) or `channels_first`.
    The ordering of the input dimensions. `channels_last` corresponds to
    input tensors with shape `(batch, ..., channels)`, while `channels_first`
    corresponds to input tensors with shape `(batch, channels, ...)`.
* <b>`activation`</b>: Activation function or `None`.
* <b>`use_bias`</b>: Boolean, whether an additive constant will be applied to each
    output channel.
* <b>`kernel_initializer`</b>: An initializer for the filter kernel.
* <b>`bias_initializer`</b>: An initializer for the bias vector.
* <b>`kernel_regularizer`</b>: Optional regularizer for the filter kernel.
* <b>`bias_regularizer`</b>: Optional regularizer for the bias vector.
* <b>`activity_regularizer`</b>: Regularizer function for the output.
* <b>`kernel_parameterizer`</b>: Reparameterization applied to filter kernel. If not
    `None`, must be a `Parameterizer` object. Defaults to `RDFTParameterizer`.
* <b>`bias_parameterizer`</b>: Reparameterization applied to bias. If not `None`, must
    be a `Parameterizer` object.
* <b>`trainable`</b>: Boolean. Whether the layer should be trained.
* <b>`name`</b>: String. The name of the layer.
* <b>`dtype`</b>: Default dtype of the layer's parameters (default of `None` means use
    the type of the first input).

Read-only properties:
* <b>`filters`</b>: See above.
* <b>`kernel_support`</b>: See above.
* <b>`corr`</b>: See above.
* <b>`strides_down`</b>: See above.
* <b>`strides_up`</b>: See above.
* <b>`padding`</b>: See above.
* <b>`extra_pad_end`</b>: See above.
* <b>`channel_separable`</b>: See above.
* <b>`data_format`</b>: See above.
* <b>`activation`</b>: See above.
* <b>`use_bias`</b>: See above.
* <b>`kernel_initializer`</b>: See above.
* <b>`bias_initializer`</b>: See above.
* <b>`kernel_regularizer`</b>: See above.
* <b>`bias_regularizer`</b>: See above.
* <b>`activity_regularizer`</b>: See above.
* <b>`kernel_parameterizer`</b>: See above.
* <b>`bias_parameterizer`</b>: See above.
* <b>`name`</b>: See above.
* <b>`dtype`</b>: See above.
* <b>`kernel`</b>: `Tensor`-like object. The convolution kernel as applied to the
    inputs, i.e. after any reparameterizations.
* <b>`bias`</b>: `Tensor`-like object. The bias vector as applied to the inputs, i.e.
    after any reparameterizations.
* <b>`trainable_variables`</b>: List of trainable variables.
* <b>`non_trainable_variables`</b>: List of non-trainable variables.
* <b>`variables`</b>: List of all variables of this layer, trainable and non-trainable.
* <b>`updates`</b>: List of update ops of this layer.
* <b>`losses`</b>: List of losses added by this layer.

Mutable properties:
* <b>`trainable`</b>: Boolean. Whether the layer should be trained.
* <b>`input_spec`</b>: Optional `InputSpec` object specifying the constraints on inputs
    that can be accepted by the layer.

## Properties

<h3 id="activation"><code>activation</code></h3>



<h3 id="activity_regularizer"><code>activity_regularizer</code></h3>

Optional regularizer function for the output of this layer.

<h3 id="bias"><code>bias</code></h3>



<h3 id="bias_initializer"><code>bias_initializer</code></h3>



<h3 id="bias_parameterizer"><code>bias_parameterizer</code></h3>



<h3 id="bias_regularizer"><code>bias_regularizer</code></h3>



<h3 id="channel_separable"><code>channel_separable</code></h3>



<h3 id="corr"><code>corr</code></h3>



<h3 id="data_format"><code>data_format</code></h3>



<h3 id="dtype"><code>dtype</code></h3>



<h3 id="extra_pad_end"><code>extra_pad_end</code></h3>



<h3 id="filters"><code>filters</code></h3>



<h3 id="graph"><code>graph</code></h3>



<h3 id="inbound_nodes"><code>inbound_nodes</code></h3>

Deprecated, do NOT use! Only for compatibility with external Keras.

<h3 id="input"><code>input</code></h3>

Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.

#### Returns:

Input tensor or list of input tensors.


#### Raises:

* <b>`AttributeError`</b>: if the layer is connected to
    more than one incoming layers.


#### Raises:

* <b>`RuntimeError`</b>: If called in Eager mode.
* <b>`AttributeError`</b>: If no inbound nodes are found.

<h3 id="input_mask"><code>input_mask</code></h3>

Retrieves the input mask tensor(s) of a layer.

Only applicable if the layer has exactly one inbound node,
i.e. if it is connected to one incoming layer.

#### Returns:

Input mask tensor (potentially None) or list of input
mask tensors.


#### Raises:

* <b>`AttributeError`</b>: if the layer is connected to
    more than one incoming layers.

<h3 id="input_shape"><code>input_shape</code></h3>

Retrieves the input shape(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer, or if all inputs
have the same shape.

#### Returns:

Input shape, as an integer shape tuple
(or list of shape tuples, one tuple per input tensor).


#### Raises:

* <b>`AttributeError`</b>: if the layer has no defined input_shape.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="kernel"><code>kernel</code></h3>



<h3 id="kernel_initializer"><code>kernel_initializer</code></h3>



<h3 id="kernel_parameterizer"><code>kernel_parameterizer</code></h3>



<h3 id="kernel_regularizer"><code>kernel_regularizer</code></h3>



<h3 id="kernel_support"><code>kernel_support</code></h3>



<h3 id="losses"><code>losses</code></h3>

Losses which are associated with this `Layer`.

Note that when executing eagerly, getting this property evaluates
regularizers. When using graph execution, variable regularization ops have
already been created and are simply returned here.

#### Returns:

A list of tensors.

<h3 id="name"><code>name</code></h3>



<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>



<h3 id="non_trainable_weights"><code>non_trainable_weights</code></h3>



<h3 id="outbound_nodes"><code>outbound_nodes</code></h3>

Deprecated, do NOT use! Only for compatibility with external Keras.

<h3 id="output"><code>output</code></h3>

Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output,
i.e. if it is connected to one incoming layer.

#### Returns:

Output tensor or list of output tensors.


#### Raises:

* <b>`AttributeError`</b>: if the layer is connected to more than one incoming
    layers.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="output_mask"><code>output_mask</code></h3>

Retrieves the output mask tensor(s) of a layer.

Only applicable if the layer has exactly one inbound node,
i.e. if it is connected to one incoming layer.

#### Returns:

Output mask tensor (potentially None) or list of output
mask tensors.


#### Raises:

* <b>`AttributeError`</b>: if the layer is connected to
    more than one incoming layers.

<h3 id="output_shape"><code>output_shape</code></h3>

Retrieves the output shape(s) of a layer.

Only applicable if the layer has one output,
or if all outputs have the same shape.

#### Returns:

Output shape, as an integer shape tuple
(or list of shape tuples, one tuple per output tensor).


#### Raises:

* <b>`AttributeError`</b>: if the layer has no defined output shape.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="padding"><code>padding</code></h3>



<h3 id="scope_name"><code>scope_name</code></h3>



<h3 id="strides_down"><code>strides_down</code></h3>



<h3 id="strides_up"><code>strides_up</code></h3>



<h3 id="trainable_variables"><code>trainable_variables</code></h3>



<h3 id="trainable_weights"><code>trainable_weights</code></h3>



<h3 id="updates"><code>updates</code></h3>



<h3 id="use_bias"><code>use_bias</code></h3>



<h3 id="variables"><code>variables</code></h3>

Returns the list of all layer variables/weights.

#### Returns:

A list of variables.

<h3 id="weights"><code>weights</code></h3>

Returns the list of all layer variables/weights.

#### Returns:

A list of variables.



## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(
    *args,
    **kwargs
)
```



<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    inputs,
    *args,
    **kwargs
)
```

Wraps `call`, applying pre- and post-processing steps.

#### Arguments:

* <b>`inputs`</b>: input tensor(s).
* <b>`*args`</b>: additional positional arguments to be passed to `self.call`.
* <b>`**kwargs`</b>: additional keyword arguments to be passed to `self.call`.
    **Note**: kwarg `scope` is reserved for use by the layer.


#### Returns:

  Output tensor(s).

Note:
  - If the layer's `call` method takes a `scope` keyword argument,
    this argument will be automatically set to the current variable scope.
  - If the layer's `call` method takes a `mask` argument (as some Keras
    layers do), its default value will be set to the mask generated
    for `inputs` by the previous layer (if `input` did come from
    a layer that generated a corresponding mask, i.e. if it came from
    a Keras layer with masking support.


#### Raises:

* <b>`ValueError`</b>: if the layer's `call` method returns None (an invalid value).

<h3 id="__deepcopy__"><code>__deepcopy__</code></h3>

``` python
__deepcopy__(memo)
```



<h3 id="add_loss"><code>add_loss</code></h3>

``` python
add_loss(
    losses,
    inputs=None
)
```



<h3 id="add_update"><code>add_update</code></h3>

``` python
add_update(
    updates,
    inputs=None
)
```

Add update op(s), potentially dependent on layer inputs.

Weight updates (for instance, the updates of the moving mean and variance
in a BatchNormalization layer) may be dependent on the inputs passed
when calling a layer. Hence, when reusing the same layer on
different inputs `a` and `b`, some entries in `layer.updates` may be
dependent on `a` and some on `b`. This method automatically keeps track
of dependencies.

The `get_updates_for` method allows to retrieve the updates relevant to a
specific set of inputs.

This call is ignored when eager execution is enabled (in that case, variable
updates are run on the fly and thus do not need to be tracked for later
execution).

#### Arguments:

* <b>`updates`</b>: Update op, or list/tuple of update ops.
* <b>`inputs`</b>: If anything other than None is passed, it signals the updates
    are conditional on some of the layer's inputs,
    and thus they should only be run where these inputs are available.
    This is the case for BatchNormalization updates, for instance.
    If None, the updates will be taken into account unconditionally,
    and you are responsible for making sure that any dependency they might
    have is available at runtime.
    A step counter might fall into this category.

<h3 id="add_variable"><code>add_variable</code></h3>

``` python
add_variable(
    *args,
    **kwargs
)
```

Alias for `add_weight`.

<h3 id="add_weight"><code>add_weight</code></h3>

``` python
add_weight(
    name,
    shape,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    constraint=None,
    use_resource=None,
    synchronization=vs.VariableSynchronization.AUTO,
    aggregation=vs.VariableAggregation.NONE,
    partitioner=None
)
```

Adds a new variable to the layer, or gets an existing one; returns it.

#### Arguments:

* <b>`name`</b>: variable name.
* <b>`shape`</b>: variable shape.
* <b>`dtype`</b>: The type of the variable. Defaults to `self.dtype` or `float32`.
* <b>`initializer`</b>: initializer instance (callable).
* <b>`regularizer`</b>: regularizer instance (callable).
* <b>`trainable`</b>: whether the variable should be part of the layer's
    "trainable_variables" (e.g. variables, biases)
    or "non_trainable_variables" (e.g. BatchNorm mean, stddev).
    Note, if the current variable scope is marked as non-trainable
    then this parameter is ignored and any added variables are also
    marked as non-trainable. `trainable` defaults to `True` unless
    `synchronization` is set to `ON_READ`.
* <b>`constraint`</b>: constraint instance (callable).
* <b>`use_resource`</b>: Whether to use `ResourceVariable`.
* <b>`synchronization`</b>: Indicates when a distributed a variable will be
    aggregated. Accepted values are constants defined in the class
    BAD_LINK. By default the synchronization is set to
    `AUTO` and the current `DistributionStrategy` chooses
    when to synchronize. If `synchronization` is set to `ON_READ`,
    `trainable` must not be set to `True`.
* <b>`aggregation`</b>: Indicates how a distributed variable will be aggregated.
    Accepted values are constants defined in the class
    BAD_LINK.
* <b>`partitioner`</b>: (optional) partitioner instance (callable).  If
    provided, when the requested variable is created it will be split
    into multiple partitions according to `partitioner`.  In this case,
    an instance of `PartitionedVariable` is returned.  Available
    partitioners include `tf.fixed_size_partitioner` and
    `tf.variable_axis_size_partitioner`.  For more details, see the
    documentation of `tf.get_variable` and the  "Variable Partitioners
    and Sharding" section of the API guide.


#### Returns:

The created variable.  Usually either a `Variable` or `ResourceVariable`
instance.  If `partitioner` is not `None`, a `PartitionedVariable`
instance is returned.


#### Raises:

* <b>`RuntimeError`</b>: If called with partioned variable regularization and
    eager execution is enabled.
* <b>`ValueError`</b>: When trainable has been set to True with synchronization
    set as `ON_READ`.

<h3 id="apply"><code>apply</code></h3>

``` python
apply(
    inputs,
    *args,
    **kwargs
)
```

Apply the layer on a input.

This simply wraps `self.__call__`.

#### Arguments:

* <b>`inputs`</b>: Input tensor(s).
* <b>`*args`</b>: additional positional arguments to be passed to `self.call`.
* <b>`**kwargs`</b>: additional keyword arguments to be passed to `self.call`.


#### Returns:

Output tensor(s).

<h3 id="build"><code>build</code></h3>

``` python
build(input_shape)
```



<h3 id="call"><code>call</code></h3>

``` python
call(inputs)
```



<h3 id="compute_mask"><code>compute_mask</code></h3>

``` python
compute_mask(
    inputs,
    mask=None
)
```

Computes an output mask tensor.

#### Arguments:

* <b>`inputs`</b>: Tensor or list of tensors.
* <b>`mask`</b>: Tensor or list of tensors.


#### Returns:

None or a tensor (or list of tensors,
    one per output tensor of the layer).

<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

``` python
compute_output_shape(input_shape)
```



<h3 id="count_params"><code>count_params</code></h3>

``` python
count_params()
```

Count the total number of scalars composing the weights.

#### Returns:

An integer count.


#### Raises:

* <b>`ValueError`</b>: if the layer isn't yet built
      (in which case its weights aren't yet defined).

<h3 id="from_config"><code>from_config</code></h3>

``` python
from_config(
    cls,
    config
)
```

Creates a layer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same layer from the config
dictionary. It does not handle layer connectivity
(handled by Network), nor weights (handled by `set_weights`).

#### Arguments:

* <b>`config`</b>: A Python dictionary, typically the
        output of get_config.


#### Returns:

A layer instance.

<h3 id="get_config"><code>get_config</code></h3>

``` python
get_config()
```

Returns the config of the layer.

A layer config is a Python dictionary (serializable)
containing the configuration of a layer.
The same layer can be reinstantiated later
(without its trained weights) from this configuration.

The config of a layer does not include connectivity
information, nor the layer class name. These are handled
by `Network` (one layer of abstraction above).

#### Returns:

Python dictionary.

<h3 id="get_input_at"><code>get_input_at</code></h3>

``` python
get_input_at(node_index)
```

Retrieves the input tensor(s) of a layer at a given node.

#### Arguments:

* <b>`node_index`</b>: Integer, index of the node
        from which to retrieve the attribute.
        E.g. `node_index=0` will correspond to the
        first time the layer was called.


#### Returns:

A tensor (or list of tensors if the layer has multiple inputs).


#### Raises:

* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_input_mask_at"><code>get_input_mask_at</code></h3>

``` python
get_input_mask_at(node_index)
```

Retrieves the input mask tensor(s) of a layer at a given node.

#### Arguments:

* <b>`node_index`</b>: Integer, index of the node
        from which to retrieve the attribute.
        E.g. `node_index=0` will correspond to the
        first time the layer was called.


#### Returns:

A mask tensor
(or list of tensors if the layer has multiple inputs).

<h3 id="get_input_shape_at"><code>get_input_shape_at</code></h3>

``` python
get_input_shape_at(node_index)
```

Retrieves the input shape(s) of a layer at a given node.

#### Arguments:

* <b>`node_index`</b>: Integer, index of the node
        from which to retrieve the attribute.
        E.g. `node_index=0` will correspond to the
        first time the layer was called.


#### Returns:

A shape tuple
(or list of shape tuples if the layer has multiple inputs).


#### Raises:

* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_losses_for"><code>get_losses_for</code></h3>

``` python
get_losses_for(inputs)
```

Retrieves losses relevant to a specific set of inputs.

#### Arguments:

* <b>`inputs`</b>: Input tensor or list/tuple of input tensors.


#### Returns:

List of loss tensors of the layer that depend on `inputs`.


#### Raises:

* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_output_at"><code>get_output_at</code></h3>

``` python
get_output_at(node_index)
```

Retrieves the output tensor(s) of a layer at a given node.

#### Arguments:

* <b>`node_index`</b>: Integer, index of the node
        from which to retrieve the attribute.
        E.g. `node_index=0` will correspond to the
        first time the layer was called.


#### Returns:

A tensor (or list of tensors if the layer has multiple outputs).


#### Raises:

* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_output_mask_at"><code>get_output_mask_at</code></h3>

``` python
get_output_mask_at(node_index)
```

Retrieves the output mask tensor(s) of a layer at a given node.

#### Arguments:

* <b>`node_index`</b>: Integer, index of the node
        from which to retrieve the attribute.
        E.g. `node_index=0` will correspond to the
        first time the layer was called.


#### Returns:

A mask tensor
(or list of tensors if the layer has multiple outputs).

<h3 id="get_output_shape_at"><code>get_output_shape_at</code></h3>

``` python
get_output_shape_at(node_index)
```

Retrieves the output shape(s) of a layer at a given node.

#### Arguments:

* <b>`node_index`</b>: Integer, index of the node
        from which to retrieve the attribute.
        E.g. `node_index=0` will correspond to the
        first time the layer was called.


#### Returns:

A shape tuple
(or list of shape tuples if the layer has multiple outputs).


#### Raises:

* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_updates_for"><code>get_updates_for</code></h3>

``` python
get_updates_for(inputs)
```

Retrieves updates relevant to a specific set of inputs.

#### Arguments:

* <b>`inputs`</b>: Input tensor or list/tuple of input tensors.


#### Returns:

List of update ops of the layer that depend on `inputs`.


#### Raises:

* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_weights"><code>get_weights</code></h3>

``` python
get_weights()
```

Returns the current weights of the layer.

#### Returns:

Weights values as a list of numpy arrays.

<h3 id="set_weights"><code>set_weights</code></h3>

``` python
set_weights(weights)
```

Sets the weights of the layer, from Numpy arrays.

#### Arguments:

* <b>`weights`</b>: a list of Numpy arrays. The number
        of arrays and their shape must match
        number of the dimensions of the weights
        of the layer (i.e. it should match the
        output of `get_weights`).


#### Raises:

* <b>`ValueError`</b>: If the provided weights list does not match the
        layer's specifications.



