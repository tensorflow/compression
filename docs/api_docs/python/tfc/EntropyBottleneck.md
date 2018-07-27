<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.EntropyBottleneck" />
<meta itemprop="property" content="activity_regularizer"/>
<meta itemprop="property" content="data_format"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="filters"/>
<meta itemprop="property" content="inbound_nodes"/>
<meta itemprop="property" content="init_scale"/>
<meta itemprop="property" content="input"/>
<meta itemprop="property" content="input_mask"/>
<meta itemprop="property" content="input_shape"/>
<meta itemprop="property" content="likelihood_bound"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="optimize_integer_offset"/>
<meta itemprop="property" content="outbound_nodes"/>
<meta itemprop="property" content="output"/>
<meta itemprop="property" content="output_mask"/>
<meta itemprop="property" content="output_shape"/>
<meta itemprop="property" content="range_coder_precision"/>
<meta itemprop="property" content="tail_mass"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="updates"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_loss"/>
<meta itemprop="property" content="add_update"/>
<meta itemprop="property" content="add_variable"/>
<meta itemprop="property" content="add_weight"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="compress"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="decompress"/>
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
<meta itemprop="property" content="visualize"/>
</div>

# tfc.EntropyBottleneck

## Class `EntropyBottleneck`



Entropy bottleneck layer.

This layer models the entropy of the tensor passing through it. During
training, this can be used to impose a (soft) entropy constraint on its
activations, limiting the amount of information flowing through the layer.
After training, the layer can be used to compress any input tensor to a
string, which may be written to a file, and to decompress a file which it
previously generated back to a reconstructed tensor. The entropies estimated
during training or evaluation are approximately equal to the average length of
the strings in bits.

The layer implements a flexible probability density model to estimate entropy
of its input tensor, which is described in the appendix of the paper (please
cite the paper if you use this code for scientific work):

> "Variational image compression with a scale hyperprior"<br />
> J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
> https://arxiv.org/abs/1802.01436

The layer assumes that the input tensor is at least 2D, with a batch dimension
at the beginning and a channel dimension as specified by `data_format`. The
layer trains an independent probability density model for each channel, but
assumes that across all other dimensions, the inputs are i.i.d. (independent
and identically distributed).

Because data compression always involves discretization, the outputs of the
layer are generally only approximations of its inputs. During training,
discretization is modeled using additive uniform noise to ensure
differentiability. The entropies computed during training are differential
entropies. During evaluation, the data is actually quantized, and the
entropies are discrete (Shannon entropies). To make sure the approximated
tensor values are good enough for practical purposes, the training phase must
be used to balance the quality of the approximation with the entropy, by
adding an entropy term to the training loss. See the example in the package
documentation to get started.

Note: the layer always produces exactly one auxiliary loss and one update op,
which are only significant for compression and decompression. To use the
compression feature, the auxiliary loss must be minimized during or after
training. After that, the update op must be executed at least once.

#### Arguments:

* <b>`init_scale`</b>: Float. A scaling factor determining the initial width of the
    probability densities. This should be chosen big enough so that the
    range of values of the layer inputs roughly falls within the interval
    [`-init_scale`, `init_scale`] at the beginning of training.
* <b>`filters`</b>: An iterable of ints, giving the number of filters at each layer of
    the density model. Generally, the more filters and layers, the more
    expressive is the density model in terms of modeling more complicated
    distributions of the layer inputs. For details, refer to the paper
    referenced above. The default is `[3, 3, 3]`, which should be sufficient
    for most practical purposes.
* <b>`tail_mass`</b>: Float, between 0 and 1. The bottleneck layer automatically
    determines the range of input values that should be represented based on
    their frequency of occurrence. Values occurring in the tails of the
    distributions will be clipped to that range during compression.
    `tail_mass` determines the amount of probability mass in the tails which
    is cut off in the worst case. For example, the default value of `1e-9`
    means that at most 1 in a billion input samples will be clipped to the
    range.
* <b>`optimize_integer_offset`</b>: Boolean. Typically, the input values of this layer
    are floats, which means that quantization during evaluation can be
    performed with an arbitrary offset. By default, the layer determines that
    offset automatically. In special situations, such as when it is known that
    the layer will receive only full integer values during evaluation, it can
    be desirable to set this argument to `False` instead, in order to always
    quantize to full integer values.
* <b>`likelihood_bound`</b>: Float. If positive, the returned likelihood values are
    ensured to be greater than or equal to this value. This prevents very
    large gradients with a typical entropy loss (defaults to 1e-9).
* <b>`range_coder_precision`</b>: Integer, between 1 and 16. The precision of the range
    coder used for compression and decompression. This trades off computation
    speed with compression efficiency, where 16 is the slowest but most
    efficient setting. Choosing lower values may increase the average
    codelength slightly compared to the estimated entropies.
* <b>`data_format`</b>: Either `'channels_first'` or `'channels_last'` (default).
* <b>`trainable`</b>: Boolean. Whether the layer should be trained.
* <b>`name`</b>: String. The name of the layer.
* <b>`dtype`</b>: Default dtype of the layer's parameters (default of `None` means use
    the type of the first input).

Read-only properties:
* <b>`init_scale`</b>: See above.
* <b>`filters`</b>: See above.
* <b>`tail_mass`</b>: See above.
* <b>`optimize_integer_offset`</b>: See above.
* <b>`likelihood_bound`</b>: See above.
* <b>`range_coder_precision`</b>: See above.
* <b>`data_format`</b>: See above.
* <b>`name`</b>: String. See above.
* <b>`dtype`</b>: See above.
* <b>`trainable_variables`</b>: List of trainable variables.
* <b>`non_trainable_variables`</b>: List of non-trainable variables.
* <b>`variables`</b>: List of all variables of this layer, trainable and non-trainable.
* <b>`updates`</b>: List of update ops of this layer. Always contains exactly one
    update op, which must be run once after the last training step, before
    `compress` or `decompress` is used.
* <b>`losses`</b>: List of losses added by this layer. Always contains exactly one
    auxiliary loss, which must be added to the training loss.

Mutable properties:
* <b>`trainable`</b>: Boolean. Whether the layer should be trained.
* <b>`input_spec`</b>: Optional `InputSpec` object specifying the constraints on inputs
    that can be accepted by the layer.

## Properties

<h3 id="activity_regularizer"><code>activity_regularizer</code></h3>

Optional regularizer function for the output of this layer.

<h3 id="data_format"><code>data_format</code></h3>



<h3 id="dtype"><code>dtype</code></h3>



<h3 id="filters"><code>filters</code></h3>



<h3 id="inbound_nodes"><code>inbound_nodes</code></h3>

Deprecated, do NOT use! Only for compatibility with external Keras.

<h3 id="init_scale"><code>init_scale</code></h3>



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

<h3 id="likelihood_bound"><code>likelihood_bound</code></h3>



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



<h3 id="optimize_integer_offset"><code>optimize_integer_offset</code></h3>



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

<h3 id="range_coder_precision"><code>range_coder_precision</code></h3>



<h3 id="tail_mass"><code>tail_mass</code></h3>



<h3 id="trainable_variables"><code>trainable_variables</code></h3>



<h3 id="trainable_weights"><code>trainable_weights</code></h3>



<h3 id="updates"><code>updates</code></h3>



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
    init_scale=10,
    filters=(3, 3, 3),
    tail_mass=1e-09,
    optimize_integer_offset=True,
    likelihood_bound=1e-09,
    range_coder_precision=16,
    data_format='channels_last',
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


#### Returns:

  Output tensor(s).

Note:
  - The following optional keyword arguments are reserved for specific uses:
    * `training`: Boolean scalar tensor of Python boolean indicating
      whether the `call` is meant for training or inference.
    * `mask`: Boolean input mask.
  - If the layer's `call` method takes a `mask` argument (as some Keras
    layers do), its default value will be set to the mask generated
    for `inputs` by the previous layer (if `input` did come from
    a layer that generated a corresponding mask, i.e. if it came from
    a Keras layer with masking support.


#### Raises:

* <b>`ValueError`</b>: if the layer's `call` method returns None (an invalid value).

<h3 id="add_loss"><code>add_loss</code></h3>

``` python
add_loss(
    losses,
    inputs=None
)
```

Add loss tensor(s), potentially dependent on layer inputs.

Some losses (for instance, activity regularization losses) may be dependent
on the inputs passed when calling a layer. Hence, when reusing the same
layer on different inputs `a` and `b`, some entries in `layer.losses` may
be dependent on `a` and some on `b`. This method automatically keeps track
of dependencies.

The `get_losses_for` method allows to retrieve the losses relevant to a
specific set of inputs.

Note that `add_loss` is not supported when executing eagerly. Instead,
variable regularizers may be added through `add_variable`. Activity
regularization is not supported directly (but such losses may be returned
from `Layer.call()`).

#### Arguments:

* <b>`losses`</b>: Loss tensor, or list/tuple of tensors.
* <b>`inputs`</b>: If anything other than None is passed, it signals the losses
    are conditional on some of the layer's inputs,
    and thus they should only be run where these inputs are available.
    This is the case for activity regularization losses, for instance.
    If `None` is passed, the losses are assumed
    to be unconditional, and will apply across all dataflows of the layer
    (e.g. weight regularization losses).


#### Raises:

* <b>`RuntimeError`</b>: If called in Eager mode.

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
    partitioner=None,
    use_resource=None,
    synchronization=vs.VariableSynchronization.AUTO,
    aggregation=vs.VariableAggregation.NONE,
    getter=None
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
* <b>`partitioner`</b>: Partitioner to be passed to the `Checkpointable` API.
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
* <b>`getter`</b>: Variable getter argument to be passed to the `Checkpointable` API.


#### Returns:

The created variable.  Usually either a `Variable` or `ResourceVariable`
instance.  If `partitioner` is not `None`, a `PartitionedVariable`
instance is returned.


#### Raises:

* <b>`RuntimeError`</b>: If called with partioned variable regularization and
    eager execution is enabled.
* <b>`ValueError`</b>: When giving unsupported dtype and no initializer or when
    trainable has been set to True with synchronization set as `ON_READ`.

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

Builds the layer.

Creates the variables for the network modeling the densities, creates the
auxiliary loss estimating the median and tail quantiles of the densities,
and then uses that to create the probability mass functions and the update
op that produces the discrete cumulative density functions used by the range
coder.

#### Args:

* <b>`input_shape`</b>: Shape of the input tensor, used to get the number of
    channels.


#### Raises:

* <b>`ValueError`</b>: if `input_shape` doesn't specify the length of the channel
    dimension.

<h3 id="call"><code>call</code></h3>

``` python
call(
    inputs,
    training
)
```

Pass a tensor through the bottleneck.

#### Args:

* <b>`inputs`</b>: The tensor to be passed through the bottleneck.
* <b>`training`</b>: Boolean. If `True`, returns a differentiable approximation of
    the inputs, and their likelihoods under the modeled probability
    densities. If `False`, returns the quantized inputs and their
    likelihoods under the corresponding probability mass function. These
    quantities can't be used for training, as they are not differentiable,
    but represent actual compression more closely.


#### Returns:

* <b>`values`</b>: `Tensor` with the same shape as `inputs` containing the perturbed
    or quantized input values.
* <b>`likelihood`</b>: `Tensor` with the same shape as `inputs` containing the
    likelihood of `values` under the modeled probability distributions.


#### Raises:

* <b>`ValueError`</b>: if `inputs` has different `dtype` or number of channels than
    a previous set of inputs the model was invoked with earlier.

<h3 id="compress"><code>compress</code></h3>

``` python
compress(inputs)
```

Compress inputs and store their binary representations into strings.

#### Args:

* <b>`inputs`</b>: `Tensor` with values to be compressed.


#### Returns:

String `Tensor` vector containing the compressed representation of each
batch element of `inputs`.

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

<h3 id="decompress"><code>decompress</code></h3>

``` python
decompress(
    strings,
    shape,
    channels=None
)
```

Decompress values from their compressed string representations.

#### Args:

* <b>`strings`</b>: A string `Tensor` vector containing the compressed data.
* <b>`shape`</b>: A `Tensor` vector of int32 type. Contains the shape of the tensor
    to be decompressed, excluding the batch dimension.
* <b>`channels`</b>: Integer. Specifies the number of channels statically. Needs only
    be set if the layer hasn't been built yet (i.e., this is the first input
    it receives).


#### Returns:

The decompressed `Tensor`. Its shape will be equal to `shape` prepended
with the batch dimension from `strings`.


#### Raises:

* <b>`ValueError`</b>: If the length of `shape` isn't available at graph construction
    time.

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

<h3 id="visualize"><code>visualize</code></h3>

``` python
visualize()
```

Multi-channel visualization of densities as images.

Creates and returns an image summary visualizing the current probabilty
density estimates. The image contains one row for each channel. Within each
row, the pixel intensities are proportional to probability values, and each
row is centered on the median of the corresponding distribution.

#### Returns:

The created image summary.



