<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.SignalConv1D" />
<meta itemprop="path" content="Stable" />
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
<meta itemprop="property" content="dynamic"/>
<meta itemprop="property" content="extra_pad_end"/>
<meta itemprop="property" content="filters"/>
<meta itemprop="property" content="input"/>
<meta itemprop="property" content="input_mask"/>
<meta itemprop="property" content="input_shape"/>
<meta itemprop="property" content="input_spec"/>
<meta itemprop="property" content="kernel"/>
<meta itemprop="property" content="kernel_initializer"/>
<meta itemprop="property" content="kernel_parameterizer"/>
<meta itemprop="property" content="kernel_regularizer"/>
<meta itemprop="property" content="kernel_support"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="metrics"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="output"/>
<meta itemprop="property" content="output_mask"/>
<meta itemprop="property" content="output_shape"/>
<meta itemprop="property" content="padding"/>
<meta itemprop="property" content="strides_down"/>
<meta itemprop="property" content="strides_up"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="updates"/>
<meta itemprop="property" content="use_bias"/>
<meta itemprop="property" content="use_explicit"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
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
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfc.SignalConv1D


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/signal_conv.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `SignalConv1D`

1D convolution layer.



### Aliases:

* Class `tfc.python.layers.signal_conv.SignalConv1D`


<!-- Placeholder for "Used in" -->

This layer creates a filter kernel that is convolved or cross correlated with
the layer input to produce an output tensor. The main difference of this class
to `tf.layers.Conv1D` is how padding, up- and downsampling, and alignment
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

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/signal_conv.py">View source</a>

``` python
__init__(
    filters,
    kernel_support,
    corr=False,
    strides_down=1,
    strides_up=1,
    padding='valid',
    extra_pad_end=True,
    channel_separable=False,
    data_format='channels_last',
    activation=None,
    use_bias=False,
    use_explicit=False,
    kernel_initializer=tf.initializers.variance_scaling(),
    bias_initializer=tf.initializers.zeros(),
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_parameterizer=parameterizers.RDFTParameterizer(),
    bias_parameterizer=None,
    **kwargs
)
```

Initializer.


#### Arguments:


* <b>`filters`</b>: Integer. If `not channel_separable`, specifies the total number
  of filters, which is equal to the number of output channels. Otherwise,
  specifies the number of filters per channel, which makes the number of
  output channels equal to `filters` times the number of input channels.
* <b>`kernel_support`</b>: An integer or iterable of {rank} integers, specifying the
  length of the convolution/correlation window in each dimension.
* <b>`corr`</b>: Boolean. If True, compute cross correlation. If False, convolution.
* <b>`strides_down`</b>: An integer or iterable of {rank} integers, specifying an
  optional downsampling stride after the convolution/correlation.
* <b>`strides_up`</b>: An integer or iterable of {rank} integers, specifying an
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
  input tensors with shape `(batch, ..., channels)`, while
  `channels_first` corresponds to input tensors with shape `(batch,
  channels, ...)`.
* <b>`activation`</b>: Activation function or `None`.
* <b>`use_bias`</b>: Boolean, whether an additive constant will be applied to each
  output channel.
* <b>`use_explicit`</b>: Boolean, whether to use `EXPLICIT` padding mode (supported
  in TensorFlow >1.14).
* <b>`kernel_initializer`</b>: An initializer for the filter kernel.
* <b>`bias_initializer`</b>: An initializer for the bias vector.
* <b>`kernel_regularizer`</b>: Optional regularizer for the filter kernel.
* <b>`bias_regularizer`</b>: Optional regularizer for the bias vector.
* <b>`kernel_parameterizer`</b>: Reparameterization applied to filter kernel. If not
  `None`, must be a `Parameterizer` object. Defaults to
  `RDFTParameterizer`.
* <b>`bias_parameterizer`</b>: Reparameterization applied to bias. If not `None`,
  must be a `Parameterizer` object. Defaults to `None`.
* <b>`**kwargs`</b>: Other keyword arguments passed to superclass (`Layer`).



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




<h3 id="dynamic"><code>dynamic</code></h3>




<h3 id="extra_pad_end"><code>extra_pad_end</code></h3>




<h3 id="filters"><code>filters</code></h3>




<h3 id="input"><code>input</code></h3>

Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.

#### Returns:

Input tensor or list of input tensors.



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

<h3 id="input_spec"><code>input_spec</code></h3>




<h3 id="kernel"><code>kernel</code></h3>




<h3 id="kernel_initializer"><code>kernel_initializer</code></h3>




<h3 id="kernel_parameterizer"><code>kernel_parameterizer</code></h3>




<h3 id="kernel_regularizer"><code>kernel_regularizer</code></h3>




<h3 id="kernel_support"><code>kernel_support</code></h3>




<h3 id="losses"><code>losses</code></h3>

Losses which are associated with this `Layer`.

Variable regularization tensors are created when this property is accessed,
so it is eager safe: accessing `losses` under a `tf.GradientTape` will
propagate gradients back to the corresponding variables.

#### Returns:

A list of tensors.


<h3 id="metrics"><code>metrics</code></h3>




<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>




<h3 id="non_trainable_weights"><code>non_trainable_weights</code></h3>




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




<h3 id="strides_down"><code>strides_down</code></h3>




<h3 id="strides_up"><code>strides_up</code></h3>




<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
a = tf.Module()
b = tf.Module()
c = tf.Module()
a.b = b
b.c = c
assert list(a.submodules) == [b, c]
assert list(b.submodules) == [c]
assert list(c.submodules) == []
```

#### Returns:

A sequence of all submodules.


<h3 id="trainable"><code>trainable</code></h3>




<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).


<h3 id="trainable_weights"><code>trainable_weights</code></h3>




<h3 id="updates"><code>updates</code></h3>




<h3 id="use_bias"><code>use_bias</code></h3>




<h3 id="use_explicit"><code>use_explicit</code></h3>




<h3 id="variables"><code>variables</code></h3>

Returns the list of all layer variables/weights.

Alias of `self.weights`.

#### Returns:

A list of variables.


<h3 id="weights"><code>weights</code></h3>

Returns the list of all layer variables/weights.


#### Returns:

A list of variables.




## Methods

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



#### Raises:


* <b>`ValueError`</b>: if the layer's `call` method returns None (an invalid value).

<h3 id="build"><code>build</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/signal_conv.py">View source</a>

``` python
build(input_shape)
```

Creates the variables of the layer (optional, for subclass implementers).

This is a method that implementers of subclasses of `Layer` or `Model`
can override if they need a state-creation step in-between
layer instantiation and layer call.

This is typically used to create the weights of `Layer` subclasses.

#### Arguments:


* <b>`input_shape`</b>: Instance of `TensorShape`, or list of instances of
  `TensorShape` if the layer expects a list of inputs
  (one instance per input).

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

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/signal_conv.py">View source</a>

``` python
compute_output_shape(input_shape)
```

Computes the output shape of the layer.

If the layer has not been built, this method will call `build` on the
layer. This assumes that the layer will later be used with inputs that
match the input shape provided here.

#### Arguments:


* <b>`input_shape`</b>: Shape tuple (tuple of integers)
    or list of shape tuples (one per output tensor of the layer).
    Shape tuples can include None for free dimensions,
    instead of an integer.


#### Returns:

An input shape tuple.


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

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

```
class MyModule(tf.Module):
  @tf.Module.with_name_scope
  def __call__(self, x):
    if not hasattr(self, 'w'):
      self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
    return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
mod = MyModule()
mod(tf.ones([8, 32]))
# ==> <tf.Tensor: ...>
mod.w
# ==> <tf.Variable ...'my_module/w:0'>
```

#### Args:


* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.




