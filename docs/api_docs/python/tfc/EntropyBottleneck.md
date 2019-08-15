<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.EntropyBottleneck" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="activity_regularizer"/>
<meta itemprop="property" content="data_format"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="dynamic"/>
<meta itemprop="property" content="filters"/>
<meta itemprop="property" content="init_scale"/>
<meta itemprop="property" content="input"/>
<meta itemprop="property" content="input_mask"/>
<meta itemprop="property" content="input_shape"/>
<meta itemprop="property" content="input_spec"/>
<meta itemprop="property" content="likelihood_bound"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="metrics"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="output"/>
<meta itemprop="property" content="output_mask"/>
<meta itemprop="property" content="output_shape"/>
<meta itemprop="property" content="range_coder_precision"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="tail_mass"/>
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="updates"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
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
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfc.EntropyBottleneck


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/entropy_models.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `EntropyBottleneck`

Entropy bottleneck layer.

Inherits From: [`EntropyModel`](../tfc/EntropyModel.md)

### Aliases:

* Class `tfc.python.layers.entropy_models.EntropyBottleneck`


<!-- Placeholder for "Used in" -->

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

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/entropy_models.py">View source</a>

``` python
__init__(
    init_scale=10,
    filters=(3, 3, 3),
    data_format='channels_last',
    **kwargs
)
```

Initializer.


#### Arguments:


* <b>`init_scale`</b>: Float. A scaling factor determining the initial width of the
  probability densities. This should be chosen big enough so that the
  range of values of the layer inputs roughly falls within the interval
  [`-init_scale`, `init_scale`] at the beginning of training.
* <b>`filters`</b>: An iterable of ints, giving the number of filters at each layer
  of the density model. Generally, the more filters and layers, the more
  expressive is the density model in terms of modeling more complicated
  distributions of the layer inputs. For details, refer to the paper
  referenced above. The default is `[3, 3, 3]`, which should be sufficient
  for most practical purposes.
* <b>`data_format`</b>: Either `'channels_first'` or `'channels_last'` (default).
* <b>`**kwargs`</b>: Other keyword arguments passed to superclass (`EntropyModel`).



## Properties

<h3 id="activity_regularizer"><code>activity_regularizer</code></h3>

Optional regularizer function for the output of this layer.


<h3 id="data_format"><code>data_format</code></h3>




<h3 id="dtype"><code>dtype</code></h3>




<h3 id="dynamic"><code>dynamic</code></h3>




<h3 id="filters"><code>filters</code></h3>




<h3 id="init_scale"><code>init_scale</code></h3>




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




<h3 id="likelihood_bound"><code>likelihood_bound</code></h3>




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

<h3 id="range_coder_precision"><code>range_coder_precision</code></h3>




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


<h3 id="tail_mass"><code>tail_mass</code></h3>




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

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/entropy_models.py">View source</a>

``` python
build(input_shape)
```

Builds the entropy model.

Creates the variables for the network modeling the densities, creates the
auxiliary loss estimating the median and tail quantiles of the densities,
and then uses that to create the probability mass functions and the discrete
cumulative density functions used by the range coder.

#### Arguments:


* <b>`input_shape`</b>: Shape of the input tensor, used to get the number of
  channels.


#### Raises:


* <b>`ValueError`</b>: if `input_shape` doesn't specify the length of the channel
  dimension.

<h3 id="compress"><code>compress</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/entropy_models.py">View source</a>

``` python
compress(inputs)
```

Compress inputs and store their binary representations into strings.


#### Arguments:


* <b>`inputs`</b>: `Tensor` with values to be compressed.


#### Returns:


* <b>`compressed`</b>: String `Tensor` vector containing the compressed
  representation of each batch element of `inputs`.


#### Raises:


* <b>`ValueError`</b>: if `inputs` has an integral or inconsistent `DType`, or
  inconsistent number of channels.

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

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/entropy_models.py">View source</a>

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

<h3 id="decompress"><code>decompress</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/entropy_models.py">View source</a>

``` python
decompress(
    strings,
    shape,
    channels=None
)
```

Decompress values from their compressed string representations.


#### Arguments:


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




