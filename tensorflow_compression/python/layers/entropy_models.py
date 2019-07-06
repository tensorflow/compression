# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Entropy model layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats
import tensorflow as tf

from tensorflow.python.keras.engine import input_spec
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import range_coding_ops


__all__ = [
    "EntropyModel",
    "EntropyBottleneck",
    "SymmetricConditional",
    "GaussianConditional",
    "LogisticConditional",
    "LaplacianConditional",
]


class EntropyModel(tf.keras.layers.Layer):
  """Entropy model (base class).

  Arguments:
    tail_mass: Float, between 0 and 1. The bottleneck layer automatically
      determines the range of input values based on their frequency of
      occurrence. Values occurring in the tails of the distributions will not be
      encoded with range coding, but using a Golomb-like code. `tail_mass`
      determines the amount of probability mass in the tails which will be
      Golomb-coded. For example, the default value of `2 ** -8` means that on
      average, one 256th of all values will use the Golomb code.
    likelihood_bound: Float. If positive, the returned likelihood values are
      ensured to be greater than or equal to this value. This prevents very
      large gradients with a typical entropy loss (defaults to 1e-9).
    range_coder_precision: Integer, between 1 and 16. The precision of the range
      coder used for compression and decompression. This trades off computation
      speed with compression efficiency, where 16 is the slowest but most
      efficient setting. Choosing lower values may increase the average
      codelength slightly compared to the estimated entropies.
    data_format: Either `'channels_first'` or `'channels_last'` (default).
    trainable: Boolean. Whether the layer should be trained.
    name: String. The name of the layer.
    dtype: `DType` of the layer's inputs, parameters, returned likelihoods, and
      outputs during training. Default of `None` means to use the type of the
      first input.

  Read-only properties:
    tail_mass: See above.
    likelihood_bound: See above.
    range_coder_precision: See above.
    data_format: See above.
    name: String. See above.
    dtype: See above.
    trainable_variables: List of trainable variables.
    non_trainable_variables: List of non-trainable variables.
    variables: List of all variables of this layer, trainable and non-trainable.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer.

  Mutable properties:
    trainable: Boolean. Whether the layer should be trained.
    input_spec: Optional `InputSpec` object specifying the constraints on inputs
      that can be accepted by the layer.
  """

  _setattr_tracking = False

  def __init__(self, tail_mass=2 ** -8, likelihood_bound=1e-9,
               range_coder_precision=16, **kwargs):
    super(EntropyModel, self).__init__(**kwargs)
    self._tail_mass = float(tail_mass)
    if not 0 < self.tail_mass < 1:
      raise ValueError(
          "`tail_mass` must be between 0 and 1, got {}.".format(self.tail_mass))
    self._likelihood_bound = float(likelihood_bound)
    self._range_coder_precision = int(range_coder_precision)

  @property
  def tail_mass(self):
    return self._tail_mass

  @property
  def likelihood_bound(self):
    return self._likelihood_bound

  @property
  def range_coder_precision(self):
    return self._range_coder_precision

  def _quantize(self, inputs, mode):
    """Perturb or quantize a `Tensor` and optionally dequantize.

    Arguments:
      inputs: `Tensor`. The input values.
      mode: String. Can take on one of three values: `'noise'` (adds uniform
        noise), `'dequantize'` (quantizes and dequantizes), and `'symbols'`
        (quantizes and produces integer symbols for range coder).

    Returns:
      The quantized/perturbed `inputs`. The returned `Tensor` should have type
      `self.dtype` if mode is `'noise'`, `'dequantize'`; `tf.int32` if mode is
      `'symbols'`.
    """
    raise NotImplementedError("Must inherit from EntropyModel.")

  def _dequantize(self, inputs, mode):
    """Dequantize a `Tensor`.

    The opposite to `_quantize(inputs, mode='symbols')`.

    Arguments:
      inputs: `Tensor`. The range coder symbols.
      mode: String. Must be `'dequantize'`.

    Returns:
      The dequantized `inputs`. The returned `Tensor` should have type
      `self.dtype`.
    """
    raise NotImplementedError("Must inherit from EntropyModel.")

  def _likelihood(self, inputs):
    """Compute the likelihood of the inputs under the model.

    Arguments:
      inputs: `Tensor`. The input values.

    Returns:
      `Tensor` of same shape and type as `inputs`, giving the likelihoods
      evaluated at `inputs`.
    """
    raise NotImplementedError("Must inherit from EntropyModel.")

  def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
    """Helper function for computing the CDF from the PMF."""

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      def loop_body(args):
        prob, length, tail = args
        prob = tf.concat([prob[:length], tail], axis=0)
        cdf = range_coding_ops.pmf_to_quantized_cdf(
            prob, precision=self.range_coder_precision)
        return tf.pad(
            cdf, [[0, max_length - length]], mode="CONSTANT", constant_values=0)

      return tf.map_fn(
          loop_body, (pmf, pmf_length, tail_mass),
          dtype=tf.int32, back_prop=False, name="pmf_to_cdf")

  def call(self, inputs, training):
    """Pass a tensor through the bottleneck.

    Arguments:
      inputs: The tensor to be passed through the bottleneck.
      training: Boolean. If `True`, returns a differentiable approximation of
        the inputs, and their likelihoods under the modeled probability
        densities. If `False`, returns the quantized inputs and their
        likelihoods under the corresponding probability mass function. These
        quantities can't be used for training, as they are not differentiable,
        but represent actual compression more closely.

    Returns:
      values: `Tensor` with the same shape as `inputs` containing the perturbed
        or quantized input values.
      likelihood: `Tensor` with the same shape as `inputs` containing the
        likelihood of `values` under the modeled probability distributions.

    Raises:
      ValueError: if `inputs` has an integral or inconsistent `DType`, or
        inconsistent number of channels.
    """
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    if inputs.dtype.is_integer:
      raise ValueError(
          "{} can't take integer inputs.".format(type(self).__name__))

    outputs = self._quantize(inputs, "noise" if training else "dequantize")
    assert outputs.dtype == self.dtype
    likelihood = self._likelihood(outputs)
    if self.likelihood_bound > 0:
      likelihood_bound = tf.constant(self.likelihood_bound, dtype=self.dtype)
      likelihood = math_ops.lower_bound(likelihood, likelihood_bound)

    if not tf.executing_eagerly():
      outputs_shape, likelihood_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(outputs_shape)
      likelihood.set_shape(likelihood_shape)

    return outputs, likelihood

  def compress(self, inputs):
    """Compress inputs and store their binary representations into strings.

    Arguments:
      inputs: `Tensor` with values to be compressed.

    Returns:
      compressed: String `Tensor` vector containing the compressed
        representation of each batch element of `inputs`.

    Raises:
      ValueError: if `inputs` has an integral or inconsistent `DType`, or
        inconsistent number of channels.
    """
    with tf.name_scope(self._name_scope()):
      inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
      if not self.built:
        # Check input assumptions set before layer building, e.g. input rank.
        input_spec.assert_input_compatibility(
            self.input_spec, inputs, self.name)
        if self.dtype is None:
          self._dtype = inputs.dtype.base_dtype.name
        self.build(inputs.shape)

      # Check input assumptions set after layer building, e.g. input shape.
      if not tf.executing_eagerly():
        input_spec.assert_input_compatibility(
            self.input_spec, inputs, self.name)
        if inputs.dtype.is_integer:
          raise ValueError(
              "{} can't take integer inputs.".format(type(self).__name__))

      symbols = self._quantize(inputs, "symbols")
      assert symbols.dtype == tf.int32

      ndim = self.input_spec.ndim
      indexes = self._prepare_indexes(shape=tf.shape(symbols)[1:])
      broadcast_indexes = (indexes.shape.ndims != ndim)
      if broadcast_indexes:
        # We can't currently broadcast over anything else but the batch axis.
        assert indexes.shape.ndims == ndim - 1
        args = (symbols,)
      else:
        args = (symbols, indexes)

      def loop_body(args):
        string = range_coding_ops.unbounded_index_range_encode(
            args[0], indexes if broadcast_indexes else args[1],
            self._quantized_cdf, self._cdf_length, self._offset,
            precision=self.range_coder_precision, overflow_width=4,
            debug_level=0)
        return string

      strings = tf.map_fn(
          loop_body, args, dtype=tf.string,
          back_prop=False, name="compress")

      if not tf.executing_eagerly():
        strings.set_shape(inputs.shape[:1])

      return strings

  def decompress(self, strings, **kwargs):
    """Decompress values from their compressed string representations.

    Arguments:
      strings: A string `Tensor` vector containing the compressed data.
      **kwargs: Model-specific keyword arguments.

    Returns:
      The decompressed `Tensor`.
    """
    with tf.name_scope(self._name_scope()):
      strings = tf.convert_to_tensor(strings, dtype=tf.string)

      indexes = self._prepare_indexes(**kwargs)
      ndim = self.input_spec.ndim
      broadcast_indexes = (indexes.shape.ndims != ndim)
      if broadcast_indexes:
        # We can't currently broadcast over anything else but the batch axis.
        assert indexes.shape.ndims == ndim - 1
        args = (strings,)
      else:
        args = (strings, indexes)

      def loop_body(args):
        symbols = range_coding_ops.unbounded_index_range_decode(
            args[0], indexes if broadcast_indexes else args[1],
            self._quantized_cdf, self._cdf_length, self._offset,
            precision=self.range_coder_precision, overflow_width=4,
            debug_level=0)
        return symbols

      symbols = tf.map_fn(
          loop_body, args, dtype=tf.int32, back_prop=False, name="decompress")

      outputs = self._dequantize(symbols, "dequantize")
      assert outputs.dtype == self.dtype

      if not tf.executing_eagerly():
        outputs.set_shape(self.input_spec.shape)

      return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    return input_shape, input_shape


class EntropyBottleneck(EntropyModel):
  """Entropy bottleneck layer.

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

  Arguments:
    init_scale: Float. A scaling factor determining the initial width of the
      probability densities. This should be chosen big enough so that the
      range of values of the layer inputs roughly falls within the interval
      [`-init_scale`, `init_scale`] at the beginning of training.
    filters: An iterable of ints, giving the number of filters at each layer of
      the density model. Generally, the more filters and layers, the more
      expressive is the density model in terms of modeling more complicated
      distributions of the layer inputs. For details, refer to the paper
      referenced above. The default is `[3, 3, 3]`, which should be sufficient
      for most practical purposes.
    tail_mass: Float, between 0 and 1. The bottleneck layer automatically
      determines the range of input values based on their frequency of
      occurrence. Values occurring in the tails of the distributions will not be
      encoded with range coding, but using a Golomb-like code. `tail_mass`
      determines the amount of probability mass in the tails which will be
      Golomb-coded. For example, the default value of `2 ** -8` means that on
      average, one 256th of all values will use the Golomb code.
    likelihood_bound: Float. If positive, the returned likelihood values are
      ensured to be greater than or equal to this value. This prevents very
      large gradients with a typical entropy loss (defaults to 1e-9).
    range_coder_precision: Integer, between 1 and 16. The precision of the range
      coder used for compression and decompression. This trades off computation
      speed with compression efficiency, where 16 is the slowest but most
      efficient setting. Choosing lower values may increase the average
      codelength slightly compared to the estimated entropies.
    data_format: Either `'channels_first'` or `'channels_last'` (default).
    trainable: Boolean. Whether the layer should be trained.
    name: String. The name of the layer.
    dtype: `DType` of the layer's inputs, parameters, returned likelihoods, and
      outputs during training. Default of `None` means to use the type of the
      first input.

  Read-only properties:
    init_scale: See above.
    filters: See above.
    tail_mass: See above.
    likelihood_bound: See above.
    range_coder_precision: See above.
    data_format: See above.
    name: String. See above.
    dtype: See above.
    trainable_variables: List of trainable variables.
    non_trainable_variables: List of non-trainable variables.
    variables: List of all variables of this layer, trainable and non-trainable.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer. Always contains exactly one
      auxiliary loss, which must be added to the training loss.

  Mutable properties:
    trainable: Boolean. Whether the layer should be trained.
    input_spec: Optional `InputSpec` object specifying the constraints on inputs
      that can be accepted by the layer.
  """

  def __init__(self, init_scale=10, filters=(3, 3, 3),
               data_format="channels_last", **kwargs):
    super(EntropyBottleneck, self).__init__(**kwargs)
    self._init_scale = float(init_scale)
    self._filters = tuple(int(f) for f in filters)
    self._data_format = str(data_format)
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

    if self.data_format not in ("channels_first", "channels_last"):
      raise ValueError("Unknown data format: '{}'.".format(self.data_format))

  @property
  def init_scale(self):
    return self._init_scale

  @property
  def filters(self):
    return self._filters

  @property
  def data_format(self):
    return self._data_format

  def _channel_axis(self, ndim):
    return {"channels_first": 1, "channels_last": ndim - 1}[self.data_format]

  def _get_input_dims(self):
    """Returns a few useful numbers related to input dimensionality.

    Returns:
      ndim: Integer. Number of input dimensions including batch.
      channel_axis: Integer. Index of dimension that enumerates channels.
      channels: Integer. Number of channels in inputs.
      input_slices: Tuple of slices. Can be used as an index to expand a vector
        to input dimensions, where the vector now runs across channels.
    """
    ndim = self.input_spec.ndim
    channel_axis = self._channel_axis(ndim)
    channels = self.input_spec.axes[channel_axis]
    # Tuple of slices for expanding tensors to input shape.
    input_slices = ndim * [None]
    input_slices[channel_axis] = slice(None)
    input_slices = tuple(input_slices)
    return ndim, channel_axis, channels, input_slices

  def _logits_cumulative(self, inputs, stop_gradient):
    """Evaluate logits of the cumulative densities.

    Arguments:
      inputs: The values at which to evaluate the cumulative densities, expected
        to be a `Tensor` of shape `(channels, 1, batch)`.
      stop_gradient: Boolean. Whether to add `tf.stop_gradient` calls so
        that the gradient of the output with respect to the density model
        parameters is disconnected (the gradient with respect to `inputs` is
        left untouched).

    Returns:
      A `Tensor` of the same shape as `inputs`, containing the logits of the
      cumulative densities evaluated at the given inputs.
    """
    logits = inputs

    for i in range(len(self.filters) + 1):
      matrix = self._matrices[i]
      if stop_gradient:
        matrix = tf.stop_gradient(matrix)
      logits = tf.linalg.matmul(matrix, logits)

      bias = self._biases[i]
      if stop_gradient:
        bias = tf.stop_gradient(bias)
      logits += bias

      if i < len(self._factors):
        factor = self._factors[i]
        if stop_gradient:
          factor = tf.stop_gradient(factor)
        logits += factor * tf.math.tanh(logits)

    return logits

  def build(self, input_shape):
    """Builds the entropy model.

    Creates the variables for the network modeling the densities, creates the
    auxiliary loss estimating the median and tail quantiles of the densities,
    and then uses that to create the probability mass functions and the discrete
    cumulative density functions used by the range coder.

    Arguments:
      input_shape: Shape of the input tensor, used to get the number of
        channels.

    Raises:
      ValueError: if `input_shape` doesn't specify the length of the channel
        dimension.
    """
    input_shape = tf.TensorShape(input_shape)
    channel_axis = self._channel_axis(input_shape.ndims)
    channels = input_shape[channel_axis].value
    if channels is None:
      raise ValueError("The channel dimension of the inputs must be defined.")
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=input_shape.ndims, axes={channel_axis: channels})
    filters = (1,) + self.filters + (1,)
    scale = self.init_scale ** (1 / (len(self.filters) + 1))

    # Create variables.
    self._matrices = []
    self._biases = []
    self._factors = []
    for i in range(len(self.filters) + 1):
      init = np.log(np.expm1(1 / scale / filters[i + 1]))
      matrix = self.add_variable(
          "matrix_{}".format(i), dtype=self.dtype,
          shape=(channels, filters[i + 1], filters[i]),
          initializer=tf.initializers.constant(init))
      matrix = tf.nn.softplus(matrix)
      self._matrices.append(matrix)

      bias = self.add_variable(
          "bias_{}".format(i), dtype=self.dtype,
          shape=(channels, filters[i + 1], 1),
          initializer=tf.initializers.random_uniform(-.5, .5))
      self._biases.append(bias)

      if i < len(self.filters):
        factor = self.add_variable(
            "factor_{}".format(i), dtype=self.dtype,
            shape=(channels, filters[i + 1], 1),
            initializer=tf.initializers.zeros())
        factor = tf.math.tanh(factor)
        self._factors.append(factor)

    # To figure out what range of the densities to sample, we need to compute
    # the quantiles given by `tail_mass / 2` and `1 - tail_mass / 2`. Since we
    # can't take inverses of the cumulative directly, we make it an optimization
    # problem:
    # `quantiles = argmin(|logit(cumulative) - target|)`
    # where `target` is `logit(tail_mass / 2)` or `logit(1 - tail_mass / 2)`.
    # Taking the logit (inverse of sigmoid) of the cumulative makes the
    # representation of the right target more numerically stable.

    # Numerically stable way of computing logits of `tail_mass / 2`
    # and `1 - tail_mass / 2`.
    target = np.log(2 / self.tail_mass - 1)
    # Compute lower and upper tail quantile as well as median.
    target = tf.constant([-target, 0, target], dtype=self.dtype)

    def quantiles_initializer(shape, dtype=None, partition_info=None):
      del partition_info  # unused
      assert tuple(shape[1:]) == (1, 3)
      init = tf.constant(
          [[[-self.init_scale, 0, self.init_scale]]], dtype=dtype)
      return tf.tile(init, (shape[0], 1, 1))

    quantiles = self.add_variable(
        "quantiles", shape=(channels, 1, 3), dtype=self.dtype,
        initializer=quantiles_initializer)
    logits = self._logits_cumulative(quantiles, stop_gradient=True)
    loss = tf.math.reduce_sum(abs(logits - target))
    self.add_loss(loss, inputs=None)

    # Quantize such that the median coincides with the center of a bin.
    medians = quantiles[:, 0, 1]
    self._medians = tf.stop_gradient(medians)

    # Largest distance observed between lower tail quantile and median, and
    # between median and upper tail quantile.
    minima = medians - quantiles[:, 0, 0]
    minima = tf.cast(tf.math.ceil(minima), tf.int32)
    minima = tf.math.maximum(minima, 0)
    maxima = quantiles[:, 0, 2] - medians
    maxima = tf.cast(tf.math.ceil(maxima), tf.int32)
    maxima = tf.math.maximum(maxima, 0)

    # PMF starting positions and lengths.
    self._offset = -minima
    pmf_start = medians - tf.cast(minima, self.dtype)
    pmf_length = maxima + minima + 1

    # Sample the densities in the computed ranges, possibly computing more
    # samples than necessary at the upper end.
    max_length = tf.math.reduce_max(pmf_length)
    samples = tf.range(tf.cast(max_length, self.dtype), dtype=self.dtype)
    samples += pmf_start[:, None, None]

    half = tf.constant(.5, dtype=self.dtype)
    # We strip the sigmoid from the end here, so we can use the special rule
    # below to only compute differences in the left tail of the sigmoid.
    # This increases numerical stability (see explanation in `call`).
    lower = self._logits_cumulative(samples - half, stop_gradient=True)
    upper = self._logits_cumulative(samples + half, stop_gradient=True)
    # Flip signs if we can move more towards the left tail of the sigmoid.
    sign = -tf.math.sign(tf.math.add_n([lower, upper]))
    pmf = abs(tf.math.sigmoid(sign * upper) - tf.math.sigmoid(sign * lower))
    pmf = pmf[:, 0, :]

    # Compute out-of-range (tail) masses.
    tail_mass = tf.math.add_n([
        tf.math.sigmoid(lower[:, 0, :1]),
        tf.math.sigmoid(-upper[:, 0, -1:]),
    ])

    # Construct a valid CDF initializer, so that we can run the model without
    # error even on the zeroth training step.
    def cdf_initializer(shape, dtype=None, partition_info=None):
      del shape, partition_info  # unused
      assert dtype == tf.int32
      fill = tf.constant(.5, dtype=self.dtype)
      prob = tf.fill((channels, 2), fill)
      cdf = range_coding_ops.pmf_to_quantized_cdf(
          prob, precision=self.range_coder_precision)
      return tf.placeholder_with_default(cdf, shape=(channels, None))

    # We need to supply an initializer without fully defined static shape
    # here, or the variable will return the wrong dynamic shape later. A
    # placeholder with default gets the trick done (see initializer above).
    quantized_cdf = self.add_variable(
        "quantized_cdf",
        shape=(channels, None),
        dtype=tf.int32,
        trainable=False,
        initializer=cdf_initializer)
    cdf_length = self.add_variable(
        "cdf_length", shape=(channels,), dtype=tf.int32, trainable=False,
        initializer=tf.initializers.constant(3))
    # Works around a weird TF issue with reading variables inside a loop.
    self._quantized_cdf = tf.identity(quantized_cdf)
    self._cdf_length = tf.identity(cdf_length)

    update_cdf = tf.assign(
        quantized_cdf,
        self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length),
        validate_shape=False)
    update_length = tf.assign(
        cdf_length,
        pmf_length + 2)
    update_op = tf.group(update_cdf, update_length)
    self.add_update(update_op, inputs=None)

    super(EntropyBottleneck, self).build(input_shape)

  def _quantize(self, inputs, mode):
    # Add noise or quantize (and optionally dequantize in one step).
    half = tf.constant(.5, dtype=self.dtype)
    _, _, _, input_slices = self._get_input_dims()

    if mode == "noise":
      noise = tf.random.uniform(tf.shape(inputs), -half, half)
      return tf.math.add_n([inputs, noise])

    medians = self._medians[input_slices]
    outputs = tf.math.floor(inputs + (half - medians))

    if mode == "dequantize":
      outputs = tf.cast(outputs, self.dtype)
      return outputs + medians
    else:
      assert mode == "symbols", mode
      outputs = tf.cast(outputs, tf.int32)
      return outputs

  def _dequantize(self, inputs, mode):
    _, _, _, input_slices = self._get_input_dims()
    medians = self._medians[input_slices]
    outputs = tf.cast(inputs, self.dtype)
    return outputs + medians

  def _likelihood(self, inputs):
    ndim, channel_axis, _, _ = self._get_input_dims()
    half = tf.constant(.5, dtype=self.dtype)

    # Convert to (channels, 1, batch) format by commuting channels to front
    # and then collapsing.
    order = list(range(ndim))
    order.pop(channel_axis)
    order.insert(0, channel_axis)
    inputs = tf.transpose(inputs, order)
    shape = tf.shape(inputs)
    inputs = tf.reshape(inputs, (shape[0], 1, -1))

    # Evaluate densities.
    # We can use the special rule below to only compute differences in the left
    # tail of the sigmoid. This increases numerical stability: sigmoid(x) is 1
    # for large x, 0 for small x. Subtracting two numbers close to 0 can be done
    # with much higher precision than subtracting two numbers close to 1.
    lower = self._logits_cumulative(inputs - half, stop_gradient=False)
    upper = self._logits_cumulative(inputs + half, stop_gradient=False)
    # Flip signs if we can move more towards the left tail of the sigmoid.
    sign = -tf.math.sign(tf.math.add_n([lower, upper]))
    sign = tf.stop_gradient(sign)
    likelihood = abs(
        tf.math.sigmoid(sign * upper) - tf.math.sigmoid(sign * lower))

    # Convert back to input tensor shape.
    order = list(range(1, ndim))
    order.insert(channel_axis, 0)
    likelihood = tf.reshape(likelihood, shape)
    likelihood = tf.transpose(likelihood, order)

    return likelihood

  def _prepare_indexes(self, shape, channels=None):
    shape = tf.convert_to_tensor(shape)

    if not self.built:
      if not (shape.shape.is_fully_defined() and shape.shape.ndims == 1):
        raise ValueError("`shape` must be a vector with known length.")
      ndim = shape.shape[0].value + 1
      channel_axis = self._channel_axis(ndim)
      input_shape = ndim * [None]
      input_shape[channel_axis] = channels
      self.build(input_shape)

    _, channel_axis, channels, input_slices = self._get_input_dims()

    # TODO(jonycgn, ssjhv): Investigate broadcasting.
    indexes = tf.range(channels, dtype=tf.int32)
    indexes = tf.cast(indexes, tf.int32)
    tiles = tf.concat(
        [shape[:channel_axis - 1], [1], shape[channel_axis:]], axis=0)
    indexes = tf.tile(indexes[input_slices[1:]], tiles)

    return indexes

  # Just giving a more useful docstring.
  def decompress(self, strings, shape, channels=None):
    """Decompress values from their compressed string representations.

    Arguments:
      strings: A string `Tensor` vector containing the compressed data.
      shape: A `Tensor` vector of int32 type. Contains the shape of the tensor
        to be decompressed, excluding the batch dimension.
      channels: Integer. Specifies the number of channels statically. Needs only
        be set if the layer hasn't been built yet (i.e., this is the first input
        it receives).

    Returns:
      The decompressed `Tensor`. Its shape will be equal to `shape` prepended
      with the batch dimension from `strings`.

    Raises:
      ValueError: If the length of `shape` isn't available at graph construction
        time.
    """
    return super(EntropyBottleneck, self).decompress(
        strings, shape=shape, channels=channels)


class SymmetricConditional(EntropyModel):
  """Symmetric conditional entropy model.

  Arguments:
    scale: `Tensor`, the scale parameters for the conditional distributions.
    scale_table: Iterable of positive floats. It's optimal to choose the scales
      in a logarithmic way. For each predicted scale, the next greater entry in
      the table is selected during compression.
    scale_bound: Float. Lower bound for scales. Any values in `scale` smaller
      than this value are set to this value to prevent non-positive scales. By
      default (or when set to `None`), uses the smallest value in `scale_table`.
      To disable, set to 0.
    mean: `Tensor`, the mean parameters for the conditional distributions. If
      `None`, the mean is assumed to be zero.
    indexes: `Tensor` of type `int32` or `None`. Can be used to override the
      selection of indexes based on `scale`. Only affects compression and
      decompression.
    tail_mass: Float, between 0 and 1. Values occurring in the tails of the
      distributions will not be encoded with range coding, but using a
      Golomb-like code. `tail_mass` determines the amount of probability mass in
      the tails which will be Golomb-coded. For example, the default value of
      `2 ** -8` means that on average, one 256th of all values will use the
      Golomb code.
    likelihood_bound: Float. If positive, the returned likelihood values are
      ensured to be greater than or equal to this value. This prevents very
      large gradients with a typical entropy loss (defaults to 1e-9).
    range_coder_precision: Integer, between 1 and 16. The precision of the range
      coder used for compression and decompression. This trades off computation
      speed with compression efficiency, where 16 is the slowest but most
      efficient setting. Choosing lower values may increase the average
      codelength slightly compared to the estimated entropies.
    data_format: Either `'channels_first'` or `'channels_last'` (default).
    trainable: Boolean. Whether the layer should be trained.
    name: String. The name of the layer.
    dtype: `DType` of the layer's inputs, parameters, returned likelihoods, and
      outputs during training. Default of `None` means to use the type of the
      first input.

  Read-only properties:
    scale: See above.
    scale_table: See above.
    scale_bound: See above.
    mean: See above.
    indexes: `Tensor` of type `int32`, giving the indexes into the scale table
      for each input element. If not overridden in the constructor, they
      correspond to the table entry with the smallest scale just larger than
      each predicted scale.
    tail_mass: See above.
    likelihood_bound: See above.
    range_coder_precision: See above.
    data_format: See above.
    name: String. See above.
    dtype: See above.
    trainable_variables: List of trainable variables.
    non_trainable_variables: List of non-trainable variables.
    variables: List of all variables of this layer, trainable and non-trainable.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer.

  Mutable properties:
    trainable: Boolean. Whether the layer should be trained.
    input_spec: Optional `InputSpec` object specifying the constraints on inputs
      that can be accepted by the layer.
  """

  def __init__(self, scale, scale_table,
               scale_bound=None, mean=None, indexes=None, **kwargs):
    super(SymmetricConditional, self).__init__(**kwargs)
    self._scale = tf.convert_to_tensor(scale)
    input_shape = self.scale.shape
    self._scale_table = tuple(sorted(float(s) for s in scale_table))
    if any(s <= 0 for s in self.scale_table):
      raise ValueError("`scale_table` must be an iterable of positive numbers.")
    self._scale_bound = None if scale_bound is None else float(scale_bound)
    self._mean = None if mean is None else tf.convert_to_tensor(mean)
    if indexes is not None:
      self._indexes = tf.convert_to_tensor(indexes)
      if self.indexes.dtype != tf.int32:
        raise ValueError("`indexes` must have `int32` dtype.")
      input_shape = input_shape.merge_with(self.indexes.shape)
    if input_shape.ndims is None:
      raise ValueError(
          "Number of dimensions of `scale` or `indexes` must be known.")
    self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

  @property
  def scale(self):
    return self._scale

  @property
  def scale_table(self):
    return self._scale_table

  @property
  def scale_bound(self):
    return self._scale_bound

  @property
  def mean(self):
    return self._mean

  @property
  def indexes(self):
    return self._indexes

  def _standardized_cumulative(self, inputs):
    """Evaluate the standardized cumulative density.

    Note: This function should be optimized to give the best possible numerical
    accuracy for negative input values.

    Arguments:
      inputs: `Tensor`. The values at which to evaluate the cumulative density.

    Returns:
      A `Tensor` of the same shape as `inputs`, containing the cumulative
      density evaluated at the given inputs.
    """
    raise NotImplementedError("Must inherit from SymmetricConditional.")

  def _standardized_quantile(self, quantile):
    """Evaluate the standardized quantile function.

    This returns the inverse of the standardized cumulative function for a
    scalar.

    Arguments:
      quantile: Float. The values at which to evaluate the quantile function.

    Returns:
      A float giving the inverse CDF value.
    """
    raise NotImplementedError("Must inherit from SymmetricConditional.")

  def build(self, input_shape):
    """Builds the entropy model.

    This function precomputes the quantized CDF table based on the scale table.
    This can be done at graph construction time. Then, it creates the graph for
    computing the indexes into that table based on the scale tensor, and then
    uses this index tensor to determine the starting positions of the PMFs for
    each scale.

    Arguments:
      input_shape: Shape of the input tensor.

    Raises:
      ValueError: If `input_shape` doesn't specify number of input dimensions.
    """
    input_shape = tf.TensorShape(input_shape)
    input_shape.assert_is_compatible_with(self.input_spec.shape)

    scale_table = tf.constant(self.scale_table, dtype=self.dtype)

    # Lower bound scales. We need to do this here, and not in __init__, because
    # the dtype may not yet be known there.
    if self.scale_bound is None:
      self._scale = math_ops.lower_bound(self._scale, scale_table[0])
    elif self.scale_bound > 0:
      self._scale = math_ops.lower_bound(self._scale, self.scale_bound)

    multiplier = -self._standardized_quantile(self.tail_mass / 2)
    pmf_center = np.ceil(np.array(self.scale_table) * multiplier).astype(int)
    pmf_length = 2 * pmf_center + 1
    max_length = np.max(pmf_length)

    # This assumes that the standardized cumulative has the property
    # 1 - c(x) = c(-x), which means we can compute differences equivalently in
    # the left or right tail of the cumulative. The point is to only compute
    # differences in the left tail. This increases numerical stability: c(x) is
    # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
    # done with much higher precision than subtracting two numbers close to 1.
    samples = abs(np.arange(max_length, dtype=int) - pmf_center[:, None])
    samples = tf.constant(samples, dtype=self.dtype)
    samples_scale = tf.expand_dims(scale_table, 1)
    upper = self._standardized_cumulative((.5 - samples) / samples_scale)
    lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
    pmf = upper - lower

    # Compute out-of-range (tail) masses.
    tail_mass = 2 * lower[:, :1]

    def cdf_initializer(shape, dtype=None, partition_info=None):
      del partition_info  # unused
      assert tuple(shape) == (len(pmf_length), max_length + 2)
      assert dtype == tf.int32
      return self._pmf_to_cdf(
          pmf, tail_mass,
          tf.constant(pmf_length, dtype=tf.int32), max_length)

    quantized_cdf = self.add_variable(
        "quantized_cdf", shape=(len(pmf_length), max_length + 2),
        initializer=cdf_initializer, dtype=tf.int32, trainable=False)
    cdf_length = self.add_variable(
        "cdf_length", shape=(len(pmf_length),),
        initializer=tf.initializers.constant(pmf_length + 2),
        dtype=tf.int32, trainable=False)
    # Works around a weird TF issue with reading variables inside a loop.
    self._quantized_cdf = tf.identity(quantized_cdf)
    self._cdf_length = tf.identity(cdf_length)

    # Now, if they haven't been overridden, compute the indexes into the table
    # for each of the passed-in scales.
    if not hasattr(self, "_indexes"):
      # Prevent tensors from bouncing back and forth between host and GPU.
      with tf.device("/cpu:0"):
        fill = tf.constant(
            len(self.scale_table) - 1, dtype=tf.int32)
        initializer = tf.fill(tf.shape(self.scale), fill)

        def loop_body(indexes, scale):
          return indexes - tf.cast(self.scale <= scale, tf.int32)

        self._indexes = tf.foldr(
            loop_body, scale_table[:-1],
            initializer=initializer, back_prop=False, name="compute_indexes")

    self._offset = tf.constant(-pmf_center, dtype=tf.int32)

    super(SymmetricConditional, self).build(input_shape)

  def _quantize(self, inputs, mode):
    # Add noise or quantize (and optionally dequantize in one step).
    half = tf.constant(.5, dtype=self.dtype)

    if mode == "noise":
      noise = tf.random.uniform(tf.shape(inputs), -half, half)
      return tf.math.add_n([inputs, noise])

    outputs = inputs
    if self.mean is not None:
      outputs -= self.mean
    outputs = tf.math.floor(outputs + half)

    if mode == "dequantize":
      if self.mean is not None:
        outputs += self.mean
      return outputs
    else:
      assert mode == "symbols", mode
      outputs = tf.cast(outputs, tf.int32)
      return outputs

  def _dequantize(self, inputs, mode):
    assert mode == "dequantize"
    outputs = tf.cast(inputs, self.dtype)
    if self.mean is not None:
      outputs += self.mean
    return outputs

  def _likelihood(self, inputs):
    values = inputs
    if self.mean is not None:
      values -= self.mean

    # This assumes that the standardized cumulative has the property
    # 1 - c(x) = c(-x), which means we can compute differences equivalently in
    # the left or right tail of the cumulative. The point is to only compute
    # differences in the left tail. This increases numerical stability: c(x) is
    # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
    # done with much higher precision than subtracting two numbers close to 1.
    values = abs(values)
    upper = self._standardized_cumulative((.5 - values) / self.scale)
    lower = self._standardized_cumulative((-.5 - values) / self.scale)
    likelihood = upper - lower

    return likelihood

  def _prepare_indexes(self, shape=None):
    del shape  # unused
    if not self.built:
      self.build(self.input_spec.shape)
    return self.indexes

  # Just giving a more useful docstring.
  def decompress(self, strings):  # pylint:disable=useless-super-delegation
    """Decompress values from their compressed string representations.

    Arguments:
      strings: A string `Tensor` vector containing the compressed data.

    Returns:
      The decompressed `Tensor`.
    """
    return super(SymmetricConditional, self).decompress(strings)


class GaussianConditional(SymmetricConditional):
  """Conditional Gaussian entropy model."""

  def _standardized_cumulative(self, inputs):
    half = tf.constant(.5, dtype=self.dtype)
    const = tf.constant(-(2 ** -0.5), dtype=self.dtype)
    # Using the complementary error function maximizes numerical precision.
    return half * tf.math.erfc(const * inputs)

  def _standardized_quantile(self, quantile):
    return scipy.stats.norm.ppf(quantile)


class LogisticConditional(SymmetricConditional):
  """Conditional logistic entropy model."""

  def _standardized_cumulative(self, inputs):
    return tf.math.sigmoid(inputs)

  def _standardized_quantile(self, quantile):
    return scipy.stats.logistic.ppf(quantile)


class LaplacianConditional(SymmetricConditional):
  """Conditional Laplacian entropy model."""

  def _standardized_cumulative(self, inputs):
    exp = tf.math.exp(-abs(inputs))
    return tf.where(inputs > 0, 2 - exp, exp) / 2

  def _standardized_quantile(self, quantile):
    return scipy.stats.laplace.ppf(quantile)
