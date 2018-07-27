# Entropy bottleneck layer

This layer exposes a high-level interface to model the entropy (the amount of
information conveyed) of the tensor passing through it. During training, this
can be use to impose a (soft) entropy constraint on its activations, limiting
the amount of information flowing through the layer. Note that this is distinct
from other types of bottlenecks, which reduce the dimensionality of the space,
for example. Dimensionality reduction does not limit the amount of information,
and does not enable efficient data compression per se.

After training, this layer can be used to compress any input tensor to a string,
which may be written to a file, and to decompress a file which it previously
generated back to a reconstructed tensor (possibly on a different machine having
access to the same model checkpoint). For this, it uses the range coder
documented in the next section. The entropies estimated during training or
evaluation are approximately equal to the average length of the strings in bits.

The layer implements a flexible probability density model to estimate entropy,
which is described in the appendix of the paper (please cite the paper if you
use this code for scientific work):

> "Variational image compression with a scale hyperprior"<br />
> J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
> https://arxiv.org/abs/1802.01436

The layer assumes that the input tensor is at least 2D, with a batch dimension
at the beginning and a channel dimension as specified by `data_format`. The
layer trains an independent probability density model for each channel, but
assumes that across all other dimensions, the inputs are i.i.d. (independent and
identically distributed). Because the entropy (and hence, average codelength) is
a function of the densities, this assumption may have a direct effect on the
compression performance.

Because data compression always involves discretization, the outputs of the
layer are generally only approximations of its inputs. During training,
discretization is modeled using additive uniform noise to ensure
differentiability. The entropies computed during training are differential
entropies. During evaluation, the data is actually quantized, and the
entropies are discrete (Shannon entropies). To make sure the approximated
tensor values are good enough for practical purposes, the training phase must
be used to balance the quality of the approximation with the entropy, by
adding an entropy term to the training loss, as in the following example.

## Training

Here, we use the entropy bottleneck to compress the latent representation of
an autoencoder. The data vectors `x` in this case are 4D tensors in
`'channels_last'` format (for example, 16x16 pixel grayscale images).

Note that `forward_transform` and `backward_transform` are placeholders and can
be any appropriate artifical neural network. We've found that it generally helps
*not* to use batch normalization, and to sandwich the bottleneck between two
linear transforms or convolutions (i.e. to have no nonlinearities directly
before and after).

```python
# Build autoencoder.
x = tf.placeholder(tf.float32, shape=[None, 16, 16, 1])
y = forward_transform(x)
entropy_bottleneck = EntropyBottleneck()
y_, likelihoods = entropy_bottleneck(y, training=True)
x_ = backward_transform(y_)

# Information content (= predicted codelength) in bits of each batch element
# (note that taking the natural logarithm and dividing by `log(2)` is
# equivalent to taking base-2 logarithms):
bits = tf.reduce_sum(tf.log(likelihoods), axis=(1, 2, 3)) / -np.log(2)

# Squared difference of each batch element:
squared_error = tf.reduce_sum(tf.squared_difference(x, x_), axis=(1, 2, 3))

# The loss is a weighted sum of mean squared error and entropy (average
# information content), where the weight controls the trade-off between
# approximation error and entropy.
main_loss = 0.5 * tf.reduce_mean(squared_error) + tf.reduce_mean(bits)

# Minimize loss and auxiliary loss, and execute update op.
main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
main_step = main_optimizer.minimize(main_loss)
# 1e-3 is a good starting point for the learning rate of the auxiliary loss,
# assuming Adam is used.
aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])
step = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
```

Note that the layer always produces exactly one auxiliary loss and one update
op, which are only significant for compression and decompression. To use the
compression feature, the auxiliary loss must be minimized during or after
training. After that, the update op must be executed at least once. Here, we
simply attach them to the main training step.

## Evaluation

```python
# Build autoencoder.
x = tf.placeholder(tf.float32, shape=[None, 16, 16, 1])
y = forward_transform(x)
y_, likelihoods = EntropyBottleneck()(y, training=False)
x_ = backward_transform(y_)

# Information content (= predicted codelength) in bits of each batch element:
bits = tf.reduce_sum(tf.log(likelihoods), axis=(1, 2, 3)) / -np.log(2)

# Squared difference of each batch element:
squared_error = tf.reduce_sum(tf.squared_difference(x, x_), axis=(1, 2, 3))

# The loss is a weighted sum of mean squared error and entropy (average
# information content), where the weight controls the trade-off between
# approximation error and entropy.
loss = 0.5 * tf.reduce_mean(squared_error) + tf.reduce_mean(bits)
```

To be able to compress the bottleneck tensor and decompress it in a different
session, or on a different machine, you need three items:

- The compressed representations stored as strings.
- The shape of the bottleneck for these string representations as a `Tensor`,
  as well as the number of channels of the bottleneck at graph construction
  time.
- The checkpoint of the trained model that was used for compression. Note:
  It is crucial that the auxiliary loss produced by this layer is minimized
  during or after training, and that the update op is run after training and
  minimization of the auxiliary loss, but *before* the checkpoint is saved.

## Compression

```python
x = tf.placeholder(tf.float32, shape=[None, 16, 16, 1])
y = forward_transform(x)
strings = EntropyBottleneck().compress(y)
shape = tf.shape(y)[1:]
```

## Decompression

```python
strings = tf.placeholder(tf.string, shape=[None])
shape = tf.placeholder(tf.int32, shape=[3])
entropy_bottleneck = EntropyBottleneck(dtype=tf.float32)
y_ = entropy_bottleneck.decompress(strings, shape, channels=5)
x_ = backward_transform(y_)
```
Here, we assumed that the tensor produced by the forward transform has 5
channels.

The above four use cases can also be implemented within the same session (i.e.
on the same `EntropyBottleneck` instance), for testing purposes, etc., by
calling the object more than once.
