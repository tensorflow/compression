This project contains data compression ops and layers for TensorFlow. The
project website is at
[tensorflow.github.io/compression](https://tensorflow.github.io/compression).

You can use this library to build your own ML models with optimized lossy
data compression built in. It's useful to find storage-efficient representations
of your data (features, examples, images, etc.) while only sacrificing a tiny
fraction of model performance. It can compress any floating point tensor to a
much smaller sequence of bits.

Specifically, the
[EntropyBottleneck class](https://tensorflow.github.io/compression/docs/entropy_bottleneck.html)
in this library simplifies the process of designing rate–distortion optimized
codes. During training, it acts like a likelihood model. Once training is
completed, it encodes floating point tensors into optimal bit sequences by
automating the design of probability tables and calling a range coder
implementation behind the scenes.

For an introduction to lossy data compression with machine learning, take a look
at @jonycgn's
[talk on Learned Image Compression](https://www.youtube.com/watch?v=x_q7cZviXkY).

# Quick start

**Please note**: You need TensorFlow 1.9 (or the master branch as of May 2018)
or later installed.

Clone the repository to a filesystem location of your choice, or download the
ZIP file and unpack it. Then include the root directory in your `PYTHONPATH`
environment variable:

```bash
cd <target directory>
git clone https://github.com/tensorflow/compression.git tensorflow_compression
export PYTHONPATH="$PWD/tensorflow_compression:$PYTHONPATH"
```

To make sure the library imports succeed, try running the unit tests:

```bash
cd tensorflow_compression
for i in tensorflow_compression/python/*/*_test.py; do
  python $i
done
```

We recommend importing the library from your Python code as follows:

```python
import tensorflow as tf
import tensorflow_compression as tfc
```

## Example model

The [examples directory](https://github.com/tensorflow/compression/tree/master/examples)
contains an implementation of the image compression model described
in:

> "End-to-end optimized image compression"<br />
> J. Ballé, V. Laparra, E. P. Simoncelli<br />
> https://arxiv.org/abs/1611.01704

To see a list of options, change to the directory and run:

```bash
cd examples
python bls2017.py -h
```

To train the model, you need to supply it with a dataset of RGB training images.
They should be provided in PNG format. Training can be as simple as the
following command:

```bash
python bls2017.py -v --train_glob="images/*.png" train
```

This will use the default settings. The most important parameter is `--lambda`,
which controls the trade-off between bitrate and distortion that the model will
be optimized for. The number of channels per layer is important, too: models
tuned for higher bitrates (or, equivalently, lower distortion) tend to require
transforms with a greater approximation capacity (i.e. more channels), so to
optimize performance, you want to make sure that the number of channels is large
enough (or larger). This is described in more detail in:

> "Efficient nonlinear transforms for lossy image compression"<br />
> J. Ballé<br />
> https://arxiv.org/abs/1802.00847

If you wish, you can monitor progress with Tensorboard. To do this, create a
Tensorboard instance in the background before starting the training, then point
your web browser to [port 6006 on your machine](http://localhost:6006):

```bash
tensorboard --logdir=. &
```

When training has finished, the Python script can be used to compress and
decompress images as follows. The same model checkpoint must be accessible to
both commands.

```bash
python bls2017.py [options] compress original.png compressed.bin
python bls2017.py [options] decompress compressed.bin reconstruction.png
```

# Help & documentation

For usage questions and discussions, please head over to our
[Google group](https://groups.google.com/forum/#!forum/tensorflow-compression).

Refer to [the API documentation](https://tensorflow.github.io/compression/docs/api_docs/python/tfc.html)
for a complete description of the Keras layers and TensorFlow ops this package
implements.

There's also an introduction to our `EntropyBottleneck` class
[here](https://tensorflow.github.io/compression/docs/entropy_bottleneck.html),
and a description of the range coding operators
[here](https://tensorflow.github.io/compression/docs/range_coding.html).

# Authors
Johannes Ballé (github: [jonycgn](https://github.com/jonycgn)),
Sung Jin Hwang (github: [ssjhv](https://github.com/ssjhv)), and
Nick Johnston (github: [nmjohn](https://github.com/nmjohn))

Note that this is not an officially supported Google product.
