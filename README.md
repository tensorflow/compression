# TensorFlow Compression

This project contains data compression ops and layers for TensorFlow. The
project website is at
[tensorflow.github.io/compression](https://tensorflow.github.io/compression).

You can use this library to build your own ML models with end-to-end optimized
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

For an introduction to lossy image compression with machine learning, take a
look at @jonycgn's
[talk on Learned Image Compression](https://www.youtube.com/watch?v=x_q7cZviXkY).

## Quick start

### Installing release 1.1 (stable)

Install TensorFlow 1.13 using one of the methods described in the
[TensorFlow installation instructions](https://www.tensorflow.org/install).

Download the ZIP file for
[release v1.1](https://github.com/tensorflow/compression/releases/tag/v1.1)
and unpack it. Then include its root directory in your `PYTHONPATH`
environment variable:

```bash
cd <target directory>
wget https://github.com/tensorflow/compression/archive/v1.1.zip
unzip v1.1.zip
export PYTHONPATH="$PWD/compression-1.1:$PYTHONPATH"
```

To make sure the library imports succeed, try running the unit tests:

```bash
cd compression-1.1
for i in tensorflow_compression/python/*/*_test.py; do python $i; done
```

### Installing release 1.2b1 (beta)

Set up an environment in which you can install precompiled binary Python
packages using the `pip` command. Refer to the
[TensorFlow installation instructions](https://www.tensorflow.org/install/pip)
for more information on how to set up such a Python environment.

Run the following command to install the binary PIP package:

```bash
pip install tensorflow-compression
```

***Note: for this beta release, we only support Python 2.7 and 3.4 versions on
Linux platforms. We are working on Darwin (Mac) binaries as well. For the time
being, if you need to run the beta release on Mac, we suggest to use Docker
Desktop for Mac, and run the above command inside a container based on the
[TensorFlow docker image](https://www.tensorflow.org/install/docker) for
Python 2.7.***

### Using the library

We recommend importing the library from your Python code as follows:

```python
import tensorflow as tf
import tensorflow_compression as tfc
```

### Using a pre-trained model to compress an image

***Note: you need to have a release >1.1 installed for pre-trained model
support.***

In the
[examples directory](https://github.com/tensorflow/compression/tree/master/examples),
you'll find a python script `tfci.py`. Download the file and run:

```bash
python tfci.py -h
```

This will give you a list of options. Briefly, the command

```bash
python tfci.py compress <model> <PNG file>
```

will compress an image using a pre-trained model and write a file ending in
`.tfci`. Execute `python tfci.py models` to give you a list of supported
pre-trained models. The command

```bash
python tfci.py decompress <TFCI file>
```

will decompress a TFCI file and write a PNG file. By default, an output file
will be named like the input file, only with the appropriate file extension
appended (any existing extensions will not be removed).

### Training your own model

The
[examples directory](https://github.com/tensorflow/compression/tree/master/examples)
contains an implementation of the image compression model described in:

> "End-to-end optimized image compression"<br />
> J. Ballé, V. Laparra, E. P. Simoncelli<br />
> https://arxiv.org/abs/1611.01704

To see a list of options, download the file `bls2017.py` and run:

```bash
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

## Help & documentation

For usage questions and discussions, please head over to our
[Google group](https://groups.google.com/forum/#!forum/tensorflow-compression).

Refer to
[the API documentation](https://tensorflow.github.io/compression/docs/api_docs/python/tfc.html)
for a complete description of the Keras layers and TensorFlow ops this package
implements.

There's also an introduction to our `EntropyBottleneck` class
[here](https://tensorflow.github.io/compression/docs/entropy_bottleneck.html),
and a description of the range coding operators
[here](https://tensorflow.github.io/compression/docs/range_coding.html).

## Building PIP package

This section describes steps to build PIP package.

Docker image `tensorflow/tensorflow:custom-op` is used for PIP package build.
Note that this is different from `tensorflow/tensorflow:devel`. To be compatible
with TensorFlow PIP package, the GCC version must match, but
`tensorflow/tensorflow:devel` has a different GCC version installed.

Inside a Docker container from the image, the following steps need to be done.

1. Install TensorFlow PIP package.
2. Clone `tensorflow-compression` repo from GitHub.
3. Run `:build_pip_pkg` inside the cloned repo.

```bash
sudo docker run -v /tmp/tensorflow_compression:/tmp/tensorflow_compression \
    tensorflow/tensorflow:custom-op \
    bash -c "pip install tensorflow &&
    git clone https://github.com/tensorflow/compression.git
    /tensorflow_compression && cd /tensorflow_compression &&
    bazel run -c opt :build_pip_pkg"
```

The wheel file is created inside `/tmp/tensorflow_compression`. Optimization
flags can be passed via `--copt` to the `bazel run` command above. For example,
`bazel` command above can be `bazel run -c opt --copt=-mavx :build_pip_pkg`.

### Testing PIP package

First install the built wheel file.

```bash
pip install /tmp/tensorflow_compression/tensorflow_compression-*.whl
```

And run tests. (Do not run the tests in the workspace directory where
`WORKSPACE` of `tensorflow_compression` repo lives. In that case, Python
interpreter attempts to import `tensorflow_compression` packages from the source
tree rather than from the installed package system directory.)

```bash
pushd /tmp
python -m tensorflow_compression.python.all_test
popd
```

When done, uninstall the PIP package. The package name is
`tensorflow-compression` with a hyphen(-).

```bash
pip uninstall tensorflow-compression
```

## Authors

Johannes Ballé (github: [jonycgn](https://github.com/jonycgn)), Sung Jin Hwang
(github: [ssjhv](https://github.com/ssjhv)), and Nick Johnston (github:
[nmjohn](https://github.com/nmjohn))

Note that this is not an officially supported Google product.
