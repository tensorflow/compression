# Introduction

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

## Documentation & getting help

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

## Installation

***Note: Precompiled packages are currently only provided for Linux (Python 2.7,
3.3-3.6) and Darwin/Mac OS (Python 2.7, 3.7). To use these packages on Windows,
consider using a
[TensorFlow Docker image](https://www.tensorflow.org/install/docker) and
installing tensorflow-compression using pip inside the Docker container.***

Set up an environment in which you can install precompiled binary Python
packages using the `pip` command. Refer to the
[TensorFlow installation instructions](https://www.tensorflow.org/install/pip)
for more information on how to set up such a Python environment.

The current version of tensorflow-compression requires TensorFlow 1.15. For
TensorFlow 1.14 or earlier, see our
[previous releases](https://github.com/tensorflow/compression/releases). You can
install TensorFlow from any source. To install it via `pip`, run the following
command:
```bash
pip install tensorflow-gpu==1.15
```
for GPU support, or
```bash
pip install tensorflow==1.15
```
for CPU-only.

Then, run the following command to install the tensorflow-compression pip
package:
```bash
pip install tensorflow-compression
```

To test that the installation works correctly, you can run the unit tests with
```bash
python -m tensorflow_compression.python.all_test
```
Once the command finishes, you should see a message ```OK (skipped=12)``` or
similar in the last line.

### Docker

To use a Docker container (e.g. on Windows), be sure to install Docker
(e.g., [Docker Desktop](https://www.docker.com/products/docker-desktop)),
use a [TensorFlow Docker image](https://www.tensorflow.org/install/docker),
and then run the `pip install` command inside the Docker container, not on the
host. For instance, you can use a command line like this:
```bash
docker run tensorflow/tensorflow:latest-py3 bash -c \
    "pip install tensorflow-compression &&
     python -m tensorflow_compression.python.all_test"
```
This will fetch the latest TensorFlow Docker image, install the pip package
and then run the unit tests to confirm that it works.

### Anaconda

It seems that [Anaconda](https://www.anaconda.com/distribution/) ships its own
binary version of TensorFlow which is incompatible with our pip package. It
also installs Python 3.7 by default, which we currently don't support on Linux.
To solve this, make sure to use Python 3.6 on Linux, and always install
TensorFlow via `pip` rather than `conda`. For example, this creates an Anaconda
environment with Python 3.6 and CUDA libraries, and then installs TensorFlow
and tensorflow-compression with GPU support:
```bash
conda create --name ENV_NAME python=3.6 cudatoolkit=10.0 cudnn
conda activate ENV_NAME
pip install tensorflow-gpu==1.15 tensorflow-compression
```

## Usage

We recommend importing the library from your Python code as follows:

```python
import tensorflow as tf
import tensorflow_compression as tfc
```

### Using a pre-trained model to compress an image

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
python bls2017.py --verbose train --train_glob="images/*.png"
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
python bls2017.py [options] compress original.png compressed.tfci
python bls2017.py [options] decompress compressed.tfci reconstruction.png
```

## Building pip packages

This section describes the necessary steps to build your own pip packages of
tensorflow-compression. This may be necessary to install it on platforms for
which we don't provide precompiled binaries (currently only Linux and Darwin).

We use the Docker image `tensorflow/tensorflow:custom-op-ubuntu16` for building
pip packages for Linux. Note that this is different from
`tensorflow/tensorflow:devel`. To be compatible with the TensorFlow pip package,
the GCC version must match, but `tensorflow/tensorflow:devel` has a different
GCC version installed.

Inside a Docker container from the image, the following steps need to be taken.

1. Install the TensorFlow pip package.
2. Clone the `tensorflow/compression` repo from GitHub.
3. Run `:build_pip_pkg` inside the cloned repo.

For example:
```bash
sudo docker run -v /tmp/tensorflow_compression:/tmp/tensorflow_compression \
    tensorflow/tensorflow:custom-op-ubuntu16 bash -c \
    "pip install tensorflow==1.15 &&
     git clone https://github.com/tensorflow/compression.git
         /tensorflow_compression &&
     cd /tensorflow_compression &&
     bazel run -c opt --copt=-mavx :build_pip_pkg"
```

The wheel file is created inside `/tmp/tensorflow_compression`. Optimization
flags can be passed via `--copt` to the `bazel run` command above.

To test the created package, first install the resulting wheel file:
```bash
pip install /tmp/tensorflow_compression/tensorflow_compression-*.whl
```

Then run the unit tests (Do not run the tests in the workspace directory where
`WORKSPACE` of `tensorflow_compression` repo lives. In that case, the Python
interpreter would attempt to import `tensorflow_compression` packages from the
source tree rather than from the installed package system directory):
```bash
pushd /tmp
python -m tensorflow_compression.python.all_test
popd
```

When done, you can uninstall the pip package again:
```bash
pip uninstall tensorflow-compression
```

To build packages for Darwin (and potentially other platforms), you can follow
the same steps, but the Docker image should not be necessary.

## Authors

Johannes Ballé (github: [jonycgn](https://github.com/jonycgn)), Sung Jin Hwang
(github: [ssjhv](https://github.com/ssjhv)), and Nick Johnston (github:
[nmjohn](https://github.com/nmjohn))

Note that this is not an officially supported Google product.
