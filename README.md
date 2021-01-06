# Introduction

This project contains data compression ops and layers for TensorFlow.

You can use this library to build your own ML models with end-to-end optimized
data compression built in. It's useful to find storage-efficient representations
of your data (images, features, examples, etc.) while only sacrificing a tiny
fraction of model performance. It can compress any floating point tensor to a
much smaller sequence of bits.

Specifically, the entropy model classes in this library simplify the process of
designing rate–distortion optimized codes. During training, they act like
likelihood models. Once training is completed, they encode floating point
tensors into optimal bit sequences by automating the design of probability
tables and calling a range coder implementation behind the scenes.

The main novelty of this method over traditional transform coding is the
stochastic minimization of the rate-distortion Lagrangian, and using nonlinear
transforms implemented by neural networks. For an introduction to this, consider
our [paper on nonlinear transform coding](https://arxiv.org/abs/2007.03034), or
watch @jonycgn's [talk on learned image
compression](https://www.youtube.com/watch?v=x_q7cZviXkY).

## Documentation & getting help

Please post all questions or comments in our [Google
group](https://groups.google.com/forum/#!forum/tensorflow-compression). Only
file Github issues for actual bugs or feature requests. If you post to the group
instead, you may get a faster answer, and you help other people find the
question or answer more easily later.

Refer to [the API
documentation](https://tensorflow.github.io/compression/docs/api_docs/python/tfc.html)
for a complete description of the Keras layers and TensorFlow ops this package
implements.
***Note: the API docs have not been updated for the current beta release yet.***

## Installation

***Note: Precompiled packages are currently only provided for Linux and
Darwin/Mac OS. To use these packages on Windows, consider using a [TensorFlow
Docker image](https://www.tensorflow.org/install/docker) and installing
tensorflow-compression using pip inside the Docker container.***

Set up an environment in which you can install precompiled binary Python
packages using the `pip` command. Refer to the
[TensorFlow installation instructions](https://www.tensorflow.org/install/pip)
for more information on how to set up such a Python environment.

The current stable version of TFC (1.3) requires TensorFlow 1.15. The current
beta release of TFC (2.0b2) is built for TensorFlow 2.4. For versions compatible
with TensorFlow 1.14 or earlier, see our [previous
releases](https://github.com/tensorflow/compression/releases).

### pip

To install TF and TFC via `pip`, run the following command:

```bash
pip install tensorflow-gpu==1.15 tensorflow-compression==1.3
```

for the stable release, or

```bash
pip install tensorflow-gpu==2.4 tensorflow-probability==0.12.1 tensorflow-compression==2.0b2
```

for the beta release. If you don't need GPU support, you can drop the `-gpu`
part.

To test that the installation works correctly, you can run the unit tests with
(respectively):

```bash
python -m tensorflow_compression.python.all_test
```

or

```bash
python -m tensorflow_compression.all_tests
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
docker run tensorflow/tensorflow:1.15.0-py3 bash -c \
    "pip install tensorflow-compression==1.3 &&
     python -m tensorflow_compression.python.all_test"
```

or (for the beta version):

```bash
docker run tensorflow/tensorflow:2.4.0 bash -c \
    "pip install tensorflow-probability==0.12.1 tensorflow-compression==2.0b2 &&
     python -m tensorflow_compression.all_tests"
```

This will fetch the TensorFlow Docker image if it's not already cached, install
the pip package and then run the unit tests to confirm that it works.

### Anaconda

It seems that [Anaconda](https://www.anaconda.com/distribution/) ships its own
binary version of TensorFlow which is incompatible with our pip package. To
solve this, always install TensorFlow via `pip` rather than `conda`. For
example, this creates an Anaconda environment with Python 3.6 and CUDA
libraries, and then installs TensorFlow and tensorflow-compression with GPU
support:

```bash
conda create --name ENV_NAME python=3.6 cudatoolkit=10.0 cudnn
conda activate ENV_NAME
pip install tensorflow-gpu==1.15 tensorflow-compression==1.3
```

## Usage

We recommend importing the library from your Python code as follows:

```python
import tensorflow as tf
import tensorflow_compression as tfc
```

### Using a pre-trained model to compress an image

In the
[models directory](https://github.com/tensorflow/compression/tree/master/models),
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
[models directory](https://github.com/tensorflow/compression/tree/master/models)
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

We use the custom-op Docker images (e.g.
`tensorflow/tensorflow:nightly-custom-op-ubuntu16`) for building pip packages
for Linux. Note that this is different from `tensorflow/tensorflow:devel`. To be
compatible with the TensorFlow pip package, the GCC version must match, but
`tensorflow/tensorflow:devel` has a different GCC version installed. For more
information, refer to the [custom-op
instructions](https://github.com/tensorflow/custom-op).

Inside a Docker container from the image, the following steps need to be taken.

1. Clone the `tensorflow-compression` repo from GitHub.
2. Run `:build_pip_pkg` inside the cloned repo.

For example:

```bash
sudo docker run -v /tmp/tensorflow_compression:/tmp/tensorflow_compression \
    tensorflow/tensorflow:nightly-custom-op-ubuntu16 bash -c \
    "git clone https://github.com/tensorflow/compression.git
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
source tree, rather than from the installed package system directory):

```bash
pushd /tmp
python -m tensorflow_compression.all_tests
popd
```

When done, you can uninstall the pip package again:

```bash
pip uninstall tensorflow-compression
```

To build packages for Darwin (and potentially other platforms), you can follow
the same steps, but the Docker image should not be necessary.

## Evaluation

We provide evaluation results for several image compression methods in terms of
different metrics in different colorspaces. Please see the
[results subdirectory](https://github.com/tensorflow/compression/tree/master/results/image_compression)
for more information.

## Authors

* Johannes Ballé (github: [jonycgn](https://github.com/jonycgn))
* Sung Jin Hwang (github: [ssjhv](https://github.com/ssjhv))
* Nick Johnston (github: [nmjohn](https://github.com/nmjohn))
* David Minnen (github: [minnend](https://github.com/minnend))
* Eirikur Agustsson (github: [relational](https://github.com/relational))

Note that this is not an officially supported Google product.
