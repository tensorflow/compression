# TensorFlow Compression

TensorFlow Compression (TFC) contains data compression tools for TensorFlow.

You can use this library to build your own ML models with end-to-end optimized
data compression built in. It's useful to find storage-efficient representations
of your data (images, features, examples, etc.) while only sacrificing a small
fraction of model performance. Take a look at the [lossy data compression
tutorial](https://www.tensorflow.org/tutorials/generative/data_compression) or
the [model compression
tutorial](https://www.tensorflow.org/tutorials/optimization/compression) to get
started.

For a more in-depth introduction from a classical data compression perspective,
consider our [paper on nonlinear transform
coding](https://arxiv.org/abs/2007.03034), or watch @jonycgn's [talk on learned
image compression](https://www.youtube.com/watch?v=x_q7cZviXkY). For an
introduction to lossy data compression from a machine learning perspective, take
a look at @yiboyang's [review paper](https://arxiv.org/abs/2202.06533).

The library contains (see the [API
docs](https://www.tensorflow.org/api_docs/python/tfc) for details):

- Range coding (a.k.a. arithmetic coding) implementations in the form of
  flexible TF ops written in C++. These include an optional "overflow"
  functionality that embeds an Elias gamma code into the range encoded bit
  sequence, making it possible to encode alphabets containing the entire set of
  signed integers rather than just a finite range.

- Entropy model classes which simplify the process of designing rate–distortion
  optimized codes. During training, they act like likelihood models. Once
  training is completed, they encode floating point tensors into optimized bit
  sequences by automating the design of range coding tables and calling the
  range coder implementation behind the scenes.

- Additional TensorFlow functions and Keras layers that are useful in the
  context of learned data compression, such as methods to numerically find
  quantiles of density functions, take expectations with respect to dithering
  noise, convolution layers with more flexible padding options and support for
  reparameterizing kernels and biases in the Fourier domain, and an
  implementation of generalized divisive normalization (GDN).


## Documentation & getting help

Refer to [the API documentation](https://www.tensorflow.org/api_docs/python/tfc)
for a complete description of the classes and functions this package implements.

Please post all questions or comments on
[Discussions](https://github.com/tensorflow/compression/discussions). Only file
[Issues](https://github.com/tensorflow/compression/issues) for actual bugs or
feature requests. On Discussions, you may get a faster answer, and you help
other people find the question or answer more easily later.

## Installation

***Note: Precompiled packages are currently only provided for Linux and
Darwin/Mac OS. To use these packages on Windows, consider installing TensorFlow
using the [instructions for
WSL2](https://www.tensorflow.org/install/pip#windows_1) or using a [TensorFlow
Docker image](https://www.tensorflow.org/install/docker), and then installing
the Linux package.***

Set up an environment in which you can install precompiled binary Python
packages using the `pip` command. Refer to the
[TensorFlow installation instructions](https://www.tensorflow.org/install/pip)
for more information on how to set up such a Python environment.

The current version of TensorFlow Compression requires TensorFlow 2. For
versions compatible with TensorFlow 1, see our [previous
releases](https://github.com/tensorflow/compression/releases).

### pip

To install TFC via `pip`, run the following command:

```bash
pip install tensorflow-compression
```

To test that the installation works correctly, you can run the unit tests with:

```bash
python -m tensorflow_compression.all_tests
```

Once the command finishes, you should see a message ```OK (skipped=29)``` or
similar in the last line.

### Colab

You can try out TFC live in a [Colab](https://colab.research.google.com/). The
following command installs the latest version of TFC that is compatible with the
installed TensorFlow version. Run it in a cell before executing your Python
code:

```
!pip install tensorflow-compression~=$(pip show tensorflow | perl -p -0777 -e 's/.*Version: (\d+\.\d+).*/\1.0/sg')
```

Note: The binary packages of TFC are tied to TF with the same minor version
(e.g., TFC 2.9.1 requires TF 2.9.x), and Colab sometimes lags behind a few days
in deploying the latest version of TensorFlow. As a result, using `pip install
tensorflow-compression` naively might attempt to upgrade TF, which can create
problems.

### Docker

To use a Docker container (e.g. on Windows), be sure to install Docker
(e.g., [Docker Desktop](https://www.docker.com/products/docker-desktop)),
use a [TensorFlow Docker image](https://www.tensorflow.org/install/docker),
and then run the `pip install` command inside the Docker container, not on the
host. For instance, you can use a command line like this:

```bash
docker run tensorflow/tensorflow:latest bash -c \
    "pip install tensorflow-compression &&
     python -m tensorflow_compression.all_tests"
```

This will fetch the TensorFlow Docker image if it's not already cached, install
the pip package and then run the unit tests to confirm that it works.

### Anaconda

It seems that [Anaconda](https://www.anaconda.com/distribution/) ships its own
binary version of TensorFlow which is incompatible with our pip package. To
solve this, always install TensorFlow via `pip` rather than `conda`. For
example, this creates an Anaconda environment with CUDA libraries, and then
installs TensorFlow and TensorFlow Compression:

```bash
conda create --name ENV_NAME python cudatoolkit cudnn
conda activate ENV_NAME
pip install tensorflow-compression
```

Depending on the requirements of the `tensorflow` pip package, you may need to
pin the CUDA libraries to specific versions. If you aren't using a GPU, CUDA is
of course not necessary.

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
contains several implementations of published image compression models to enable
easy experimentation. Note that in order to reproduce published results, more
tuning of the code and training dataset may be necessary. Use the `tfci.py`
script above to access published models.

The following instructions talk about a re-implementation of the model published
in:

> "End-to-end optimized image compression"<br />
> J. Ballé, V. Laparra, E. P. Simoncelli<br />
> https://arxiv.org/abs/1611.01704

Note that the models directory is not contained in the pip package. The models
are meant to be downloaded individually. Download the file `bls2017.py` and run:

```bash
python bls2017.py -h
```

This will list the available command line options for the implementation.
Training can be as simple as the following command:

```bash
python bls2017.py -V train
```

This will use the default settings. Note that unless a custom training dataset
is provided via `--train_glob`, the
[CLIC dataset](https://www.tensorflow.org/datasets/catalog/clic) will be
downloaded using TensorFlow Datasets.

The most important training parameter is `--lambda`, which controls the
trade-off between bitrate and distortion that the model will be optimized for.
The number of channels per layer is important, too: models tuned for higher
bitrates (or, equivalently, lower distortion) tend to require transforms with a
greater approximation capacity (i.e. more channels), so to optimize performance,
you want to make sure that the number of channels is large enough (or larger).
This is described in more detail in:

> "Efficient nonlinear transforms for lossy image compression"<br />
> J. Ballé<br />
> https://arxiv.org/abs/1802.00847

If you wish, you can monitor progress with Tensorboard. To do this, create a
Tensorboard instance in the background before starting the training, then point
your web browser to [port 6006 on your machine](http://localhost:6006):

```bash
tensorboard --logdir=/tmp/train_bls2017 &
```

When training has finished, the Python script saves the trained model to the
directory specified with `--model_path` (by default, `bls2017` in the current
directory) in TensorFlow's `SavedModel` format. The script can then be used to
compress and decompress images as follows. The same saved model must be
accessible to both commands.

```bash
python bls2017.py [options] compress original.png compressed.tfci
python bls2017.py [options] decompress compressed.tfci reconstruction.png
```

## Building pip packages

This section describes the necessary steps to build your own pip packages of
TensorFlow Compression. This may be necessary to install it on platforms for
which we don't provide precompiled binaries (currently only Linux and Darwin).

To be compatible with the official TensorFlow pip package, the TFC pip package
must be linked against a matching version of the C libraries. For this reason,
to build the official Linux pip packages, we use [these Docker
images](https://hub.docker.com/r/tensorflow/build) and use the same toolchain
that TensorFlow uses.

Inside the Docker container, the following steps need to be taken:

1. Clone the `tensorflow/compression` repo from GitHub.
2. Install Python dependencies.
3. Run `:build_pip_pkg` inside the cloned repo.

For example:

```bash
sudo docker run -i --rm -v /tmp/tensorflow_compression:/tmp/tensorflow_compression \
    tensorflow/build:latest-python3.10 bash -c \
    "git clone https://github.com/tensorflow/compression.git /tensorflow_compression &&
     cd /tensorflow_compression &&
     python -m pip install -U pip setuptools wheel &&
     python -m pip install -r requirements.txt &&
     bazel run -c opt --copt=-mavx --crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain :build_pip_pkg -- . /tmp/tensorflow_compression <custom-version>"
```

For Darwin, the Docker image and specifying the toolchain is not necessary. We
just build the package like this (note that you may want to create a clean
Python virtual environment to do this):

```bash
git clone https://github.com/tensorflow/compression.git /tensorflow_compression
cd /tensorflow_compression
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
bazel run -c opt --copt=-mavx --macos_minimum_os=10.14 :build_pip_pkg -- . /tmp/tensorflow_compression <custom-version>"
```

In both cases, the wheel file is created inside `/tmp/tensorflow_compression`.

To test the created package, first install the resulting wheel file:

```bash
pip install /tmp/tensorflow_compression/tensorflow_compression-*.whl
```

Then run the unit tests (Do not run the tests in the workspace directory where
the `WORKSPACE` file lives. In that case, the Python interpreter would attempt
to import `tensorflow_compression` packages from the source tree, rather than
from the installed package system directory):

```bash
pushd /tmp
python -m tensorflow_compression.all_tests
popd
```

When done, you can uninstall the pip package again:

```bash
pip uninstall tensorflow-compression
```

## Evaluation

We provide evaluation results for several image compression methods in terms of
different metrics in different colorspaces. Please see the
[results subdirectory](https://github.com/tensorflow/compression/tree/master/results/image_compression)
for more information.

## Citation

If you use this library for research purposes, please cite:
```
@software{tfc_github,
  author = "Ballé, Johannes and Hwang, Sung Jin and Agustsson, Eirikur",
  title = "{T}ensor{F}low {C}ompression: Learned Data Compression",
  url = "http://github.com/tensorflow/compression",
  version = "2.13.0",
  year = "2023",
}
```
In the above BibTeX entry, names are top contributors sorted by number of
commits. Please adjust version number and year according to the version that was
actually used.

Note that this is not an officially supported Google product.
