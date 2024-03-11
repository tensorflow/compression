# TensorFlow Compression Ops

TensorFlow Compression Ops (TFC-ops) contains data compression ops for
TensorFlow.

This is a subset package of TensorFlow Compression (TFC) that contains
C++-implemented TensorFlow operations only. For the full TFC package, please
refer to the [TFC homepage](https://github.com/tensorflow/compression/).


## Documentation & getting help

Refer to [the TFC API
documentation](https://www.tensorflow.org/api_docs/python/tfc) for a complete
description of the functions this package implements.

This subset pockage implements the following functions in the API:

  * `create_range_encoder`
  * `create_range_decoder`
  * `entropy_decode_channel`
  * `entropy_decode_finalize`
  * `entropy_decode_index`
  * `entropy_encode_channel`
  * `entropy_encode_finalize`
  * `entropy_encode_index`
  * `pmf_to_quantized_cdf`
  * `range_decode` (deprecated)
  * `range_encode` (deprecated)
  * `run_length_decode`
  * `run_length_encode`
  * `run_length_gamma_decode` (deprecated)
  * `run_length_gamma_encode` (deprecated)
  * `stochastic_round`

Please post all questions or comments on
[Discussions](https://github.com/tensorflow/compression/discussions). Only file
[Issues](https://github.com/tensorflow/compression/issues) for actual bugs or
feature requests. On Discussions, you may get a faster answer, and you help
other people find the question or answer more easily later.


## Installation

***Note: Precompiled packages are currently only provided for Linux and
Darwin/Mac OS.***

Set up an environment in which you can install precompiled binary Python
packages using the `pip` command. Refer to the
[TensorFlow installation instructions](https://www.tensorflow.org/install/pip)
for more information on how to set up such a Python environment.

The current version of TensorFlow Compression requires TensorFlow 2.

### pip

To install TFC via `pip`, run the following command:

```bash
python -m pip install tensorflow-compression-ops
```

To test that the installation works correctly, you can run the unit tests with:

```bash
python -m tensorflow_compression_ops.tests.all
```

Once the command finishes, you should see a message ```OK (skipped=2)``` or
similar in the last line.

### Colab

You can try out TFC live in a [Colab](https://colab.research.google.com/). The
following command installs the latest version of TFC that is compatible with the
installed TensorFlow version. Run it in a cell before executing your Python
code:

```
%pip install tensorflow-compression-ops~=$(pip show tensorflow | perl -p -0777 -e 's/.*Version: (\d+\.\d+).*/\1.0/sg')
```

Note: The binary packages of TFC are tied to TF with the same minor version
(e.g., TFC 2.9.1 requires TF 2.9.x), and Colab sometimes lags behind a few days
in deploying the latest version of TensorFlow. As a result, using `%pip install
tensorflow-compression-ops` naively might attempt to upgrade TF, which may
create problems.

### Docker

To use a Docker container (e.g. on Windows), be sure to install Docker
(e.g., [Docker Desktop](https://www.docker.com/products/docker-desktop)),
use a [TensorFlow Docker image](https://www.tensorflow.org/install/docker),
and then run the `pip install` command inside the Docker container, not on the
host. For instance, you can use a command line like this:

```bash
docker run tensorflow/tensorflow:latest bash -c \
    "python -m pip install tensorflow-compression-ops &&
     python -m tensorflow_compression_ops.tests.all"
```

This will fetch the TensorFlow Docker image if it's not already cached, install
the pip package and then run the unit tests to confirm that it works.

### Anaconda

It seems that [Anaconda](https://www.anaconda.com/distribution/) ships its own
binary version of TensorFlow which is incompatible with our pip package. To
solve this, always install TensorFlow via `pip` rather than `conda`. For
example, this creates an Anaconda environment with CUDA libraries, and then
installs TensorFlow and TensorFlow Compression Ops:

```bash
conda create --name ENV_NAME python cudatoolkit cudnn
conda activate ENV_NAME
python -m pip install tensorflow-compression-ops
```

Depending on the requirements of the `tensorflow` pip package, you may need to
pin the CUDA libraries to specific versions. If you aren't using a GPU, CUDA is
of course not necessary.


## Usage

We recommend importing the library from your Python code as follows:

```python
import tensorflow as tf
import tensorflow_compression_ops as tfc
```


## Building pip packages

This section describes the necessary steps to build your own pip packages of
TensorFlow Compression Ops. This may be necessary to install it on platforms for
which we don't provide precompiled binaries (currently only Linux and Darwin).

To be compatible with the official TensorFlow pip package, the TFC pip package
must be linked against a matching version of the C libraries. For this reason,
to build the official Linux pip packages, we use [these Docker
images](https://hub.docker.com/r/tensorflow/build) and use the same toolchain
that TensorFlow uses.

Inside the Docker container, the following steps need to be taken:

1. Clone the `tensorflow/compression` repo from GitHub.
2. Run `tensorflow_compression_ops/build_pip_pkg.sh` inside the cloned repo.

For example:

```bash
git clone https://github.com/tensorflow/compression.git /tensorflow_compression
docker run -i --rm \
    -v /tmp/tensorflow_compression_ops:/tmp/tensorflow_compression_ops \
    -v /tensorflow_compression:/tensorflow_compression \
    -w /tensorflow_compression \
    -e "BAZEL_OPT=--config=manylinux_2_17_x86_64" \
    tensorflow/build:latest-python3.10 \
    bash tensorflow_compression_ops/build_pip_pkg.sh /tmp/tensorflow_compression_ops <custom-version>
```

For Darwin, the Docker image and specifying the Bazel config is not necessary.
We just build the package like this (note that you may want to create a clean
Python virtual environment to do this):

```bash
git clone https://github.com/tensorflow/compression.git /tensorflow_compression
cd /tensorflow_compression
BAZEL_OPT="--macos_minimum_os=10.14" bash \
  tensorflow_compression_ops/build_pip_pkg.sh \
  /tmp/tensorflow_compression_ops <custom-version>
```

In both cases, the wheel file is created inside `/tmp/tensorflow_compression_ops`.

To test the created package, first install the resulting wheel file:

```bash
python -m pip install /tmp/tensorflow_compression_ops/tensorflow_compression_ops-*.whl
```

Then run the unit tests (Do not run the tests in the workspace directory where
the `WORKSPACE` file lives. In that case, the Python interpreter would attempt
to import `tensorflow_compression_ops` packages from the source tree, rather
than from the installed package system directory):

```bash
pushd /tmp
python -m tensorflow_compression_ops.tests.all
popd
```

When done, you can uninstall the pip package again:

```bash
python -m pip uninstall tensorflow-compression-ops
```


## Citation

If you use this library for research purposes, please cite:

```
@software{tfc_github,
  author = "Ballé, Johannes and Hwang, Sung Jin and Agustsson, Eirikur",
  title = "{T}ensor{F}low {C}ompression: Learned Data Compression",
  url = "http://github.com/tensorflow/compression",
  version = "2.14.1",
  year = "2024",
}
```

In the above BibTeX entry, names are top contributors sorted by number of
commits. Please adjust version number and year according to the version that was
actually used.

Note that this is not an officially supported Google product.
