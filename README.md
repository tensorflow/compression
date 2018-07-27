# TensorFlow data compression library

This package contains data compression ops and layers for TensorFlow.

For usage questions and discussions, please head over to our
[Google group](https://groups.google.com/forum/#!forum/tensorflow-compression)!

## Prerequisite

**Please note**: You need TensorFlow 1.9 (or the master branch as of May 2018)
or later.

To make sure the library imports succeed, try running the unit tests:

```bash
for i in tensorflow_compression/python/*/*_test.py; do
  python $i
done
```

## Example model

The [examples directory](https://github.com/tensorflow/compression/tree/master/examples)
directory contains an implementation of the image compression model described
in:

> J. Ballé, V. Laparra, E. P. Simoncelli:
> "End-to-end optimized image compression"
> https://arxiv.org/abs/1611.01704

To see a list of options, change to the directory and run:

```bash
python bls2017.py -h
```

To train the model, you need to supply it with a dataset of RGB training images.
They should be provided in PNG format and must all have the same shape.
Following training, the python script can be used to compress and decompress
images as follows:

```bash
python bls2017.py [options] compress original.png compressed.bin
python bls2017.py [options] decompress compressed.bin reconstruction.png
```

## Documentation

Refer to [the API documentation](https://tensorflow.github.io/compression/docs/api_docs/python/tfc.html)
for a full description of the Keras layers and TensorFlow ops this package
implements.

There's also an introduction to our `EntropyBottleneck` class
[here](https://tensorflow.github.io/compression/docs/entropy_bottleneck.html),
and a description of the range coding ops
[here](https://tensorflow.github.io/compression/docs/range_coding.html).

## Authors
Johannes Ballé (github: [jonycgn](https://github.com/jonycgn)),
Sung Jin Hwang (github: [ssjhv](https://github.com/ssjhv)), and
Nick Johnston (github: [nmjohn](https://github.com/nmjohn))

Note that this is not an officially supported Google product.
