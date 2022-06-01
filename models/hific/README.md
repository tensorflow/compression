# High-Fidelity Generative Image Compression

<div align="center">
  <a href='https://hific.github.io'>
  <img src='https://hific.github.io/social/thumb.jpg' width="80%"/>
  </a>
</div>

## [[Demo]](https://hific.github.io) [[Paper]](https://arxiv.org/abs/2006.09965) [[Colab]](https://colab.research.google.com/github/tensorflow/compression/blob/master/models/hific/colab.ipynb)


## Abstract

We extensively study how to combine Generative Adversarial Networks and learned
compression to obtain a state-of-the-art generative lossy compression system. In
particular, we investigate normalization layers, generator and discriminator
architectures, training strategies, as well as perceptual losses. In contrast to
previous work, i) we obtain visually pleasing reconstructions that are
perceptually similar to the input, ii) we operate in a broad range of bitrates,
and iii) our approach can be applied to high-resolution images. We bridge the
gap between rate-distortion-perception theory and practice by evaluating our
approach both quantitatively with various perceptual metrics and a user study.
The study shows that our method is preferred to previous approaches even if they
use more than 2&times; the bitrate.

## Try it out!

[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/tensorflow/compression/blob/master/models/hific/colab.ipynb)

We show some images on the [demo page](https://hific.github.io) and we
release a
[colab](https://colab.research.google.com/github/tensorflow/compression/blob/master/models/hific/colab.ipynb)
for interactively using our models on your own images.

## Running models trained by us locally

Use `tfci.py` for locally running our models to encode and decode images:

```bash
git clone https://github.com/tensorflow/compression
cd compression/models
python tfci.py compress <model> <PNG file>
```

where `model` can be one of `"hific-lo", "hific-mi", "hific-hi"`.

**NOTE**: This is also directly available in the
[colab](https://colab.research.google.com/github/tensorflow/compression/blob/master/models/hific/colab.ipynb)!

## Using the code

To use the code, create a conda environment using Python 3.7, and the packages
listed in
[requirements.txt](https://github.com/tensorflow/compression/blob/master/models/hific/requirements.txt).

**NOTE**: At the moment, we only support CUDA 10.0, Python 3.6-3.7, TensorFlow
1.15, and Tensorflow Compression 1.3. TensorFlow must be installed via pip, not
conda. Unfortunately, newer versions of Tensorflow or Python will not work due
to various constraints in the dependencies and in the TF binary API.

```bash
conda create --name hific python=3.7 cudatoolkit=10.0 cudnn
conda activate hific
pip install -r hific/requirements.txt
```

#### Note on CUDNN Errors

On some of our test machines, the code crashes with one of "Could not create
cudnn handle: CUDNN_STATUS_INTERNAL_ERROR", "terminate called after throwing an
instance of 'std::bad_alloc'",  "Segmentation fault", "Unknown: Failed to get
convolution algorithm. This is probably because cuDNN".

In this case, try setting `TF_FORCE_GPU_ALLOW_GROWTH=true`, e.g.:
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true python -m hific.train ...
```

#### Note on Memory Consumption

This model trains best on a V100. If you get out-of-memory errors
("Resource exhausted: OOM"), try lowering the batch size
(e.g., `--batch_size 6`), or tweak `num_residual_blocks` in `archs.py/Decoder`.

If you get slow training/stalling, try tweaking the `DATASET_NUM_PARALLEL` and
`DATASET_PREFETCH` constants in `model.py`.


### Training your own models.

The architecture is defined in `arch.py`, which is used to build the model from
`model.py`. Our configurations are in `configs.py`.

We release a _simplified_ trainer in `train.py` as a starting point for custom
training. Note that it's using
[coco2014](https://cocodataset.org) from
[tfds](https://www.tensorflow.org/datasets/api_docs/python/tfds) which likely
needs to be adapted to a bigger dataset to obtain good results
(see below).

For the paper, we initialize our GAN models from a MSE+LPIPS checkpoint. To
replicate this, first train a model for MSE + LPIPS only, and then use that as a
starting point:

```bash
# Need to be in the models directory such that hific is a subdirectory.
cd models

# First train a model for MSE+LPIPS:
python -m hific.train --config mselpips --ckpt_dir ckpts/mse_lpips \
    --num_steps 1M --tfds_dataset_name coco2014

# Once that finishes, train a GAN model:
python -m hific.train --config hific --ckpt_dir ckpts/hific \
                --init_autoencoder_from_ckpt_dir ckpts/mse_lpips \
                --num_steps 1M \
                --tfds_dataset_name coco2014
```

Additional helpful arguments are `--tfds_dataset_name`,
and `--tfds_download_dir`, see `--help` for more.

Note that TensorBoard summaries will be saved in `--ckpts` as well. By default,
we create summaries of inputs and reconstructions, which can use a lot of
memory. Disable with `--no-image-summaries`.

To test a trained model, use `evaluate.py` (it also supports the `--tfds_*`
flags):

```bash
python -m hific.evaluate --config hific --ckpt_dir ckpts/hific --out_dir out/ \
                   --tfds_dataset_name coco2014
```

#### Adapting the dataset

You can change to any other TFDS dataset by adapting the `--tfds_dataset_name`,
`--tfds_feature_key`, and `--tfds_download_dir` flags of `train.py`.

Note that when using TFDS, the dataset first has to be downloaded, which can
take time. To do this separately, use the following code snippet:
```python
import tensorflow_datasets as tfds
builder = tfds.builder(TFDS_DATASET_NAME, data_dir=TFDS_DOWNLOAD_DIR)
builder.download_and_prepare()
```

To train on a custom dataset, you can replace the `_get_dataset`
call in `train.py`.

## Metrics

Metrics reported in Figs 4, A10, A11 are available in [data.csv](https://github.com/tensorflow/compression/blob/master/models/hific/data.csv).

## Citation

If you use the work released here for your research, please cite this paper:

```
@article{mentzer2020high,
  title={High-Fidelity Generative Image Compression},
  author={Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur},
  journal={arXiv preprint arXiv:2006.09965},
  year={2020}
}
```

