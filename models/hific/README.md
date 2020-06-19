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
update for interactively using our models on your own images.

## Using the code

In addition to `tensorflow_compression`, you need to install [`compare_gan`](https://github.com/google/compare_gan)
and TensorFlow 1.15:

```bash
pip install -r requirements.txt
```

## Running our models locally

Use `tfci.py` for locally running our models to encode and decode images:

```python
python tfci.py compress <model> <PNG file>
```

where `model` can be one of `"hific-lo", "hific-mi", "hific-hi"`.

## Code

The architecture is defined in `arch.py` , which is used to build the model in
`model.py`. Our configurations are in `configs.py`.

### Training your own models.

We release a _simplified_ trainer in `train.py` as a starting point for custom
training. Note that it's using [LSUN]() from [tfds]() which likely needs to be
adapted to a bigger dataset to obtain state-of-the-art results (see below).

For the paper, we initialize our GAN models from a MSE+LPIPS checkpoint. To
replicate this, first train a model for MSE + LPIPS only, and then use that as a
starting point:

```bash
# First train a model for MSE+LPIPS:
python train.py --config mselpips --ckpt_dir ckpts --num_steps 1M

# Once that finishes, train a GAN model:
python train.py --config hific --ckpt_dir ckpts \
                --init_from ckpts/mselpips --num_steps 1M
```

To test a trained model, use `eval.py`:

```bash
python eval.py --config hific --ckpt_dir ckpts/hific
```

#### Adapting the dataset

You can change to any other TFDS dataset by changing the `tfds_name` flag for
`build_input`. To train on a custom dataset, you can replace the `_get_dataset`
call in `train.py`.

## Citation

If you use the work released here for your research, please cite this paper:

```
@inproceedings{mentzer2020hific,
  title={High-Fidelity Generative Image Compression},
  author={Fabian Mentzer and George Toderici and Michael Tschannen and Eirikur Agustsson},
  year={2020}
}
```

