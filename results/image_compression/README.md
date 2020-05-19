# Rate-distortion data for image compression

Subdirectories contain CSV files with rate-distortion (RD) data for different
image compression methods. We include data for standard codecs (JPG, J2K, WebP,
etc.) and many learning-based methods. Quality is measured by PSNR and MS-SSIM.

Note that not all combinations of compression methods, quality metrics, and
evaluation data sets are covered.

### Table of Contents

* [Image Compression Methods](#image-compression-methods)
* [Quality Metrics](#quality-metrics)
* [Data Sets for Evaluation](#data-sets-for-evaluation)

## Image Compression Methods

### Standard (Hand-Engineered) Codecs

*   JPEG (4:2:0)
*   JPEG 2000 ([OpenJPEG](https://www.openjpeg.org) and
               [Kakadu](https://kakadusoftware.com/))
*   [WebP](https://developers.google.com/speed/webp)
*   [BPG](https://bellard.org/bpg/) (4:4:4 and 4:2:0)

### Learning-based Methods

1.  [Channel-wise autoregressive entropy models for learned image compression](http://research.minnen.org/papers/minnen-submitted-icip2020-draft.pdf)\
    David Minnen and Saurabh Singh\
    Int. Conf. on Image Processing (ICIP) 2020

2.  [Context-adaptive Entropy Model for End-to-end Optimized Image Compression](https://openreview.net/forum?id=HyxKIiAqYQ)\
    Jooyoung Lee, Seunghyun Cho, and Seung-Kwon Beack\
    Int. Conf. on Learning Representations (ICLR) 2019

3.  [Joint autoregressive and hierarchical priors for learned image
    compression](https://arxiv.org/abs/1809.02736)\
    David Minnen, Johannes Ballé, and George Toderici\
    Advances in Neural Information Processing Systems (NeurIPS) 2018

4.  [Learning a Code-Space Predictor by Exploiting Intra-Image-Dependencies](http://bmvc2018.org/contents/papers/0491.pdf)\
    Jan P. Klopp, Yu-Chiang Frank Wang, Shao-Yi Chien, and Liang-Gee Chen\
    British Machine Vision Conference (BMVC) 2018

5.  [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436)\
    Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick
    Johnston\
    Int. Conf. on Learning Representations (ICLR) 2018

6.  [Image-dependent local entropy models for image compression with deep
    networks](https://arxiv.org/abs/1805.12295)\
    David Minnen, George Toderici, Saurabh Singh, Sung Jin Hwang, and Michele
    Covell\
    Int. Conf. on Image Processing (ICIP) 2018

7.  [Improved Lossy Image Compression With Priming and Spatially Adaptive Bit
    Rates for Recurrent Networks](https://arxiv.org/abs/1703.10114)\
    Nick Johnston, Damien Vincent, David Minnen, Michele Covell, Saurabh Singh,
    Troy Chinen, Sung Jin Hwang, Joel Shor, and George Toderici\
    IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) 2018

8.  [Real-Time Adaptive Image Compression](https://arxiv.org/abs/1705.05823)\
    Oren Rippel and Lubomir Bourdev\
    International Conference on Machine Learning (ICML) 2017

9.  [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704)\
    Johannes Ballé, Valero Laparra, and Eero P. Simoncelli\
    Int. Conf. on Learning Representations (ICLR) 2017

10.  [Lossy Image Compression with Compressive Autoencoders](https://openreview.net/forum?id=rJiNwv9gg)\
    Lucas Theis, Wenzhe Shi, Andrew Cunningham, and Ferenc Huszár\
    Int. Conf. on Learning Representations (ICLR) 2017

11. [Spatially adaptive image compression using a tiled deep network](https://arxiv.org/abs/1802.02629)\
    David Minnen, George Toderici, Michele Covell, Troy Chinen, Nick Johnston,
    Joel Shor, Sung Jin Hwang, Damien Vincent, and Saurabh Singh\
    Int. Conference on Image Processing (ICIP) 2017

12. [Full Resolution Image Compression with Recurrent Neural Networks](https://arxiv.org/abs/1608.05148)\
    George Toderici, Damien Vincent, Nick Johnston, Sung Jin Hwang, David
    Minnen, Joel Shor, and Michele Covell\
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017

## Quality Metrics

### Peak Signal-to-Noise Ratio (PSNR)

According to
[wikipedia](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio):

> Peak signal-to-noise ratio, often abbreviated PSNR, is an engineering term for
> the ratio between the maximum possible power of a signal and the power of
> corrupting noise that affects the fidelity of its representation. Because many
> signals have a very wide dynamic range, PSNR is usually expressed in terms of
> the logarithmic decibel scale.

PSNR is commonly used to measure image quality even though its correlation with
human preferences is rather low (see the [TID 2013
study](http://www.ponomarenko.info/tid2013.htm)). You can calculate the PSNR
between two images using
[tf.image.psnr()](https://www.tensorflow.org/api_docs/python/tf/image/psnr).

### Multiscale Structural Similarity (MS-SSIM)

Multiscale Structural Similarity (MS-SSIM) is an extension of [structural
similarity (SSIM)](https://en.wikipedia.org/wiki/Structural_similarity) that
adds flexibility by measuring similarity at different spatial scales. It was
developed in 2003 by Wang, Simoncelli, and Bovik
([PDF](https://www.cns.nyu.edu/pub/eero/wang03b.pdf)). MS-SSIM is typically
thought to better match human preferences than PSNR although optimizing directly
for MS-SSIM can lead to objectionable distortion, e.g. blurrier reconstructions
around text and faces.

You can calculate the MS-SSIM score between two images using
[tf.image.ssim_multiscale()](
https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale). Note that
both SSIM and MS-SSIM have a maximum score of 1.0, and very small quantitative
differences can imply very large visual differences. For this reason, we often
graph MS-SSIM as decibels to improve readability using: `ms_ssim_db = -10 *
log10(1 - ms_ssim)`.

### Colorspaces

Many research papers on learned image compression report image quality results
(distortion) averaged over the RGB channels. While mathematically valid, this
approach does not match the sensitivity of the human visual system (e.g. we're
more sensitive to green than blue) and is **not** in line with common practice
in the image processing community.

We provide RGB evaluation results to facilitate comparing against older papers,
but we **strongly recommend** that future papers report results only the
luminance channel (`Y'` in `Y'CbCr`) or by using a 6:1:1 weighted average over
`YCbCr`.

## Data Sets for Evaluation

### Kodak

The Kodak data set is a collection of 24 images with resolution 768x512 (or
512x768). The images are available as PNG files here:
[http://r0k.us/graphics/kodak](http://r0k.us/graphics/kodak)

    @misc{kodak,
      title = "Kodak Lossless True Color Image Suite ({PhotoCD PCD0992})",
      author = "Eastman Kodak",
      url = "http://r0k.us/graphics/kodak",
    }

### Tecnick

The Tecnick data set contains 100 1200x1200 images. It is available for download
here (511 MB):
[https://sourceforge.net/projects/testimages/files/OLD/OLD_SAMPLING/testimages.zip](https://sourceforge.net/projects/testimages/files/OLD/OLD_SAMPLING/testimages.zip)

    @inproceedings{tecnick,
      author = "N. Asuni and A. Giachetti",
      title = "{TESTIMAGES}: A large-scale archive for testing visual devices and basic image processing algorithms {(SAMPLING 1200 RGB set)}",
      year = "2014",
      booktitle = "{STAG}: Smart Tools and Apps for Graphics",
      url = "https://sourceforge.net/projects/testimages/files/OLD/OLD_SAMPLING/testimages.zip",
    }
