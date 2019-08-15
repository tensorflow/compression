<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.same_padding_for_kernel" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.same_padding_for_kernel


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/ops/padding_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Determine correct amount of padding for `same` convolution.

### Aliases:

* `tfc.python.ops.padding_ops.same_padding_for_kernel`


``` python
tfc.same_padding_for_kernel(
    shape,
    corr,
    strides_up=None
)
```



<!-- Placeholder for "Used in" -->

To implement `'same'` convolutions, we first pad the image, and then perform a
`'valid'` convolution or correlation. Given the kernel shape, this function
determines the correct amount of padding so that the output of the convolution
or correlation is the same size as the pre-padded input.

#### Args:


* <b>`shape`</b>: Shape of the convolution kernel (without the channel dimensions).
* <b>`corr`</b>: Boolean. If `True`, assume cross correlation, if `False`, convolution.
* <b>`strides_up`</b>: If this is used for an upsampled convolution, specify the
  strides here. (For downsampled convolutions, specify `(1, 1)`: in that
  case, the strides don't matter.)


#### Returns:

The amount of padding at the beginning and end for each dimension.
