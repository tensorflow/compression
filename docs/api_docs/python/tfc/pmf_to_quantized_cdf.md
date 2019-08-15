<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.pmf_to_quantized_cdf" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.pmf_to_quantized_cdf


<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

Defined in generated file: `python/ops/gen_range_coding_ops.py`



Converts PMF to quantized CDF. This op uses floating-point operations

### Aliases:

* `tfc.python.ops.range_coding_ops.pmf_to_quantized_cdf`


``` python
tfc.pmf_to_quantized_cdf(
    pmf,
    precision,
    name=None
)
```



<!-- Placeholder for "Used in" -->

internally. Therefore the quantized output may not be consistent across multiple
platforms. For entropy encoders and decoders to have the same quantized CDF on
different platforms, the quantized CDF should be produced once and saved, then
the saved quantized CDF should be used everywhere.

After quantization, if PMF does not sum to 2^precision, then some values of PMF
are increased or decreased to adjust the sum to equal to 2^precision.

Note that the input PMF is pre-quantization. The input PMF is not normalized
by this op prior to quantization. Therefore the user is responsible for
normalizing PMF if necessary.

#### Args:


* <b>`pmf`</b>: A `Tensor` of type `float32`.
* <b>`precision`</b>: An `int` that is `>= 1`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` of type `int32`.
