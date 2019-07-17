<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.array_fingerprint" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.array_fingerprint

Produces fingerprint of the input data.

### Aliases:

* `tfc.array_fingerprint`
* `tfc.python.ops.range_coding_ops.array_fingerprint`

``` python
tfc.array_fingerprint(
    input,
    name=None
)
```



Defined in generated file: `python/ops/gen_range_coding_ops.py`

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
  Tensor to be fingerprinted.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` of type `int64`. Fingerprint value of input.
