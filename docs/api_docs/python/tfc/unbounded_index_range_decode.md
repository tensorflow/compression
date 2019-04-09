
# tfc.unbounded_index_range_decode

### Aliases:

* `tfc.python.ops.range_coding_ops.unbounded_index_range_decode`
* `tfc.unbounded_index_range_decode`

``` python
tfc.unbounded_index_range_decode(
    encoded,
    index,
    cdf,
    cdf_size,
    offset,
    precision,
    overflow_width,
    name=None
)
```



Defined in [`python/ops/range_coding_ops.py`](https://github.com/tensorflow/compression/tree/master/python/ops/range_coding_ops.py).

<!-- Placeholder for "Used in" -->

This is the reverse op of `UnboundedIndexRangeEncode`, and decodes the range

encoded stream `code` into an int32 tensor `decoded`. The other inputs `index`,
`cdf`, `cdf_size`, and `offset` should be the identical tensors passed to the
`UnboundedIndexRangeEncode` op that generated the `decoded` tensor.

Implementation notes:

- If a wrong input was given (e.g., a corrupt `encoded` string, or `cdf` or
`precision` not matching the encoder), the decode is unsuccessful. Because of
potential performance issues, the decoder does not return an error status.

#### Args:

* <b>`encoded`</b>: A `Tensor` of type `string`.
    A scalar string tensor from `UnboundedIndexRangeEncode`.
* <b>`index`</b>: A `Tensor` of type `int32`.
    An int32 tensor of the same shape as `data`.
* <b>`cdf`</b>: A `Tensor` of type `int32`.
    An int32 tensor representing the CDF's of `data`. Each integer is divided
    by `2^precision` to represent a fraction.
* <b>`cdf_size`</b>: A `Tensor` of type `int32`. An int32 tensor.
* <b>`offset`</b>: A `Tensor` of type `int32`. An int32 tensor.
* <b>`precision`</b>: An `int` that is `>= 1`.
    The number of bits for probability quantization. Must be <= 16, and
    must match the precision used by `UnboundedIndexRangeEncode` that produced
    `encoded`.
* <b>`overflow_width`</b>: An `int` that is `>= 1`.
    The bit width of the variable-length overflow code. Must be <=
    precision, and must match the width used by `UnboundedIndexRangeEncode` that
    produced `encoded`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` of type `int32`.
An int32 tensor with the same shape as `index`.