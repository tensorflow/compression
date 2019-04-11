
# tfc.array_fingerprint

### Aliases:

* `tfc.array_fingerprint`
* `tfc.python.ops.range_coding_ops.array_fingerprint`

``` python
tfc.array_fingerprint(
    input,
    name=None
)
```



Defined in [`python/ops/_range_coding_ops.py`](https://github.com/tensorflow/compression/tree/master/python/ops/_range_coding_ops.py).

<!-- Placeholder for "Used in" -->

Produces fingerprint of the input data.

#### Args:

* <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    Tensor to be fingerprinted.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` of type `int64`. Fingerprint value of input.