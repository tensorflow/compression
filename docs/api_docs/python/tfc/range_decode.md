<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.range_decode" />
</div>

# tfc.range_decode

``` python
tfc.range_decode(
    encoded,
    shape,
    cdf,
    precision,
    name=None
)
```

Decodes a range-coded `code` into an int32 tensor of shape `shape`.

This is the reverse op of RangeEncode. The shape of the tensor that was encoded
should be known by the caller.

Implementation notes:

- If wrong input was given (e.g., corrupt `encoded` string, or `cdf` or
`precision` do not match encoder), the decode is unsuccessful. Because of
potential performance issues, the decoder does not return error status.

#### Args:

* <b>`encoded`</b>: A `Tensor` of type `string`.
    A scalar string tensor from RangeEncode.
* <b>`shape`</b>: A `Tensor` of type `int32`.
    An int32 1-D tensor representing the shape of the data encoded by
    RangeEncode.
* <b>`cdf`</b>: A `Tensor` of type `int32`.
* <b>`precision`</b>: An `int` that is `>= 1`.
    The number of bits for probability quantization. Must be <= 16, and
    must match the precision used by RangeEncode that produced `encoded`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` of type `int16`. An int16 tensor with shape equal to `shape`.