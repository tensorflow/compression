<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.unbounded_index_range_encode" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.unbounded_index_range_encode


<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

Defined in generated file: `python/ops/gen_range_coding_ops.py`



Range encodes unbounded integer `data` using an indexed probability table.

### Aliases:

* `tfc.python.ops.range_coding_ops.unbounded_index_range_encode`


``` python
tfc.unbounded_index_range_encode(
    data,
    index,
    cdf,
    cdf_size,
    offset,
    precision,
    overflow_width,
    debug_level=1,
    name=None
)
```



<!-- Placeholder for "Used in" -->

For each value in `data`, the corresponding value in `index` determines which
probability model in `cdf` is used to encode it. The data can be arbitrary
signed integers, where the integer intervals determined by `offset` and
`cdf_size` are modeled using the cumulative distribution functions (CDF) in
`cdf`. Everything else is encoded with a variable length code.

The argument `cdf` is a 2-D tensor and its each row contains a CDF. The argument
`cdf_size` is a 1-D tensor, and its length should be the same as the number of
rows of `cdf`. The values in `cdf_size` denotes the length of CDF vector in the
corresponding row of `cdf`.

For i = 0,1,..., let `m = cdf_size[i]`. Then for j = 0,1,...,m-1,

```
   cdf[..., 0] / 2^precision = Pr(X < 0) = 0
   cdf[..., 1] / 2^precision = Pr(X < 1) = Pr(X <= 0)
   cdf[..., 2] / 2^precision = Pr(X < 2) = Pr(X <= 1)
   ...
   cdf[..., m-1] / 2^precision = Pr(X < m-1) = Pr(X <= m-2).
```

We require that `1 < m <= cdf.shape[1]` and that all elements of `cdf` be in the
closed interval `[0, 2^precision]`.

Arguments `data` and `index` should have the same shape. `data` contains the
values to be encoded. `index` denotes which row in `cdf` should be used to
encode the corresponding value in `data`, and which element in `offset`
determines the integer interval the cdf applies to. Naturally, the elements of
`index` should be in the half-open interval `[0, cdf.shape[0])`.

When a value from `data` is in the interval `[offset[i], offset[i] + m - 2)`,
then the value is range encoded using the CDF values. The last entry in each
CDF (the one at `m - 1`) is an overflow code. When a value from `data` is
outside of the given interval, the overflow value is encoded, followed by a
variable-length encoding of the actual data value.

The encoded output contains neither the shape information of the encoded data
nor a termination symbol. Therefore the shape of the encoded data must be
explicitly provided to the decoder.

#### Implementation notes:



- Because of potential performance issues, the op does not check if `cdf`
satisfies monotonic increase property.

- For the range coder to decode the encoded string correctly, the decoder should
be able to reproduce the internal states of the encoder precisely. Otherwise,
the decoding would fail and once an error occur, all subsequent decoded values
are incorrect. For this reason, the range coder uses integer arithmetics and
avoids using any floating point operations internally, and `cdf` should contain
integers representing quantized probability mass rather than floating points.

#### Args:


* <b>`data`</b>: A `Tensor` of type `int32`. An int32 tensor.
* <b>`index`</b>: A `Tensor` of type `int32`.
  An int32 tensor of the same shape as `data`.
* <b>`cdf`</b>: A `Tensor` of type `int32`.
  An int32 tensor representing the CDF's of `data`. Each integer is divided
  by `2^precision` to represent a fraction.
* <b>`cdf_size`</b>: A `Tensor` of type `int32`. An int32 tensor.
* <b>`offset`</b>: A `Tensor` of type `int32`. An int32 tensor.
* <b>`precision`</b>: An `int` that is `>= 1`.
  The number of bits for probability quantization. Must be <= 16.
* <b>`overflow_width`</b>: An `int` that is `>= 1`.
  The bit width of the variable-length overflow code. Must be <=
  precision.
* <b>`debug_level`</b>: An optional `int`. Defaults to `1`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` of type `string`.
A range-coded scalar string and a prefix varint string.
