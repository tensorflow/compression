
# tfc.unbounded_index_range_decode_eager_fallback

### Aliases:

* `tfc.python.ops.range_coding_ops.unbounded_index_range_decode_eager_fallback`
* `tfc.unbounded_index_range_decode_eager_fallback`

``` python
tfc.unbounded_index_range_decode_eager_fallback(
    encoded,
    index,
    cdf,
    cdf_size,
    offset,
    precision,
    overflow_width,
    name=None,
    ctx=None
)
```



Defined in [`python/ops/range_coding_ops.py`](https://github.com/tensorflow/compression/tree/master/python/ops/range_coding_ops.py).

<!-- Placeholder for "Used in" -->

This is the slowpath function for Eager mode.
This is for function unbounded_index_range_decode