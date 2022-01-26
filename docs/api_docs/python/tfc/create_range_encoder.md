description: Creates range encoder objects to be used by EntropyEncode* ops.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.create_range_encoder" />
<meta itemprop="path" content="Stable" />
</div>

# tfc.create_range_encoder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates range encoder objects to be used by `EntropyEncode*` ops.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.create_range_encoder(
    shape, lookup, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The output `handle` has the shape specified by the input `shape`. Each element
in `handle` is an independent range encoder object, and `EntropyEncode*`
processes as many concurrent code streams as contained in `handle`.

This op expects `lookup` to be either a concatenation (1-D) or stack (2-D) of
CDFs, where each CDF is preceded by a corresponding precision value. In case of
a stack:

```
   lookup[..., 0] = precision in [1, 16],
   lookup[..., 1] / 2^precision = Pr(X < 0) = 0,
   lookup[..., 2] / 2^precision = Pr(X < 1),
   lookup[..., 3] / 2^precision = Pr(X < 2),
   ...
   lookup[..., -1] / 2^precision = 1,
```

Subsequent values in each CDF may be equal, indicating a symbol with zero
probability. Attempting to encode such a symbol will result in undefined
behavior. However, any number of trailing zero-probability symbols will be
interpreted as padding, and attempting to use those will result in an encoding
error (unless overflow functionality is used).

Overflow functionality can be enabled by negating the precision value in
`lookup`. In that case, the last non-zero probability symbol in the CDF is used
as an escape code, allowing negative integers and integers greater or equal to
the last non-zero probability symbol to be encoded using an Elias gamma code,
which is interleaved into the code stream. Attempting to encode a
zero-probability symbol within the valid range still causes undefined behavior.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`lookup`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `variant`.
</td>
</tr>

</table>

