<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc" />
</div>

# Module: tfc

Data compression tools.

## Classes

[`class EntropyBottleneck`](./tfc/EntropyBottleneck.md): Entropy bottleneck layer.

[`class GDN`](./tfc/GDN.md): Generalized divisive normalization layer.

[`class IdentityInitializer`](./tfc/IdentityInitializer.md): Initialize to the identity kernel with the given shape.

[`class NonnegativeParameterizer`](./tfc/NonnegativeParameterizer.md): Object encapsulating nonnegative parameterization as needed for GDN.

[`class Parameterizer`](./tfc/Parameterizer.md): Parameterizer object (abstract base class).

[`class RDFTParameterizer`](./tfc/RDFTParameterizer.md): Object encapsulating RDFT reparameterization.

[`class SignalConv1D`](./tfc/SignalConv1D.md): 1D convolution layer.

[`class SignalConv2D`](./tfc/SignalConv2D.md): 2D convolution layer.

[`class SignalConv3D`](./tfc/SignalConv3D.md): 3D convolution layer.

[`class StaticParameterizer`](./tfc/StaticParameterizer.md): A parameterization object that always returns a constant tensor.

## Functions

[`irdft_matrix(...)`](./tfc/irdft_matrix.md): Matrix for implementing kernel reparameterization with `tf.matmul`.

[`lower_bound(...)`](./tfc/lower_bound.md): Same as `tf.maximum`, but with helpful gradient for `inputs < bound`.

[`pmf_to_quantized_cdf(...)`](./tfc/pmf_to_quantized_cdf.md): Converts PMF to quantized CDF. This op uses floating-point operations

[`range_decode(...)`](./tfc/range_decode.md): Decodes a range-coded `code` into an int32 tensor of shape `shape`.

[`range_encode(...)`](./tfc/range_encode.md): Using the provided cumulative distribution functions (CDF) inside `cdf`, returns

[`same_padding_for_kernel(...)`](./tfc/same_padding_for_kernel.md): Determine correct amount of padding for `same` convolution.

[`upper_bound(...)`](./tfc/upper_bound.md): Same as `tf.minimum`, but with helpful gradient for `inputs > bound`.

