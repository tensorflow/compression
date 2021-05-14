description: Adapt a continuous distribution via an ascending monotonic function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.MonotonicAdapter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="cdf"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="covariance"/>
<meta itemprop="property" content="cross_entropy"/>
<meta itemprop="property" content="entropy"/>
<meta itemprop="property" content="event_shape_tensor"/>
<meta itemprop="property" content="experimental_default_event_space_bijector"/>
<meta itemprop="property" content="inverse_transform"/>
<meta itemprop="property" content="is_scalar_batch"/>
<meta itemprop="property" content="is_scalar_event"/>
<meta itemprop="property" content="kl_divergence"/>
<meta itemprop="property" content="log_cdf"/>
<meta itemprop="property" content="log_prob"/>
<meta itemprop="property" content="log_survival_function"/>
<meta itemprop="property" content="mean"/>
<meta itemprop="property" content="mode"/>
<meta itemprop="property" content="param_shapes"/>
<meta itemprop="property" content="param_static_shapes"/>
<meta itemprop="property" content="parameter_properties"/>
<meta itemprop="property" content="prob"/>
<meta itemprop="property" content="quantile"/>
<meta itemprop="property" content="sample"/>
<meta itemprop="property" content="stddev"/>
<meta itemprop="property" content="survival_function"/>
<meta itemprop="property" content="transform"/>
<meta itemprop="property" content="unnormalized_log_prob"/>
<meta itemprop="property" content="variance"/>
<meta itemprop="property" content="with_name_scope"/>
<meta itemprop="property" content="invertible"/>
</div>

# tfc.MonotonicAdapter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/distributions/round_adapters.py#L36-L157">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Adapt a continuous distribution via an ascending monotonic function.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.MonotonicAdapter(
    base, name=&#x27;MonotonicAdapter&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is described in Appendix E. in the paper
> "Universally Quantized Neural Compression"<br />
> Eirikur Agustsson & Lucas Theis<br />
> https://arxiv.org/abs/2006.09952

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`base`
</td>
<td>
A `tfp.distributions.Distribution` object representing a
continuous-valued random variable.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String. A name for this distribution.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`allow_nan_stats`
</td>
<td>
Python `bool` describing behavior when a stat is undefined.

Stats return +/- infinity when it makes sense. E.g., the variance of a
Cauchy distribution is infinity. However, sometimes the statistic is
undefined, e.g., if a distribution's pdf does not achieve a maximum within
the support of the distribution, the mode is undefined. If the mean is
undefined, then by definition the variance is undefined. E.g. the mean for
Student's T for df = 1 is undefined (no clear way to say it is either + or -
infinity), so the variance = E[(X - mean)**2] is also undefined.
</td>
</tr><tr>
<td>
`base`
</td>
<td>
The base distribution.
</td>
</tr><tr>
<td>
`batch_shape`
</td>
<td>
Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The `DType` of `Tensor`s handled by this `Distribution`.
</td>
</tr><tr>
<td>
`event_shape`
</td>
<td>
Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.
</td>
</tr><tr>
<td>
`experimental_shard_axis_names`
</td>
<td>
The list or structure of lists of active shard axis names.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Name prepended to all ops created by this `Distribution`.
</td>
</tr><tr>
<td>
`name_scope`
</td>
<td>
Returns a `tf.name_scope` instance for this class.
</td>
</tr><tr>
<td>
`non_trainable_variables`
</td>
<td>
Sequence of non-trainable variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr><tr>
<td>
`parameters`
</td>
<td>
Dictionary of parameters used to instantiate this `Distribution`.
</td>
</tr><tr>
<td>
`reparameterization_type`
</td>
<td>
Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`tfd.FULLY_REPARAMETERIZED` or `tfd.NOT_REPARAMETERIZED`.
</td>
</tr><tr>
<td>
`submodules`
</td>
<td>
Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> list(a.submodules) == [b, c]
True
>>> list(b.submodules) == [c]
True
>>> list(c.submodules) == []
True
```
</td>
</tr><tr>
<td>
`trainable_variables`
</td>
<td>
Sequence of trainable variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr><tr>
<td>
`validate_args`
</td>
<td>
Python `bool` indicating possibly expensive checks are enabled.
</td>
</tr><tr>
<td>
`variables`
</td>
<td>
Sequence of variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr>
</table>



## Methods

<h3 id="batch_shape_tensor"><code>batch_shape_tensor</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_shape_tensor(
    name=&#x27;batch_shape_tensor&#x27;
)
</code></pre>

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
name to give to the op
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`batch_shape`
</td>
<td>
`Tensor`.
</td>
</tr>
</table>



<h3 id="cdf"><code>cdf</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cdf(
    value, name=&#x27;cdf&#x27;, **kwargs
)
</code></pre>

Cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```none
cdf(x) := P[X <= x]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
`float` or `double` `Tensor`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`cdf`
</td>
<td>
a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
values of type `self.dtype`.
</td>
</tr>
</table>



<h3 id="copy"><code>copy</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy(
    **override_parameters_kwargs
)
</code></pre>

Creates a deep copy of the distribution.

Note: the copy distribution may continue to depend on the original
initialization arguments.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`**override_parameters_kwargs`
</td>
<td>
String/value dictionary of initialization
arguments to override with new values.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`distribution`
</td>
<td>
A new instance of `type(self)` initialized from the union
of self.parameters and override_parameters_kwargs, i.e.,
`dict(self.parameters, **override_parameters_kwargs)`.
</td>
</tr>
</table>



<h3 id="covariance"><code>covariance</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>covariance(
    name=&#x27;covariance&#x27;, **kwargs
)
</code></pre>

Covariance.

Covariance is (possibly) defined only for non-scalar-event distributions.

For example, for a length-`k`, vector-valued distribution, it is calculated
as,

```none
Cov[i, j] = Covariance(X_i, X_j) = E[(X_i - E[X_i]) (X_j - E[X_j])]
```

where `Cov` is a (batch of) `k x k` matrix, `0 <= (i, j) < k`, and `E`
denotes expectation.

Alternatively, for non-vector, multivariate distributions (e.g.,
matrix-valued, Wishart), `Covariance` shall return a (batch of) matrices
under some vectorization of the events, i.e.,

```none
Cov[i, j] = Covariance(Vec(X)_i, Vec(X)_j) = [as above]
```

where `Cov` is a (batch of) `k' x k'` matrices,
`0 <= (i, j) < k' = reduce_prod(event_shape)`, and `Vec` is some function
mapping indices of this distribution's event dimensions to indices of a
length-`k'` vector.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`covariance`
</td>
<td>
Floating-point `Tensor` with shape `[B1, ..., Bn, k', k']`
where the first `n` dimensions are batch coordinates and
`k' = reduce_prod(self.event_shape)`.
</td>
</tr>
</table>



<h3 id="cross_entropy"><code>cross_entropy</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cross_entropy(
    other, name=&#x27;cross_entropy&#x27;
)
</code></pre>

Computes the (Shannon) cross entropy.

Denote this distribution (`self`) by `P` and the `other` distribution by
`Q`. Assuming `P, Q` are absolutely continuous with respect to
one another and permit densities `p(x) dr(x)` and `q(x) dr(x)`, (Shannon)
cross entropy is defined as:

```none
H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)
```

where `F` denotes the support of the random variable `X ~ P`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
`tfp.distributions.Distribution` instance.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`cross_entropy`
</td>
<td>
`self.dtype` `Tensor` with shape `[B1, ..., Bn]`
representing `n` different calculations of (Shannon) cross entropy.
</td>
</tr>
</table>



<h3 id="entropy"><code>entropy</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>entropy(
    name=&#x27;entropy&#x27;, **kwargs
)
</code></pre>

Shannon entropy in nats.


<h3 id="event_shape_tensor"><code>event_shape_tensor</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>event_shape_tensor(
    name=&#x27;event_shape_tensor&#x27;
)
</code></pre>

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
name to give to the op
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`event_shape`
</td>
<td>
`Tensor`.
</td>
</tr>
</table>



<h3 id="experimental_default_event_space_bijector"><code>experimental_default_event_space_bijector</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_default_event_space_bijector(
    *args, **kwargs
)
</code></pre>

Bijector mapping the reals (R**n) to the event space of the distribution.

Distributions with continuous support may implement
`_default_event_space_bijector` which returns a subclass of
`tfp.bijectors.Bijector` that maps R**n to the distribution's event space.
For example, the default bijector for the `Beta` distribution
is `tfp.bijectors.Sigmoid()`, which maps the real line to `[0, 1]`, the
support of the `Beta` distribution. The default bijector for the
`CholeskyLKJ` distribution is `tfp.bijectors.CorrelationCholesky`, which
maps R^(k * (k-1) // 2) to the submanifold of k x k lower triangular
matrices with ones along the diagonal.

The purpose of `experimental_default_event_space_bijector` is
to enable gradient descent in an unconstrained space for Variational
Inference and Hamiltonian Monte Carlo methods. Some effort has been made to
choose bijectors such that the tails of the distribution in the
unconstrained space are between Gaussian and Exponential.

For distributions with discrete event space, or for which TFP currently
lacks a suitable bijector, this function returns `None`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
Passed to implementation `_default_event_space_bijector`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Passed to implementation `_default_event_space_bijector`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`event_space_bijector`
</td>
<td>
`Bijector` instance or `None`.
</td>
</tr>
</table>



<h3 id="inverse_transform"><code>inverse_transform</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/distributions/round_adapters.py#L76-L82">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>inverse_transform(
    y
)
</code></pre>

The backward transform.


<h3 id="is_scalar_batch"><code>is_scalar_batch</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_scalar_batch(
    name=&#x27;is_scalar_batch&#x27;
)
</code></pre>

Indicates that `batch_shape == []`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`is_scalar_batch`
</td>
<td>
`bool` scalar `Tensor`.
</td>
</tr>
</table>



<h3 id="is_scalar_event"><code>is_scalar_event</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_scalar_event(
    name=&#x27;is_scalar_event&#x27;
)
</code></pre>

Indicates that `event_shape == []`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`is_scalar_event`
</td>
<td>
`bool` scalar `Tensor`.
</td>
</tr>
</table>



<h3 id="kl_divergence"><code>kl_divergence</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>kl_divergence(
    other, name=&#x27;kl_divergence&#x27;
)
</code></pre>

Computes the Kullback--Leibler divergence.

Denote this distribution (`self`) by `p` and the `other` distribution by
`q`. Assuming `p, q` are absolutely continuous with respect to reference
measure `r`, the KL divergence is defined as:

```none
KL[p, q] = E_p[log(p(X)/q(X))]
         = -int_F p(x) log q(x) dr(x) + int_F p(x) log p(x) dr(x)
         = H[p, q] - H[p]
```

where `F` denotes the support of the random variable `X ~ p`, `H[., .]`
denotes (Shannon) cross entropy, and `H[.]` denotes (Shannon) entropy.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
`tfp.distributions.Distribution` instance.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`kl_divergence`
</td>
<td>
`self.dtype` `Tensor` with shape `[B1, ..., Bn]`
representing `n` different calculations of the Kullback-Leibler
divergence.
</td>
</tr>
</table>



<h3 id="log_cdf"><code>log_cdf</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>log_cdf(
    value, name=&#x27;log_cdf&#x27;, **kwargs
)
</code></pre>

Log cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```none
log_cdf(x) := Log[ P[X <= x] ]
```

Often, a numerical approximation can be used for `log_cdf(x)` that yields
a more accurate answer than simply taking the logarithm of the `cdf` when
`x << -1`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
`float` or `double` `Tensor`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`logcdf`
</td>
<td>
a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
values of type `self.dtype`.
</td>
</tr>
</table>



<h3 id="log_prob"><code>log_prob</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>log_prob(
    value, name=&#x27;log_prob&#x27;, **kwargs
)
</code></pre>

Log probability density/mass function.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
`float` or `double` `Tensor`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`log_prob`
</td>
<td>
a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
values of type `self.dtype`.
</td>
</tr>
</table>



<h3 id="log_survival_function"><code>log_survival_function</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>log_survival_function(
    value, name=&#x27;log_survival_function&#x27;, **kwargs
)
</code></pre>

Log survival function.

Given random variable `X`, the survival function is defined:

```none
log_survival_function(x) = Log[ P[X > x] ]
                         = Log[ 1 - P[X <= x] ]
                         = Log[ 1 - cdf(x) ]
```

Typically, different numerical approximations can be used for the log
survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
`float` or `double` `Tensor`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
`self.dtype`.
</td>
</tr>

</table>



<h3 id="mean"><code>mean</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mean(
    name=&#x27;mean&#x27;, **kwargs
)
</code></pre>

Mean.


<h3 id="mode"><code>mode</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mode(
    name=&#x27;mode&#x27;, **kwargs
)
</code></pre>

Mode.


<h3 id="param_shapes"><code>param_shapes</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>param_shapes(
    sample_shape, name=&#x27;DistributionParamShapes&#x27;
)
</code></pre>

Shapes of parameters given the desired shape of a call to `sample()`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2021-03-01.
Instructions for updating:
The `param_shapes` method of `tfd.Distribution` is deprecated; use `parameter_properties` instead.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`.

Subclasses should override class method `_param_shapes`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`sample_shape`
</td>
<td>
`Tensor` or python list/tuple. Desired shape of a call to
`sample()`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
name to prepend ops with.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`dict` of parameter name to `Tensor` shapes.
</td>
</tr>

</table>



<h3 id="param_static_shapes"><code>param_static_shapes</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>param_static_shapes(
    sample_shape
)
</code></pre>

param_shapes with static (i.e. `TensorShape`) shapes. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2021-03-01.
Instructions for updating:
The `param_static_shapes` method of `tfd.Distribution` is deprecated; use `parameter_properties` instead.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`. Assumes that the sample's
shape is known statically.

Subclasses should override class method `_param_shapes` to return
constant-valued tensors when constant values are fed.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`sample_shape`
</td>
<td>
`TensorShape` or python list/tuple. Desired shape of a call
to `sample()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`dict` of parameter name to `TensorShape`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if `sample_shape` is a `TensorShape` and is not fully defined.
</td>
</tr>
</table>



<h3 id="parameter_properties"><code>parameter_properties</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>parameter_properties(
    dtype=tf.float32, num_classes=None
)
</code></pre>

Returns a dict mapping constructor arg names to property annotations.

This dict should include an entry for each of the distribution's
`Tensor`-valued constructor arguments.

Distribution subclasses are not required to implement
`_parameter_properties`, so this method may raise `NotImplementedError`.
Providing a `_parameter_properties` implementation enables several advanced
features, including:
  - Distribution batch slicing (`sliced_distribution = distribution[i:j]`).
  - Automatic inference of `_batch_shape` and
    `_batch_shape_tensor`, which must otherwise be computed explicitly.
  - Automatic instantiation of the distribution within TFP's internal
    property tests.
  - Automatic construction of 'trainable' instances of the distribution
    using appropriate bijectors to avoid violating parameter constraints.
    This enables the distribution family to be used easily as a
    surrogate posterior in variational inference.

In the future, parameter property annotations may enable additional
functionality; for example, returning Distribution instances from
`tf.vectorized_map`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`dtype`
</td>
<td>
Optional float `dtype` to assume for continuous-valued parameters.
Some constraining bijectors require advance knowledge of the dtype
because certain constants (e.g., `tfb.Softplus.low`) must be
instantiated with the same dtype as the values to be transformed.
</td>
</tr><tr>
<td>
`num_classes`
</td>
<td>
Optional `int` `Tensor` number of classes to assume when
inferring the shape of parameters for categorical-like distributions.
Otherwise ignored.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`parameter_properties`
</td>
<td>
A
`str -> `tfp.python.internal.parameter_properties.ParameterProperties`
dict mapping constructor argument names to `ParameterProperties`
instances.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`NotImplementedError`
</td>
<td>
if the distribution class does not implement
`_parameter_properties`.
</td>
</tr>
</table>



<h3 id="prob"><code>prob</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>prob(
    value, name=&#x27;prob&#x27;, **kwargs
)
</code></pre>

Probability density/mass function.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
`float` or `double` `Tensor`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`prob`
</td>
<td>
a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
values of type `self.dtype`.
</td>
</tr>
</table>



<h3 id="quantile"><code>quantile</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>quantile(
    value, name=&#x27;quantile&#x27;, **kwargs
)
</code></pre>

Quantile function. Aka 'inverse cdf' or 'percent point function'.

Given random variable `X` and `p in [0, 1]`, the `quantile` is:

```none
quantile(p) := x such that P[X <= x] == p
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
`float` or `double` `Tensor`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`quantile`
</td>
<td>
a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
values of type `self.dtype`.
</td>
</tr>
</table>



<h3 id="sample"><code>sample</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sample(
    sample_shape=(), seed=None, name=&#x27;sample&#x27;, **kwargs
)
</code></pre>

Generate samples of the specified shape.

Note that a call to `sample()` without arguments will generate a single
sample.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`sample_shape`
</td>
<td>
0D or 1D `int32` `Tensor`. Shape of the generated samples.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Python integer or `tfp.util.SeedStream` instance, for seeding PRNG.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
name to give to the op.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`samples`
</td>
<td>
a `Tensor` with prepended dimensions `sample_shape`.
</td>
</tr>
</table>



<h3 id="stddev"><code>stddev</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>stddev(
    name=&#x27;stddev&#x27;, **kwargs
)
</code></pre>

Standard deviation.

Standard deviation is defined as,

```none
stddev = E[(X - E[X])**2]**0.5
```

where `X` is the random variable associated with this distribution, `E`
denotes expectation, and `stddev.shape = batch_shape + event_shape`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`stddev`
</td>
<td>
Floating-point `Tensor` with shape identical to
`batch_shape + event_shape`, i.e., the same shape as `self.mean()`.
</td>
</tr>
</table>



<h3 id="survival_function"><code>survival_function</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>survival_function(
    value, name=&#x27;survival_function&#x27;, **kwargs
)
</code></pre>

Survival function.

Given random variable `X`, the survival function is defined:

```none
survival_function(x) = P[X > x]
                     = 1 - P[X <= x]
                     = 1 - cdf(x).
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
`float` or `double` `Tensor`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
`Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
`self.dtype`.
</td>
</tr>

</table>



<h3 id="transform"><code>transform</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/distributions/round_adapters.py#L72-L74">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>transform(
    x
)
</code></pre>

The forward transform.


<h3 id="unnormalized_log_prob"><code>unnormalized_log_prob</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unnormalized_log_prob(
    value, name=&#x27;unnormalized_log_prob&#x27;, **kwargs
)
</code></pre>

Potentially unnormalized log probability density/mass function.

This function is similar to `log_prob`, but does not require that the
return value be normalized.  (Normalization here refers to the total
integral of probability being one, as it should be by definition for any
probability distribution.)  This is useful, for example, for distributions
where the normalization constant is difficult or expensive to compute.  By
default, this simply calls `log_prob`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
`float` or `double` `Tensor`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`unnormalized_log_prob`
</td>
<td>
a `Tensor` of shape
`sample_shape(x) + self.batch_shape` with values of type `self.dtype`.
</td>
</tr>
</table>



<h3 id="variance"><code>variance</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>variance(
    name=&#x27;variance&#x27;, **kwargs
)
</code></pre>

Variance.

Variance is defined as,

```none
Var = E[(X - E[X])**2]
```

where `X` is the random variable associated with this distribution, `E`
denotes expectation, and `Var.shape = batch_shape + event_shape`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
Python `str` prepended to names of ops created by this function.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Named arguments forwarded to subclass implementation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`variance`
</td>
<td>
Floating-point `Tensor` with shape identical to
`batch_shape + event_shape`, i.e., the same shape as `self.mean()`.
</td>
</tr>
</table>



<h3 id="with_name_scope"><code>with_name_scope</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>with_name_scope(
    method
)
</code></pre>

Decorator to automatically enter the module name scope.

```
>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
...     return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
>>> mod = MyModule()
>>> mod(tf.ones([1, 2]))
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
>>> mod.w
<tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
numpy=..., dtype=float32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`method`
</td>
<td>
The method to wrap.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The original method wrapped such that it enters the module's name scope.
</td>
</tr>

</table>



<h3 id="__getitem__"><code>__getitem__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    slices
)
</code></pre>

Slices the batch axes of this distribution, returning a new instance.

```python
b = tfd.Bernoulli(logits=tf.zeros([3, 5, 7, 9]))
b.batch_shape  # => [3, 5, 7, 9]
b2 = b[:, tf.newaxis, ..., -2:, 1::2]
b2.batch_shape  # => [3, 1, 5, 2, 4]

x = tf.random.normal([5, 3, 2, 2])
cov = tf.matmul(x, x, transpose_b=True)
chol = tf.linalg.cholesky(cov)
loc = tf.random.normal([4, 1, 3, 1])
mvn = tfd.MultivariateNormalTriL(loc, chol)
mvn.batch_shape  # => [4, 5, 3]
mvn.event_shape  # => [2]
mvn2 = mvn[:, 3:, ..., ::-1, tf.newaxis]
mvn2.batch_shape  # => [4, 2, 3, 1]
mvn2.event_shape  # => [2]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`slices`
</td>
<td>
slices from the [] operator
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
`dist`
</td>
<td>
A new `tfd.Distribution` instance with sliced parameters.
</td>
</tr>
</table>



<h3 id="__iter__"><code>__iter__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>








<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
invertible<a id="invertible"></a>
</td>
<td>
`True`
</td>
</tr>
</table>

