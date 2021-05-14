description: Nonnegative parameterization as needed for GDN parameters.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.GDNParameter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfc.GDNParameter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameters.py#L169-L248">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Nonnegative parameterization as needed for GDN parameters.

Inherits From: [`Parameter`](../tfc/Parameter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.GDNParameter(
    initial_value, name=None, minimum=0.0, offset=(2 ** -18), shape=None, dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The variable is subjected to an invertible transformation that slows down the
learning rate for small values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`initial_value`
</td>
<td>
`tf.Tensor` or `None`. The initial value of the kernel. If
not provided, its `shape` must be given, and the initial value of the
parameter will be undefined.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String. The name of the parameter.
</td>
</tr><tr>
<td>
`minimum`
</td>
<td>
Float. Lower bound for the parameter (defaults to zero).
</td>
</tr><tr>
<td>
`offset`
</td>
<td>
Float. Offset added to the reparameterization. The
parameterization of beta/gamma as their square roots lets the training
slow down when values are close to zero, which is desirable as small
values in the denominator can lead to a situation where gradient noise
on beta/gamma leads to extreme amounts of noise in the GDN activations.
However, without the offset, we would get zero gradients if any elements
of beta or gamma were exactly zero, and thus the training could get
stuck. To prevent this, we add this small constant. The default value
was empirically determined as a good starting point. Making it bigger
potentially leads to more gradient noise on the activations, making it
too small may lead to numerical precision issues.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
`tf.TensorShape` or compatible. Ignored unless `initial_value is
None`.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
`tf.dtypes.DType` or compatible. DType of this parameter. If not
given, inferred from `initial_value`.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`minimum`
</td>
<td>
Float. The `minimum` parameter provided on initialization.
</td>
</tr><tr>
<td>
`offset`
</td>
<td>
Float. The `offset` parameter provided on initialization.
</td>
</tr><tr>
<td>
`variable`
</td>
<td>
`tf.Variable`. The reparameterized variable.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.
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

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameters.py#L239-L248">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>

Returns the configuration of the `Parameter`.


<h3 id="get_weights"><code>get_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameters.py#L47-L48">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_weights()
</code></pre>




<h3 id="set_weights"><code>set_weights</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameters.py#L50-L55">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_weights(
    weights
)
</code></pre>




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



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/layers/parameters.py#L225-L229">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__() -> tf.Tensor
</code></pre>

Computes and returns the non-negative value as a `tf.Tensor`.




