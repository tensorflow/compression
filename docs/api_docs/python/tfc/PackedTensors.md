description: Packed representation of compressed tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.PackedTensors" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="pack"/>
<meta itemprop="property" content="unpack"/>
</div>

# tfc.PackedTensors

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/util/packed_tensors.py#L25-L96">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Packed representation of compressed tensors.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tfc.PackedTensors(
    string=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class can pack and unpack several tensor values into a single string. It
can also optionally store a model identifier.

The tensors currently must be rank 1 (vectors) and either have integer or
string type.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`model`
</td>
<td>
A model identifier.
</td>
</tr><tr>
<td>
`string`
</td>
<td>
The string representation of this object.
</td>
</tr>
</table>



## Methods

<h3 id="pack"><code>pack</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/util/packed_tensors.py#L64-L82">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pack(
    tensors
)
</code></pre>

Packs `Tensor` values into this object.


<h3 id="unpack"><code>unpack</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/util/packed_tensors.py#L84-L96">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unpack(
    dtypes
)
</code></pre>

Unpacks values from this object based on dtypes.




