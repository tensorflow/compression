<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfc.PackedTensors" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="model"/>
<meta itemprop="property" content="string"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="pack"/>
<meta itemprop="property" content="unpack"/>
</div>

# tfc.PackedTensors


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/util/packed_tensors.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `PackedTensors`

Packed representation of compressed tensors.



### Aliases:

* Class `tfc.python.util.packed_tensors.PackedTensors`


<!-- Placeholder for "Used in" -->

This class can pack and unpack several tensor values into a single string. It
can also optionally store a model identifier.

The tensors currently must be rank 1 (vectors) and either have integer or
string type.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/util/packed_tensors.py">View source</a>

``` python
__init__(string=None)
```

Initialize self.  See help(type(self)) for accurate signature.




## Properties

<h3 id="model"><code>model</code></h3>

A model identifier.


<h3 id="string"><code>string</code></h3>

The string representation of this object.




## Methods

<h3 id="pack"><code>pack</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/util/packed_tensors.py">View source</a>

``` python
pack(
    tensors,
    arrays
)
```

Packs `Tensor` values into this object.


<h3 id="unpack"><code>unpack</code></h3>

<a target="_blank" href="https://github.com/tensorflow/compression/tree/master/tensorflow_compression/python/util/packed_tensors.py">View source</a>

``` python
unpack(tensors)
```

Unpacks `Tensor` values from this object.




