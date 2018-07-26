.. py:module:: tensorflow_compression

.. toctree::
   :caption: Table of Contents
   :collapse: False
   :maxdepth: 4


Keras layers
============

Entropy models
--------------

EntropyBottleneck
^^^^^^^^^^^^^^^^^
.. autoclass:: EntropyBottleneck

Convolution
-----------

SignalConv1D
^^^^^^^^^^^^
.. autoclass:: SignalConv1D

SignalConv2D
^^^^^^^^^^^^
.. autoclass:: SignalConv2D

SignalConv3D
^^^^^^^^^^^^
.. autoclass:: SignalConv3D

Activation functions
--------------------

GDN
^^^
.. autoclass:: GDN

Parameterizers
--------------

These classes specify reparameterizations of variables created in Keras layers.

StaticParameterizer
^^^^^^^^^^^^^^^^^^^
.. autoclass:: StaticParameterizer

RDFTParameterizer
^^^^^^^^^^^^^^^^^
.. autoclass:: RDFTParameterizer

NonnegativeParameterizer
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: NonnegativeParameterizer


TensorFlow operations
=====================

Range coding
------------

pmf_to_quantized_cdf
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: pmf_to_quantized_cdf

range_encode
^^^^^^^^^^^^
.. autofunction:: range_encode

range_decode
^^^^^^^^^^^^
.. autofunction:: range_decode


Index & search
==============

* :ref:`genindex`
* :ref:`search`
