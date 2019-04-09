
# tfc.deprecated_endpoints

### Aliases:

* `tfc.deprecated_endpoints`
* `tfc.python.ops.range_coding_ops.deprecated_endpoints`

``` python
tfc.deprecated_endpoints(*args)
```

<!-- Placeholder for "Used in" -->

Decorator for marking endpoints deprecated.

This decorator does not print deprecation messages.
TODO(annarev): eventually start printing deprecation warnings when
@deprecation_endpoints decorator is added.

#### Args:

* <b>`*args`</b>: Deprecated endpoint names.


#### Returns:

A function that takes symbol as an argument and adds
_tf_deprecated_api_names to that symbol.
_tf_deprecated_api_names would be set to a list of deprecated
endpoint names for the symbol.