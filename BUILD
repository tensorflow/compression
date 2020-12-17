licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_compression",
    srcs = ["tensorflow_compression/__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_compression/python/distributions:deep_factorized",
        "//tensorflow_compression/python/distributions:helpers",
        "//tensorflow_compression/python/distributions:round_adapters",
        "//tensorflow_compression/python/distributions:uniform_noise",
        "//tensorflow_compression/python/entropy_models:continuous_batched",
        "//tensorflow_compression/python/entropy_models:continuous_indexed",
        "//tensorflow_compression/python/entropy_models:universal",
        "//tensorflow_compression/python/layers:entropy_models",
        "//tensorflow_compression/python/layers:gdn",
        "//tensorflow_compression/python/layers:initializers",
        "//tensorflow_compression/python/layers:parameterizers",
        "//tensorflow_compression/python/layers:signal_conv",
        "//tensorflow_compression/python/layers:soft_round",
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:padding_ops",
        "//tensorflow_compression/python/ops:range_coding_ops",
        "//tensorflow_compression/python/ops:soft_round_ops",
        "//tensorflow_compression/python/ops:spectral_ops",
        "//tensorflow_compression/python/util:packed_tensors",
    ],
)

filegroup(
    name = "pip_src",
    srcs = [
        "MANIFEST.in",
        "tensorflow_compression/all_tests.py",
    ],
)

py_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.py"],
    data = [
        "LICENSE",
        "README.md",
        ":pip_src",
        ":tensorflow_compression",
        # The following targets are for Python test files.
        "//tensorflow_compression/python/distributions:py_src",
        "//tensorflow_compression/python/entropy_models:py_src",
        "//tensorflow_compression/python/layers:py_src",
        "//tensorflow_compression/python/ops:py_src",
        "//tensorflow_compression/python/util:py_src",
    ],
    python_version = "PY3",
)
