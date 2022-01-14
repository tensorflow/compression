package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_compression",
    srcs = ["tensorflow_compression/__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_compression/python/datasets:y4m_dataset",
        "//tensorflow_compression/python/distributions:deep_factorized",
        "//tensorflow_compression/python/distributions:helpers",
        "//tensorflow_compression/python/distributions:round_adapters",
        "//tensorflow_compression/python/distributions:uniform_noise",
        "//tensorflow_compression/python/entropy_models:continuous_batched",
        "//tensorflow_compression/python/entropy_models:continuous_indexed",
        "//tensorflow_compression/python/entropy_models:universal",
        "//tensorflow_compression/python/layers:gdn",
        "//tensorflow_compression/python/layers:initializers",
        "//tensorflow_compression/python/layers:parameters",
        "//tensorflow_compression/python/layers:signal_conv",
        "//tensorflow_compression/python/layers:soft_round",
        "//tensorflow_compression/python/ops:gen_ops",
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:padding_ops",
        "//tensorflow_compression/python/ops:round_ops",
        "//tensorflow_compression/python/util:packed_tensors",
    ],
)

py_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.py"],
    data = [
        "LICENSE",
        "README.md",
        "MANIFEST.in",
        "requirements.txt",
        "tensorflow_compression/all_tests.py",
        ":tensorflow_compression",
        # The following targets are for Python unit tests.
        "//tensorflow_compression/python/datasets:py_src",
        "//tensorflow_compression/python/distributions:py_src",
        "//tensorflow_compression/python/entropy_models:py_src",
        "//tensorflow_compression/python/layers:py_src",
        "//tensorflow_compression/python/ops:py_src",
        "//tensorflow_compression/python/util:py_src",
    ],
    python_version = "PY3",
)
