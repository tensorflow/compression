licenses(["notice"])  # Apache 2.0

py_library(
    name = "tensorflow_compression",
    srcs = [
        "tensorflow_compression/__init__.py",
        "tensorflow_compression/python/__init__.py",
        "tensorflow_compression/python/layers/__init__.py",
        "tensorflow_compression/python/layers/entropy_models.py",
        "tensorflow_compression/python/layers/gdn.py",
        "tensorflow_compression/python/layers/initializers.py",
        "tensorflow_compression/python/layers/parameterizers.py",
        "tensorflow_compression/python/layers/signal_conv.py",
        "tensorflow_compression/python/ops/__init__.py",
        "tensorflow_compression/python/ops/math_ops.py",
        "tensorflow_compression/python/ops/namespace_helper.py",
        "tensorflow_compression/python/ops/padding_ops.py",
        "tensorflow_compression/python/ops/range_coding_ops.py",
        "tensorflow_compression/python/ops/spectral_ops.py",
    ],
    data = [
        "//tensorflow_compression/cc:libtensorflow_compression.so",
    ],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.sh"],
    data = [
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "setup.py",
        ":tensorflow_compression",
    ],
)

py_test(
    name = "entropy_models_test",
    timeout = "long",
    srcs = ["tensorflow_compression/python/layers/entropy_models_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "gdn_test",
    srcs = ["tensorflow_compression/python/layers/gdn_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "parameterizers_test",
    srcs = ["tensorflow_compression/python/layers/parameterizers_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "signal_conv_test",
    timeout = "long",
    srcs = ["tensorflow_compression/python/layers/signal_conv_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "math_ops_test",
    srcs = ["tensorflow_compression/python/ops/math_ops_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "padding_ops_test",
    srcs = ["tensorflow_compression/python/ops/padding_ops_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "spectral_ops_test",
    srcs = ["tensorflow_compression/python/ops/spectral_ops_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "range_coding_ops_test",
    srcs = ["tensorflow_compression/python/ops/range_coding_ops_test.py"],
    deps = [":tensorflow_compression"],
)
