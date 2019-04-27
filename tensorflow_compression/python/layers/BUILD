package(
    default_visibility = [
        "//research/vision/piedpiper:__subpackages__",
        "//third_party/tensorflow_compression:__pkg__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "layers",
    srcs = ["__init__.py"],
    deps = [
        ":entropy_models",
        ":gdn",
        ":initializers",
        ":parameterizers",
        ":signal_conv",
    ],
)

py_library(
    name = "entropy_models",
    srcs = ["entropy_models.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_compression/python/ops:math_ops",
        "//third_party/tensorflow_compression/python/ops:range_coding_ops",
    ],
)

py_library(
    name = "gdn",
    srcs = ["gdn.py"],
    deps = [
        ":parameterizers",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    visibility = ["//visibility:private"],
    deps = ["//third_party/py/tensorflow"],
)

py_library(
    name = "parameterizers",
    srcs = ["parameterizers.py"],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_compression/python/ops:math_ops",
        "//third_party/tensorflow_compression/python/ops:spectral_ops",
    ],
)

py_library(
    name = "signal_conv",
    srcs = ["signal_conv.py"],
    deps = [
        ":parameterizers",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_compression/python/ops:padding_ops",
    ],
)

load("//devtools/python/blaze:python3.bzl", "py2and3_test")

py2and3_test(
    name = "entropy_models_test",
    srcs = ["entropy_models_test.py"],
    deps = [
        ":entropy_models",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "gdn_test",
    srcs = ["gdn_test.py"],
    deps = [
        ":gdn",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "parameterizers_test",
    srcs = ["parameterizers_test.py"],
    deps = [
        ":parameterizers",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "signal_conv_test",
    timeout = "long",
    srcs = ["signal_conv_test.py"],
    shard_count = 3,
    deps = [
        ":initializers",
        ":parameterizers",
        ":signal_conv",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
    ],
)
