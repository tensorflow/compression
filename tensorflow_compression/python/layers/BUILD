package(
    default_visibility = ["//:__subpackages__"],
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
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:range_coding_ops",
    ],
)

py_library(
    name = "gdn",
    srcs = ["gdn.py"],
    deps = [":parameterizers"],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    visibility = ["//visibility:private"],
)

py_library(
    name = "parameterizers",
    srcs = ["parameterizers.py"],
    deps = [
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:spectral_ops",
    ],
)

py_library(
    name = "signal_conv",
    srcs = ["signal_conv.py"],
    deps = [
        ":parameterizers",
        "//tensorflow_compression/python/ops:padding_ops",
    ],
)

py_test(
    name = "entropy_models_test",
    srcs = ["entropy_models_test.py"],
    deps = [":entropy_models"],
)

py_test(
    name = "gdn_test",
    srcs = ["gdn_test.py"],
    deps = [":gdn"],
)

py_test(
    name = "parameterizers_test",
    srcs = ["parameterizers_test.py"],
    deps = [":parameterizers"],
)

py_test(
    name = "signal_conv_test",
    timeout = "long",
    srcs = ["signal_conv_test.py"],
    shard_count = 3,
    deps = [
        ":initializers",
        ":parameterizers",
        ":signal_conv",
    ],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
