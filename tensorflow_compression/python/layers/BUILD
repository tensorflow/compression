package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "entropy_models",
    srcs = ["entropy_models.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:range_coding_ops",
    ],
)

py_library(
    name = "gdn",
    srcs = ["gdn.py"],
    srcs_version = "PY3",
    deps = [":parameterizers"],
)

py_test(
    name = "gdn_test",
    srcs = ["gdn_test.py"],
    python_version = "PY3",
    deps = [":gdn"],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    srcs_version = "PY3",
)

py_library(
    name = "parameterizers",
    srcs = ["parameterizers.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:spectral_ops",
    ],
)

py_library(
    name = "signal_conv",
    srcs = ["signal_conv.py"],
    srcs_version = "PY3",
    deps = [
        ":parameterizers",
        "//tensorflow_compression/python/ops:padding_ops",
    ],
)

py_library(
    name = "soft_round",
    srcs = ["soft_round.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_compression/python/ops:soft_round_ops"],
)

py_test(
    name = "entropy_models_test",
    srcs = ["entropy_models_test.py"],
    python_version = "PY3",
    deps = [":entropy_models"],
)

py_test(
    name = "parameterizers_test",
    srcs = ["parameterizers_test.py"],
    python_version = "PY3",
    deps = [":parameterizers"],
)

py_test(
    name = "signal_conv_test",
    timeout = "long",
    srcs = ["signal_conv_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        ":initializers",
        ":parameterizers",
        ":signal_conv",
    ],
)

py_test(
    name = "soft_round_test",
    srcs = ["soft_round_test.py"],
    python_version = "PY3",
    deps = [":soft_round"],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
