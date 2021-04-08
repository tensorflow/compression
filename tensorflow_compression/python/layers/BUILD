package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "gdn",
    srcs = ["gdn.py"],
    srcs_version = "PY3",
    deps = [":parameters"],
)

py_test(
    name = "gdn_test",
    srcs = ["gdn_test.py"],
    python_version = "PY3",
    deps = [
        ":gdn",
        ":parameters",
    ],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    srcs_version = "PY3",
)

py_test(
    name = "initializers_test",
    srcs = ["initializers_test.py"],
    python_version = "PY3",
    deps = [":initializers"],
)

py_library(
    name = "parameters",
    srcs = ["parameters.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_compression/python/ops:math_ops"],
)

py_test(
    name = "parameters_test",
    srcs = ["parameters_test.py"],
    python_version = "PY3",
    deps = [":parameters"],
)

py_library(
    name = "signal_conv",
    srcs = ["signal_conv.py"],
    srcs_version = "PY3",
    deps = [
        ":parameters",
        "//tensorflow_compression/python/ops:padding_ops",
    ],
)

py_test(
    name = "signal_conv_test",
    timeout = "long",
    srcs = ["signal_conv_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        ":initializers",
        ":parameters",
        ":signal_conv",
    ],
)

py_library(
    name = "soft_round",
    srcs = ["soft_round.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_compression/python/ops:soft_round_ops"],
)

py_test(
    name = "soft_round_test",
    srcs = ["soft_round_test.py"],
    python_version = "PY3",
    deps = [
        ":soft_round",
        "//tensorflow_compression/python/ops:soft_round_ops",
    ],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
