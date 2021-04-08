package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "gen_ops",
    srcs = ["gen_ops.py"],
    data = ["//tensorflow_compression/cc:libtensorflow_compression.so"],
    srcs_version = "PY3",
)

py_library(
    name = "math_ops",
    srcs = ["math_ops.py"],
    srcs_version = "PY3",
)

py_test(
    name = "math_ops_test",
    srcs = ["math_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":math_ops",
        ":soft_round_ops",
    ],
)

py_library(
    name = "padding_ops",
    srcs = ["padding_ops.py"],
    srcs_version = "PY3",
)

py_test(
    name = "padding_ops_test",
    srcs = ["padding_ops_test.py"],
    python_version = "PY3",
    deps = [":padding_ops"],
)

py_test(
    name = "range_coding_ops_test",
    srcs = ["range_coding_ops_test.py"],
    python_version = "PY3",
    deps = [":gen_ops"],
)

py_library(
    name = "soft_round_ops",
    srcs = ["soft_round_ops.py"],
    srcs_version = "PY3",
)

py_test(
    name = "soft_round_ops_test",
    srcs = ["soft_round_ops_test.py"],
    python_version = "PY3",
    deps = [":soft_round_ops"],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
