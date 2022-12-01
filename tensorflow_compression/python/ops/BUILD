package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "ops",
    srcs = ["__init__.py"],
    deps = [
        ":gen_ops",
        ":math_ops",
        ":padding_ops",
        ":round_ops",
    ],
)

py_library(
    name = "gen_ops",
    srcs = ["gen_ops.py"],
    data = ["//tensorflow_compression/cc:libtensorflow_compression.so"],
)

py_library(
    name = "math_ops",
    srcs = ["math_ops.py"],
)

py_test(
    name = "math_ops_test",
    srcs = ["math_ops_test.py"],
    deps = [
        ":math_ops",
        ":round_ops",
    ],
)

py_library(
    name = "padding_ops",
    srcs = ["padding_ops.py"],
)

py_test(
    name = "padding_ops_test",
    srcs = ["padding_ops_test.py"],
    deps = [":padding_ops"],
)

py_test(
    name = "range_coding_ops_test",
    srcs = ["range_coding_ops_test.py"],
    deps = [":gen_ops"],
)

py_library(
    name = "round_ops",
    srcs = ["round_ops.py"],
)

py_test(
    name = "round_ops_test",
    srcs = ["round_ops_test.py"],
    deps = [":round_ops"],
)

py_test(
    name = "quantization_ops_test",
    srcs = ["quantization_ops_test.py"],
    deps = [":gen_ops"],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
