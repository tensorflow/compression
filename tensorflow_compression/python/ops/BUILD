package(
    default_visibility = [
        "//research/vision/piedpiper:__subpackages__",
        "//third_party/tensorflow_compression:__pkg__",
        "//third_party/tensorflow_compression/python/layers:__pkg__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "ops",
    srcs = ["__init__.py"],
    deps = [
        ":math_ops",
        ":padding_ops",
        ":range_coding_ops",
        ":spectral_ops",
    ],
)

py_library(
    name = "math_ops",
    srcs = ["math_ops.py"],
    deps = ["//third_party/py/tensorflow"],
)

py_library(
    name = "padding_ops",
    srcs = ["padding_ops.py"],
)

py_library(
    name = "range_coding_ops",
    srcs = ["range_coding_ops.py"],
    deps = [
        ":gen_range_coding_ops",
        ":namespace_helper",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow_compression/cc:all_kernels",
    ],
)

py_library(
    name = "spectral_ops",
    srcs = ["spectral_ops.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "namespace_helper",
    srcs = ["namespace_helper.py"],
    visibility = ["//visibility:private"],
)

load("//devtools/python/blaze:python3.bzl", "py2and3_test")

py2and3_test(
    name = "math_ops_test",
    srcs = ["math_ops_test.py"],
    deps = [
        ":math_ops",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "padding_ops_test",
    srcs = ["padding_ops_test.py"],
    deps = [
        ":padding_ops",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "range_coding_ops_test",
    srcs = ["range_coding_ops_test.py"],
    deps = [
        ":range_coding_ops",
        "//third_party/py/tensorflow",
    ],
)

py2and3_test(
    name = "spectral_ops_test",
    srcs = ["spectral_ops_test.py"],
    deps = [
        ":spectral_ops",
        "//third_party/py/tensorflow",
    ],
)

load("//third_party/tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

tf_gen_op_wrapper_py(
    name = "gen_range_coding_ops",
    out = "gen_range_coding_ops.py",
    visibility = ["//visibility:private"],
    deps = [
        "//third_party/tensorflow_compression/cc:range_coding_ops_op_lib",
    ],
)
