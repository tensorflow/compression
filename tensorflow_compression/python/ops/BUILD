package(
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "ops",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
)

py_library(
    name = "padding_ops",
    srcs = ["padding_ops.py"],
    srcs_version = "PY3",
)

py_library(
    name = "range_coding_ops",
    srcs = ["range_coding_ops.py"],
    data = ["//tensorflow_compression/cc:libtensorflow_compression.so"],
    srcs_version = "PY3",
    deps = [":namespace_helper"],
)

py_library(
    name = "spectral_ops",
    srcs = ["spectral_ops.py"],
    srcs_version = "PY3",
)

py_library(
    name = "namespace_helper",
    srcs = ["namespace_helper.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:private"],
)

py_test(
    name = "math_ops_test",
    srcs = ["math_ops_test.py"],
    python_version = "PY3",
    deps = [":math_ops"],
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
    deps = [":range_coding_ops"],
)

py_test(
    name = "spectral_ops_test",
    srcs = ["spectral_ops_test.py"],
    python_version = "PY3",
    deps = [":spectral_ops"],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
