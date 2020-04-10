package(
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "entropy_models",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":continuous_base",
        ":continuous_batched",
        ":continuous_indexed",
    ],
)

py_library(
    name = "continuous_base",
    srcs = ["continuous_base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_compression/python/distributions:helpers",
        "//tensorflow_compression/python/ops:range_coding_ops",
    ],
)

py_library(
    name = "continuous_batched",
    srcs = ["continuous_batched.py"],
    srcs_version = "PY3",
    deps = [
        ":continuous_base",
        "//tensorflow_compression/python/distributions:helpers",
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:range_coding_ops",
    ],
)

py_test(
    name = "continuous_batched_test",
    srcs = ["continuous_batched_test.py"],
    python_version = "PY3",
    deps = [
        ":continuous_batched",
        "//tensorflow_compression/python/distributions:uniform_noise",
    ],
)

py_library(
    name = "continuous_indexed",
    srcs = ["continuous_indexed.py"],
    srcs_version = "PY3",
    deps = [
        ":continuous_base",
        "//tensorflow_compression/python/distributions:helpers",
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:range_coding_ops",
    ],
)

py_test(
    name = "continuous_indexed_test",
    srcs = ["continuous_indexed_test.py"],
    python_version = "PY3",
    deps = [
        ":continuous_indexed",
        "//tensorflow_compression/python/distributions:uniform_noise",
    ],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
