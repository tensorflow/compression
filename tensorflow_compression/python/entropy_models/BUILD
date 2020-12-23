package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

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

py_library(
    name = "universal",
    srcs = ["universal.py"],
    srcs_version = "PY3",
    deps = [
        ":continuous_batched",
        ":continuous_indexed",
        "//tensorflow_compression/python/ops:math_ops",
    ],
)

py_test(
    name = "universal_test",
    timeout = "long",
    srcs = ["universal_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        ":universal",
        "//tensorflow_compression/python/distributions:deep_factorized",
        "//tensorflow_compression/python/distributions:uniform_noise",
    ],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
