package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "entropy_models",
    srcs = ["__init__.py"],
    deps = [
        ":continuous_batched",
        ":continuous_indexed",
        ":power_law",
        ":universal",
    ],
)

py_library(
    name = "continuous_base",
    srcs = ["continuous_base.py"],
    deps = [
        "//tensorflow_compression/python/distributions:helpers",
        "//tensorflow_compression/python/distributions:uniform_noise",
        "//tensorflow_compression/python/ops:gen_ops",
    ],
)

py_library(
    name = "continuous_batched",
    srcs = ["continuous_batched.py"],
    deps = [
        ":continuous_base",
        "//tensorflow_compression/python/distributions:helpers",
        "//tensorflow_compression/python/ops:gen_ops",
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:round_ops",
    ],
)

py_test(
    name = "continuous_batched_test",
    srcs = ["continuous_batched_test.py"],
    deps = [
        ":continuous_batched",
        "//tensorflow_compression/python/distributions:uniform_noise",
    ],
)

py_library(
    name = "continuous_indexed",
    srcs = ["continuous_indexed.py"],
    deps = [
        ":continuous_base",
        "//tensorflow_compression/python/ops:gen_ops",
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:round_ops",
    ],
)

py_test(
    name = "continuous_indexed_test",
    timeout = "long",
    srcs = ["continuous_indexed_test.py"],
    shard_count = 6,
    deps = [
        ":continuous_indexed",
        "//tensorflow_compression/python/distributions:uniform_noise",
    ],
)

py_library(
    name = "power_law",
    srcs = ["power_law.py"],
    deps = [
        "//tensorflow_compression/python/ops:gen_ops",
        "//tensorflow_compression/python/ops:round_ops",
    ],
)

py_test(
    name = "power_law_test",
    srcs = ["power_law_test.py"],
    deps = [":power_law"],
)

py_library(
    name = "universal",
    srcs = ["universal.py"],
    deps = [
        ":continuous_base",
        "//tensorflow_compression/python/ops:gen_ops",
        "//tensorflow_compression/python/ops:math_ops",
    ],
)

py_test(
    name = "universal_test",
    timeout = "long",
    srcs = ["universal_test.py"],
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
