package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "deep_factorized",
    srcs = ["deep_factorized.py"],
    srcs_version = "PY3",
    deps = [
        ":helpers",
        ":uniform_noise",
    ],
)

py_test(
    name = "deep_factorized_test",
    srcs = ["deep_factorized_test.py"],
    python_version = "PY3",
    deps = [
        ":deep_factorized",
        ":helpers",
    ],
)

py_library(
    name = "helpers",
    srcs = ["helpers.py"],
    srcs_version = "PY3",
)

py_test(
    name = "helpers_test",
    srcs = ["helpers_test.py"],
    python_version = "PY3",
    deps = [
        ":deep_factorized",
        ":helpers",
    ],
)

py_library(
    name = "uniform_noise",
    srcs = ["uniform_noise.py"],
    srcs_version = "PY3",
    deps = [":helpers"],
)

py_test(
    name = "uniform_noise_test",
    srcs = ["uniform_noise_test.py"],
    python_version = "PY3",
    deps = [
        ":helpers",
        ":uniform_noise",
    ],
)

py_library(
    name = "round_adapters",
    srcs = ["round_adapters.py"],
    srcs_version = "PY3",
    deps = [
        ":deep_factorized",
        ":helpers",
        ":uniform_noise",
        "//tensorflow_compression/python/ops:soft_round_ops",
    ],
)

py_test(
    name = "round_adapters_test",
    srcs = ["round_adapters_test.py"],
    python_version = "PY3",
    deps = [
        ":deep_factorized",
        ":round_adapters",
        "//tensorflow_compression/python/ops:soft_round_ops",
    ],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
