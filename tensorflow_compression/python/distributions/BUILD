package(
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "distributions",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":deep_factorized",
        ":helpers",
        ":uniform_noise",
    ],
)

py_library(
    name = "deep_factorized",
    srcs = ["deep_factorized.py"],
    srcs_version = "PY3",
    deps = [":uniform_noise"],
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

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
