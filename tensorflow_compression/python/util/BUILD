package(
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "util",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":packed_tensors",
    ],
)

py_library(
    name = "packed_tensors",
    srcs = ["packed_tensors.py"],
    srcs_version = "PY3",
)

py_test(
    name = "packed_tensors_test",
    srcs = ["packed_tensors_test.py"],
    python_version = "PY3",
    deps = [":packed_tensors"],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
