package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

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
