package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "util",
    srcs = ["__init__.py"],
    deps = [
        ":packed_tensors",
    ],
)

py_library(
    name = "packed_tensors",
    srcs = ["packed_tensors.py"],
)

py_test(
    name = "packed_tensors_test",
    srcs = ["packed_tensors_test.py"],
    deps = [":packed_tensors"],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
