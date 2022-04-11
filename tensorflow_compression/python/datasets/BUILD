package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    deps = [
        ":y4m_dataset",
    ],
)

py_library(
    name = "y4m_dataset",
    srcs = ["y4m_dataset.py"],
    deps = ["//tensorflow_compression/python/ops:gen_ops"],
)

py_test(
    name = "y4m_dataset_test",
    srcs = ["y4m_dataset_test.py"],
    deps = [":y4m_dataset"],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
