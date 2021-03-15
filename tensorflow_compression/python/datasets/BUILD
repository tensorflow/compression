package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "y4m_dataset",
    srcs = ["y4m_dataset.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_compression/python/ops:gen_ops"],
)

py_test(
    name = "y4m_dataset_test",
    srcs = ["y4m_dataset_test.py"],
    python_version = "PY3",
    deps = [":y4m_dataset"],
)

filegroup(
    name = "py_src",
    srcs = glob(["*.py"]),
)
