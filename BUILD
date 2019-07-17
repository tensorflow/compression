licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_compression",
    srcs = [
        "tensorflow_compression/__init__.py",
        "tensorflow_compression/python/__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_compression/python/layers",
        "//tensorflow_compression/python/ops",
    ],
)

filegroup(
    name = "pip_src",
    srcs = [
        "MANIFEST.in",
        "tensorflow_compression/python/all_test.py",
    ],
)

py_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.py"],
    data = [
        "LICENSE",
        "README.md",
        ":pip_src",
        ":tensorflow_compression",
        # The following targets are for Python test files.
        "//tensorflow_compression/python/layers:py_src",
        "//tensorflow_compression/python/ops:py_src",
    ],
)

py_binary(
    name = "tfci",
    srcs = ["examples/tfci.py"],
    deps = [":tensorflow_compression"],
)

py_binary(
    name = "generate_docs",
    srcs = ["tools/generate_docs.py"],
    deps = [":tensorflow_compression"],
)
