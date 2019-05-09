licenses(["notice"])  # Apache 2.0

py_library(
    name = "tensorflow_compression",
    srcs = [
        "tensorflow_compression/__init__.py",
        "tensorflow_compression/python/__init__.py",
    ],
    deps = [
        "//tensorflow_compression/python/layers",
        "//tensorflow_compression/python/ops",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "pip_src",
    srcs = [
        "MANIFEST.in",
        "setup.py",
        "tensorflow_compression/python/all_test.py",
    ],
)

sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.sh"],
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
