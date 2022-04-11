package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_compression",
    srcs = ["tensorflow_compression/__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_compression/python/datasets",
        "//tensorflow_compression/python/distributions",
        "//tensorflow_compression/python/entropy_models",
        "//tensorflow_compression/python/layers",
        "//tensorflow_compression/python/ops",
        "//tensorflow_compression/python/util",
    ],
)

py_binary(
    name = "build_api_docs",
    srcs = ["tools/build_api_docs.py"],
    deps = [":tensorflow_compression"],
)

py_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.py"],
    data = [
        "LICENSE",
        "README.md",
        "MANIFEST.in",
        "requirements.txt",
        "tensorflow_compression/all_tests.py",
        ":tensorflow_compression",
        # The following targets are for Python unit tests.
        "//tensorflow_compression/python/datasets:py_src",
        "//tensorflow_compression/python/distributions:py_src",
        "//tensorflow_compression/python/entropy_models:py_src",
        "//tensorflow_compression/python/layers:py_src",
        "//tensorflow_compression/python/ops:py_src",
        "//tensorflow_compression/python/util:py_src",
    ],
)
