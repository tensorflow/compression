licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_compression",
    srcs = [
        "tensorflow_compression/__init__.py",
        "tensorflow_compression/python/__init__.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_compression/python/distributions",
        "//tensorflow_compression/python/entropy_models",
        "//tensorflow_compression/python/layers",
        "//tensorflow_compression/python/ops",
        "//tensorflow_compression/python/util",
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
        "//tensorflow_compression/python/distributions:py_src",
        "//tensorflow_compression/python/entropy_models:py_src",
        "//tensorflow_compression/python/layers:py_src",
        "//tensorflow_compression/python/ops:py_src",
        "//tensorflow_compression/python/util:py_src",
    ],
    python_version = "PY3",
)

py_binary(
    name = "generate_docs",
    srcs = ["tools/generate_docs.py"],
    python_version = "PY3",
    deps = [":tensorflow_compression"],
)

py_binary(
    name = "tfci",
    srcs = ["models/tfci.py"],
    python_version = "PY3",
    deps = [":tensorflow_compression"],
)

py_binary(
    name = "bls2017",
    srcs = ["models/bls2017.py"],
    python_version = "PY3",
    deps = [":tensorflow_compression"],
)

py_binary(
    name = "bmshj2018",
    srcs = ["models/bmshj2018.py"],
    python_version = "PY3",
    deps = [":tensorflow_compression"],
)
