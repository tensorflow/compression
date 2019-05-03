licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "_range_coding_ops.so",
    srcs = [
        "tensorflow_compression/cc/kernels/range_coder.cc",
        "tensorflow_compression/cc/kernels/range_coder.h",
        "tensorflow_compression/cc/kernels/range_coding_helper_kernels.cc",
        "tensorflow_compression/cc/kernels/range_coding_kernels_util.cc",
        "tensorflow_compression/cc/kernels/range_coding_kernels_util.h",
        "tensorflow_compression/cc/kernels/range_coding_kernels.cc",
        "tensorflow_compression/cc/kernels/unbounded_index_range_coding_kernels.cc",
        "tensorflow_compression/cc/ops/range_coding_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = [
        "-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-Wno-sign-compare", "-Wno-maybe-uninitialized",
    ],
)

py_library(
    name = "tensorflow_compression",
    srcs = [
        "__init__.py",
        "tensorflow_compression/__init__.py",
        "tensorflow_compression/python/__init__.py",
        "tensorflow_compression/python/layers/__init__.py",
        "tensorflow_compression/python/layers/entropy_models.py",
        "tensorflow_compression/python/layers/gdn.py",
        "tensorflow_compression/python/layers/initializers.py",
        "tensorflow_compression/python/layers/parameterizers.py",
        "tensorflow_compression/python/layers/signal_conv.py",
        "tensorflow_compression/python/ops/__init__.py",
        "tensorflow_compression/python/ops/namespace_helper.py",
        "tensorflow_compression/python/ops/math_ops.py",
        "tensorflow_compression/python/ops/padding_ops.py",
        "tensorflow_compression/python/ops/range_coding_ops.py",
        "tensorflow_compression/python/ops/spectral_ops.py",
    ],
    data = [
        ":_range_coding_ops.so"
    ],
)

sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.sh"],
    data = [
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "setup.py",
        ":tensorflow_compression",
    ],
)

py_test(
    name = "entropy_models_test",
    timeout = "long",
    srcs = ["tensorflow_compression/python/layers/entropy_models_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "gdn_test",
    srcs = ["tensorflow_compression/python/layers/gdn_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "parameterizers_test",
    srcs = ["tensorflow_compression/python/layers/parameterizers_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "signal_conv_test",
    timeout = "long",
    srcs = ["tensorflow_compression/python/layers/signal_conv_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "math_ops_test",
    srcs = ["tensorflow_compression/python/ops/math_ops_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "padding_ops_test",
    srcs = ["tensorflow_compression/python/ops/padding_ops_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "spectral_ops_test",
    srcs = ["tensorflow_compression/python/ops/spectral_ops_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "range_coding_ops_test",
    srcs = ["tensorflow_compression/python/ops/range_coding_ops_test.py"],
    deps = [":tensorflow_compression"],
)
