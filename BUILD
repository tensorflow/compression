licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "_range_coding_ops.so",
    srcs = [
        "cc/kernels/range_coder.cc",
        "cc/kernels/range_coder.h",
        "cc/kernels/range_coding_helper_kernels.cc",
        "cc/kernels/range_coding_kernels_util.cc",
        "cc/kernels/range_coding_kernels_util.h",
        "cc/kernels/range_coding_kernels.cc",
        "cc/kernels/unbounded_index_range_coding_kernels.cc",
        "cc/ops/range_coding_ops.cc",
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
        "python/__init__.py",
        "python/layers/__init__.py",
        "python/layers/entropy_models.py",
        "python/layers/gdn.py",
        "python/layers/initializers.py",
        "python/layers/parameterizers.py",
        "python/layers/signal_conv.py",
        "python/ops/__init__.py",
        "python/ops/_namespace_helper.py",
        "python/ops/math_ops.py",
        "python/ops/padding_ops.py",
        "python/ops/range_coding_ops.py",
        "python/ops/spectral_ops.py",
    ],
    data = [
        ":_range_coding_ops.so"
    ],
)

py_test(
    name = "entropy_models_test",
    timeout = "long",
    srcs = ["python/layers/entropy_models_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "gdn_test",
    srcs = ["python/layers/gdn_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "parameterizers_test",
    srcs = ["python/layers/parameterizers_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "signal_conv_test",
    timeout = "long",
    srcs = ["python/layers/signal_conv_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "math_ops_test",
    srcs = ["python/ops/math_ops_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "padding_ops_test",
    srcs = ["python/ops/padding_ops_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "spectral_ops_test",
    srcs = ["python/ops/spectral_ops_test.py"],
    deps = [":tensorflow_compression"],
)

py_test(
    name = "range_coding_ops_test",
    srcs = ["python/ops/range_coding_ops_test.py"],
    deps = [":tensorflow_compression"],
)
