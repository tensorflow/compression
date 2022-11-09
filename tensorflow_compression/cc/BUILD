package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "libtensorflow_compression.so",
    srcs = glob(
        include = [
            "kernels/*.h",
            "kernels/*.cc",
            "lib/*.h",
            "lib/*.cc",
            "ops/*.cc",
        ],
        exclude = [
            "**/*_test.cc",
        ],
    ),
    copts = [
        "-pthread",
        "-std=c++17",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        # When building this binary together with TensorFlow, i.e., when using
        # manually built TensorFlow, add the TensorFlow repo in WORKSPACE file
        # as, say "org_tensorflow" and replace the deps with
        #   "@org_tensorflow//tensorflow/core:framework",
        #   "@org_tensorflow//tensorflow/core:lib",
        "@tensorflow_pip//:libtensorflow_framework",
    ],
)

cc_library(
    name = "bit_coder",
    srcs = ["lib/bit_coder.cc"],
    hdrs = ["lib/bit_coder.h"],
    deps = [
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "run_length_gamma_kernels",
    srcs = [
        "kernels/run_length_gamma_kernels.cc",
        "ops/run_length_gamma_ops.cc",
    ],
    linkstatic = 1,
    deps = [
        ":bit_coder",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:framework_lite",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
    alwayslink = 1,
)
