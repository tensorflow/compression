licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "libtensorflow_compression.so",
    srcs = glob(
        include = [
            "**/*.h",
            "**/*.cc",
        ],
        exclude = ["**/*test.cc"],
    ),
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    linkstatic = 0,
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
