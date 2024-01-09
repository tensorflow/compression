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
        "//tensorflow_compression/python/datasets:y4m_dataset",
        "//tensorflow_compression/python/distributions",
        "//tensorflow_compression/python/distributions:deep_factorized",
        "//tensorflow_compression/python/distributions:helpers",
        "//tensorflow_compression/python/distributions:round_adapters",
        "//tensorflow_compression/python/distributions:uniform_noise",
        "//tensorflow_compression/python/entropy_models",
        "//tensorflow_compression/python/entropy_models:continuous_batched",
        "//tensorflow_compression/python/entropy_models:continuous_indexed",
        "//tensorflow_compression/python/entropy_models:power_law",
        "//tensorflow_compression/python/entropy_models:universal",
        "//tensorflow_compression/python/layers",
        "//tensorflow_compression/python/layers:gdn",
        "//tensorflow_compression/python/layers:initializers",
        "//tensorflow_compression/python/layers:parameters",
        "//tensorflow_compression/python/layers:signal_conv",
        "//tensorflow_compression/python/layers:soft_round",
        "//tensorflow_compression/python/ops",
        "//tensorflow_compression/python/ops:gen_ops",
        "//tensorflow_compression/python/ops:math_ops",
        "//tensorflow_compression/python/ops:padding_ops",
        "//tensorflow_compression/python/ops:round_ops",
        "//tensorflow_compression/python/util:packed_tensors",
    ],
)
