workspace(name = "tensorflow_compression")

load("//:workspace.bzl", "tensorflow_compression_workspace")
tensorflow_compression_workspace()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "org_tensorflow",
    sha256 = "d7876f4bb0235cac60eb6316392a7c48676729860da1ab659fb440379ad5186d",
    strip_prefix = "tensorflow-2.18.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.18.0.tar.gz",
    ],
)

# Copied from `@org_tensorflow//:WORKSPACE`.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

# Initialize hermetic Python
load("@local_tsl//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@local_tsl//third_party/py:python_init_repositories.bzl", "python_init_repositories")
python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.9": "@org_tensorflow//:requirements_lock_3_9.txt",
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
    },
)

load("@local_tsl//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")
python_init_toolchains()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

load(
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)
cuda_configure(name = "local_config_cuda")
