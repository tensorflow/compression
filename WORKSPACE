workspace(name = "tensorflow_compression")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "tensorflow_compression_workspace")
tensorflow_compression_workspace()

http_archive(
    name = "org_tensorflow",
    sha256 = "e58c939079588623e6fa1d054aec2f90f95018266e0a970fd353a5244f5173dc",
    strip_prefix = "tensorflow-2.13.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.13.0.tar.gz",
    ],
)

# Copied from `@org_tensorflow//:WORKSPACE`.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()
