workspace(name = "tensorflow_compression")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "tensorflow_compression_workspace")
tensorflow_compression_workspace()

http_archive(
    name = "org_tensorflow",
    sha256 = "ce357fd0728f0d1b0831d1653f475591662ec5bca736a94ff789e6b1944df19f",
    strip_prefix = "tensorflow-2.14.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.14.0.tar.gz",
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
