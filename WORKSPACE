workspace(name = "tensorflow_compression")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "tensorflow_compression_workspace")
tensorflow_compression_workspace()

http_archive(
    name = "org_tensorflow",
    sha256 = "b5a1bb04c84b6fe1538377e5a1f649bb5d5f0b2e3625a3c526ff3a8af88633e8",
    strip_prefix = "tensorflow-2.10.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.10.0.tar.gz",
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
