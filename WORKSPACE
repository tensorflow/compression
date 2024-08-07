workspace(name = "tensorflow_compression")

load("//:workspace.bzl", "tensorflow_compression_workspace")
tensorflow_compression_workspace()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# `bazel_skylib` and `rules_python` versions should match the ones used in
# `org_tensorflow`.
http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

http_archive(
    name = "rules_python",
    sha256 = "9d04041ac92a0985e344235f5d946f71ac543f1b1565f2cdbc9a2aaee8adf55b",
    strip_prefix = "rules_python-0.26.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.26.0/rules_python-0.26.0.tar.gz",
)

http_archive(
    name = "org_tensorflow",
    sha256 = "9cc4d5773b8ee910079baaecb4086d0c28939f024dd74b33fc5e64779b6533dc",
    strip_prefix = "tensorflow-2.17.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.17.0.tar.gz",
    ],
)

# Copied from `@org_tensorflow//:WORKSPACE`.
load(
    "@rules_python//python:repositories.bzl",
    "py_repositories",
    "python_register_toolchains",
)
py_repositories()

load(
    "@org_tensorflow//tensorflow/tools/toolchains/python:python_repo.bzl",
    "python_repository",
)
python_repository(name = "python_version_repo")
load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")

# TF workspace scripts below requires `@python` toolchains repo.
# Toolchain setup here is to please the TF workspace scripts,
# and we do not use this Python version to build pip packages.
python_register_toolchains(
    name = "python",
    ignore_root_user_error = True,
    python_version = HERMETIC_PYTHON_VERSION,
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()
