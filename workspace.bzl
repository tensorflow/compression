# -*- Python -*-
"""WORKSPACE setup functions."""

# @tf_custom_op needs to be setup in WORKSPACE.
load("@tf_custom_op//tf:tf_configure.bzl", "tf_configure")

def tf_compression_workspace():
  tf_configure(name = "local_config_tf")
