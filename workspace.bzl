"""WORKSPACE setup functions."""

def _tensorflow_pip_impl(ctx):
  python_program = ctx.which(ctx.attr.python_program)

  library_path = ctx.execute([
      python_program,
      "-c",
      "import tensorflow; print(tensorflow.sysconfig.get_lib())",
  ])
  include_path = ctx.execute([
      python_program,
      "-c",
      "import tensorflow; print(tensorflow.sysconfig.get_include())",
  ])

  if library_path.return_code != 0:
    fail("Failed to find library path. Did you remember to pip install " +
         "tensorflow?: %s" % library_path.stderr)
  if include_path.return_code != 0:
    fail("Failed to find include path. Did you remember to pip install " +
         "tensorflow?: %s" % include_path.stderr)

  if "linux" in ctx.os.name:
    library_filename = "libtensorflow_framework.so.2"
  elif "mac" in ctx.os.name:
    library_filename = "libtensorflow_framework.2.dylib"

  ctx.symlink("/".join([library_path.stdout.strip(), library_filename]),
              library_filename)
  ctx.symlink(include_path.stdout.strip(), "include")
  ctx.file("BUILD", """
cc_library(
    name = "libtensorflow_framework",
    srcs = ["{0}"],
    hdrs = glob(["include/**"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""".format(library_filename))

tensorflow_pip = repository_rule(
    implementation = _tensorflow_pip_impl,
    attrs = {
        "python_program": attr.string(default = "python3"),
    },
)

def tensorflow_compression_workspace():
  tensorflow_pip(
      name = "tensorflow_pip",
      python_program = "python3",
  )
