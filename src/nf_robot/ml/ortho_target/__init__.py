"""Predict where the robot could reach next in the ortho floor view; readme.md has the pipeline."""

import importlib

# Old ortho_target.<name> lookups resolve lazily to the submodule that now holds the name.
_FORMER_CONTENTS = (
    f"{__name__}.model",
    f"{__name__}.dataset",
    f"{__name__}.training",
    f"{__name__}.__main__",
    "nf_robot.ml.image_input",
)


def __getattr__(name):
    if not name.startswith("__"):
        for module_name in _FORMER_CONTENTS:
            module = importlib.import_module(module_name)
            if name in vars(module):
                return vars(module)[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
