"""Shared configuration for the tilelang CPU kernels.

The stock tilelang wheel ships a TVM built without LLVM codegen, so the CPU
target is the C backend.  There is deliberately no in-repo code that patches
site-packages to swap TVM libraries: that experiment left the environment
unbootable once its source tree moved.  If a source-built LLVM TVM is needed,
use a dedicated virtualenv and point ``TVM_LIBRARY_PATH`` at it.
"""

from __future__ import annotations

import os

DT = "float32"
TARGET = os.environ.get("BREEZETOKEN_CPU_TARGET", "c")
if TARGET != "c":
    raise RuntimeError(
        "only the 'c' CPU target is supported; a source-built LLVM TVM must be "
        "installed in a separate environment (see docs/architecture.md)"
    )

# -march=native is what lets gcc vectorize the scalar C output; it also makes
# the compiled kernel machine-specific.  Set BREEZETOKEN_CPU_NATIVE=0 for
# portable artifacts.
_CC_FLAGS = ["-O3"]
if os.environ.get("BREEZETOKEN_CPU_NATIVE", "1") != "0":
    _CC_FLAGS.append("-march=native")
_CC_FLAGS += ["-ffast-math"]
CC_FLAGS = _CC_FLAGS


def _exp(x):
    """exp() via libm; the C codegen only knows a few intrinsics."""
    import tilelang.language as T

    return T.call_extern("float32", "expf", x)
