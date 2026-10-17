"""breezetoken: a small, local LLM inference engine.

The package is organised in one direction:

    gguf / tokenizer -> model -> engine -> cli / server
                          ^
                     backends -> kernels

Only the backend imports the compiled kernel modules; the model and everything
above it stay backend-agnostic.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("breezetoken")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.2.0"

__all__ = ["__version__"]
