# Examples

This directory is historical/reference material; the installable package and
its tests do not depend on it.

- `qwen3_0_6b/` — four hand-written Qwen3-0.6B decode engines (CUDA graph,
  TileOPs components, CPU, CPU-GPU hybrid) from the earlier iteration. They
  document the performance work that the current `breezetoken` package is
  gradually absorbing. Their kernels use the pre-refactor APIs and are kept
  for comparison, not as a supported entry point.

Run the current engine with `uv run breezetoken ...` (see the top-level
README); the tests under `tests/` are the supported validation path.
