# Qwen3.6-35B-A3B IQ3_S: native tilelang plan

> Status: **phases 1-5 implemented and validated.** `Engine.load` runs the
> checkpoint natively with hybrid placement (GPU non-expert bf16 + mmap
> quantized experts staged per token). Greedy output matches llama.cpp
> token-for-token. Phases 6-7 (batched prefill, graphs, tilelang quant
> kernels, expert caching) remain.

## 1. Why this is a separate project-sized path

The checkpoint is `Qwen3.6-35B-A3B-UD-IQ3_S.gguf`:

- 15 GiB on disk, 40 decoder layers + 1 MTP layer (MTP ignored for inference);
- 30 GDN layers + 10 GQA layers;
- hidden 2048, 16 Q heads, 2 KV heads, head_dim 256, partial RoPE 64;
- GDN 16 key heads / 32 value heads, K/V dim 128, conv 4;
- 256 experts, top-8, expert FF 512, shared expert FF 512;
- mixed quant types: IQ3_S, IQ2_S, IQ4_XS, Q2_K, Q4_K, Q6_K, Q8_0, BF16, F32.

On this host (RTX 4060 Laptop 8 GB, 23 GiB RAM) a bf16 copy is ~70 GiB and f32
is ~140 GiB. The engine therefore has to keep the **quantized bytes** and fuse
dequantization into the matmul; it cannot reuse the native bf16/f32 loader.

TileFoundry's 35B example is not a shortcut here: it runs bf16 safetensors on an
H200. The model *shape* is reusable; the weight representation is not.

## 2. Scope

In scope:

- native GGUF loading of the quant types in this checkpoint, via mmap;
- quantized GEMV on CPU and CUDA for decode;
- Qwen3.6 MoE router, 256 experts, top-8 and shared expert;
- GDN 16K/32V and attention 16Q/2KV with output gate;
- component-level hybrid placement (non-expert tensors on GPU, expert tensors
  quantized on CPU, or the reverse when it fits);
- tokenizer/chat/OpenAI server integration through the existing `Engine`;
- correctness tests and a repeatable benchmark.

Out of scope initially:

- vision/audio, MTP/nextn draft, batched speculative decoding;
- a native C++/AVX library separate from tilelang;
- converting the checkpoint to bf16/f32 on disk.

## 3. Quant representation

All K-quants and I-quants use `QK_K = 256`; Q8_0 uses 32. Sizes from
`ggml/src/ggml-common.h`:

| GGUF name | type id | elements/block | bytes/block |
|---|---:|---:|---:|
| F32 | 0 | 1 | 4 |
| Q8_0 | 8 | 32 | 34 |
| Q2_K | 10 | 256 | 84 |
| Q4_K | 12 | 256 | 144 |
| Q6_K | 14 | 256 | 210 |
| IQ3_S | 21 | 256 | 110 |
| IQ2_S | 22 | 256 | 82 |
| IQ4_XS | 23 | 256 | 136 |
| BF16 | 30 | 1 | 2 |

Key structs (little-endian):

- `block_iq2_s`: `half d; u8 qs[64]; u8 qh[8]; u8 scales[8]`;
- `block_iq3_s`: `half d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]`;
- `block_iq4_xs`: `half d; u16 scales_h; u8 scales_l[4]; u8 qs[128]`;
- `block_q2_K`: `u8 scales[16]; u8 qs[64]; half d; half dmin`;
- `block_q4_K`: `half d; half dmin; u8 scales[12]; u8 qs[128]`;
- `block_q6_K`: `u8 ql[128]; u8 qh[64]; i8 scales[16]; half d`.

The IQ formats need the lookup grids/sign tables from `ggml-quants.c`
(`iq3s_grid`, `iq2s_grid`, `kvalues_iq4nl`). Phase 1 ports them into a
reference dequantizer; those tables are data, not a kernel.

Tensors are row-major after GGUF's reversed-dimension reconstruction. Quant
blocks run along the last dimension. For an expert tensor shaped
`(experts, out, in)`, each `(expert, out)` row has `ceil(in / block_elems)`
blocks. Never materialize the full 3-D tensor.

## 4. Kernels

### 4.1 `QuantizedTensor` / layout

`breezetoken/quant/tensor.py` exposes a descriptor: name, shape, format, mmap
offset, row bytes, block count. `row_slice(row)` returns a byte range. Loading a
layer keeps descriptors only; pages are faulted on demand.

### 4.2 CPU reference dequant

`breezetoken/quant/dequant.py`:

```python
def dequantize_row(raw: memoryview, fmt: QuantFormat, out: Tensor) -> Tensor
```

Used first for unit vectors and per-layer golden tests. It is intentionally
slow and never used in the decode hot path.

### 4.3 CPU quantized GEMV

`breezetoken/kernels/cpu/quant_gemv.py`:

```python
def quant_gemv(x, rows, out, fmt, ...): ...
```

One tilelang `target="c"` kernel per (quant format, K/N tile). Each block decodes
its quant block into registers/shared, multiplies by the corresponding `x`
slice, and accumulates. It must not allocate a f32 weight row. Phase 2 includes a
fallback per-format reference and a benchmark; if tilelang-C codegen is too slow,
the kernel can be split into a small C helper with the same interface, but the
default remains tilelang.

### 4.4 CUDA quantized GEMV

`breezetoken/kernels/cuda/quant_gemv.py` mirrors the CPU contract. A warp decodes
one or more 256-element blocks into shared memory, then a dot-product/warp
reduction writes one output row. The first target is decode (`M=1`); prefill
GEMM (`M=S`) is phase 6.

### 4.5 MoE

`breezetoken/kernels/cuda/moe.py` and `breezetoken/model/moe.py` define:

- router: `logits = x @ gate_inp` (f32, 2048x256), softmax/top-8;
- expert gate/up: selected `(out=512, in=2048)` quantized rows;
- expert down: selected `(out=2048, in=512)` quantized rows;
- shared expert: Q6_K gate/up/down + scalar sigmoid gate
  (`ffn_gate_inp_shexp`);
- combine: `sum_i w_i * expert_i(x) + sigmoid(x @ shared_gate) * shared(x)`.

For decode, selected experts are a 8-element list per layer. The CPU path runs
`quant_gemv` for each selected expert; the GPU path may stage the selected
expert's quantized rows to VRAM, but only after measuring PCIe traffic. The
router and shared expert stay in the non-expert placement.

## 5. Model layers

`breezetoken/model/qwen36_moe.py`:

- `Qwen36Config.from_gguf`: parse the `qwen35moe.*` metadata, including
  GDN key/value heads, `attn_output_gate`, expert counts and nextn;
- GDN projection layout: q 2048, k 2048, v 4096 inside `attn_qkv`; z 4096;
  out 4096->2048; `ssm_alpha/beta` 32; state `(32,128,128)`;
- attention q 8192 (q + gate), k/v 512 each, output 4096->2048;
- no MTP block in the main loop.

The existing `kernels/cuda/gdn.py` already supports 16 key / 32 value heads. The
native `layers.py` needs a 35B variant for the wider QKV and output gate.

## 6. Placement

`planner.py` is layer-granular today. The 35B path needs **component-granular**
placement:

- non-expert tensors (attention/GDN/shared/head) are small and should be on GPU;
- expert tensors are ~13 GiB and must stay quantized on CPU;
- KV/GDN state follows the owning component;
- the router stays with the non-expert side;
- selected expert rows can optionally be staged to GPU later.

Initial target: GPU non-expert + CPU quantized experts, with the residual stream
crossing only at the expert boundary. A CPU-only mode is required for CI.

## 7. Phases and acceptance

### Phase 0 - remove the llama.cpp call (done in this change)

- no runtime subprocess, no `httpx` server dependency, no llama license;
- `Engine.load` returns a clear `NotImplementedError` for `qwen35moe`;
- placeholder modules import cleanly and raise `NotImplementedError` only when
  called.

### Phase 1 - format + dequant reference

> Done. `quant/dequant.py` decodes all nine formats with vectorized torch ops
> (CPU/CUDA); IQ grids are generated from ggml-common.h by
> `scripts/gen_qtables.py`. A differential tool (`scripts/ggml_ref_dump.cpp`,
> links only `ggml-quants.c`) reproduces ggml's `dequantize_row_*` on real
> checkpoint rows: every format matches **bit-for-bit** (12 tensors sampled).

- format table and descriptor;
- reference dequant for F32/BF16/Q8_0/Q2_K/Q4_K/Q6_K/IQ2_S/IQ3_S/IQ4_XS;
- unit vectors for every block format (including IQ grids/signs).

Acceptance: dequantized rows match a one-time reference fixture bit-for-bit
within f32 tolerance for random blocks.

### Phase 2 - quantized GEMV

> Done. `kernels/cpu/quant_gemv.py` keeps the reference contract (vectorized
> dequant + matmul, one f32 row tile at a time).  `kernels/cuda/quant_gemv.py`
> adds **fused tilelang CUDA kernels** for the two expert formats in this
> checkpoint -- `iq2s_gemv` (gate/up, K=2048) and `iq4xs_gemv` (down, K=512)
> decode the quant blocks in-registers and reduce the dot with `x` in shared
> memory; no f32 weight tile is ever materialized.  `model/moe.py::ExpertPool`
> stages the selected experts' raw bytes through one pinned buffer + one H2D
> per layer and calls the kernels directly.  Experts cannot be GPU-resident:
> 13 GiB of quant bytes plus ~4.7 GiB non-expert bf16 plus KV/workspace
> exceeds the 8 GiB card, and measured staging is only ~1 ms/layer anyway --
> the old bottleneck was torch dequant op overhead (~15 ms/layer), which the
> fused kernels removed.  Steady-state decode on the RTX 4060 Laptop:
> **~5 tok/s** (was ~0.4 tok/s with the first correct version).

- CPU/CUDA kernels for one row and one expert;
- no full-weight temporary; peak allocation proportional to an output row tile;
- benchmark decode-shaped (M=1) and document tok/s.

Acceptance: max relative error vs reference dequant+matmul within 1e-3 for
Q6_K/IQ4_XS; exact for Q8_0/f32; no >2x slowdown vs a dequantized bf16 kernel.

### Phase 3 - 35B non-expert layers

> Done. `model/qwen36_moe.py` implements GDN (16 key heads, 32 value heads --
> value head *v* reads key head *v % 16*, matching ggml's fused
> `gated_delta_net`, not GQA-style grouping), attention 16Q/2KV with the
> per-head `[q | output-gate]` projection, and partial RoPE. Verified
> end-to-end against llama.cpp greedy tokens.

### Phase 4 - MoE

> Done. `model/moe.py`: softmax router with top-8 + renormalization (matches
> `build_moe_ffn` SOFTMAX gating; `expert_weights_scale` is unset/1.0 in this
> checkpoint), shared expert with scalar sigmoid gate, and `ExpertPool` which
> stages the selected experts' quant bytes to the compute device and runs
> dequant+matmul there (gate+up fused into one GEMV).

### Phase 5 - full model

> Done. 40 layers (MTP block skipped), expert tensors stay mmap'd quant
> bytes, non-expert dequantized to bf16 on GPU (~4.7 GiB including the
> 248320-token embedding and lm_head). Prompt is processed token-by-token
> (batched prefill is phase 6). Acceptance: greedy 8-token continuation of
> "The capital of France is" matches llama.cpp (`llama-server`
> `/v1/completions`, temperature 0) token-for-token; steady-state decode
> **~5 tok/s** on the RTX 4060 Laptop (0.4 -> 2.1 -> 5.0 across the fused
> expert GEMV work); host RSS bounded by page cache (the 15 GiB mapping is
> not copied).

- 40 layers, mmap descriptors, no full materialization;
- component placement; server/CLI integration;
- generation must produce the same first tokens as the offline reference.

Acceptance: load under 23 GiB host, generate without OOM, and document peak RSS
and tok/s. First target is correctness; GPU expert staging is phase 7.

### Phase 6 - prefill / graph

> Partly done. Batched prefill (prompt >= 8 tokens) batches attention/GDN
> projections, the conv1d, the shared expert and the head; expert execution
> dequantizes all selected experts in one shot (bf16 tile-dequant kernels)
> then runs per-expert GEMMs; the GDN delta-rule recurrence stays sequential
> per token (a chunked scan is the path to go faster).  Measured on the
> RTX 4060 Laptop: ~17 tok/s at S=128, ~36 tok/s at S=256 (prompt-length
> bound by the per-token GDN loop and expert staging over PCIe).
> **Prefill >1000 tok/s is not reachable on this host**: every prefill batch
> must move the touched expert weights (~10 MiB/token) across PCIe/page
> cache, and the 23 GiB RAM cannot hold working sets for the batch sizes
> that would amortize it.  CUDA graph for the S=1 decode step is the next
> item (kills the remaining per-op launch overhead).

- batched prompt projections and MoE expert batching;
- CUDA graph only if all compute is on GPU; otherwise keep eager decode.

### Phase 7 - performance

> In progress. Done: fused decode quant-GEMV kernels (iq2_s/iq4_xs),
> gate/up single GEMV, pinned single-H2D staging, OpenMP-parallel host
> copies, split-K head GEMV, tilelang GDN recurrence.  Decode went
> 0.4 -> 2.1 -> **~6 tok/s** (greedy, RTX 4060 Laptop).
> MTP/nextn speculative decoding (`--mtp`, greedy-only) is implemented and
> **lossless** (output identical to plain greedy), with measured draft-1
> acceptance ~0.88, but is not yet profitable (verify of 2 tokens plus
> state-replay costs ~4x a plain step in this architecture) — it is
> disabled by default until: (a) per-position GDN state snapshots replace
> the replay, (b) the fused expert kernels gain an M<=D variant so verify
> costs ~one step, (c) the decode step is graph-captured and read-bound.
> The ~52 steps/s raw ceiling (4.7 GiB bf16 weights / ~250 GB/s) bounds
> decode at ~50 tok/s effective only with MTP acceptance >= ~2.

- Q8_0/IQ4_XS fast paths, selected-expert GPU staging, CPU AVX/threading,
  prefix cache.

## 8. Risks

- **Memory bandwidth**: 8 selected experts x 3 matrices x 40 layers is ~1.3 GiB
  of quantized reads per token; CPU bandwidth may cap generation at a few tok/s.
- **Page cache**: 15 GiB file on a 23 GiB host; mmap page eviction can cause
  stalls. `mlock` is not viable for the whole file.
- **IQ dequant complexity**: the lookup grids and sign handling are the highest
  correctness risk. Reference vectors are mandatory before kernels.
- **tilelang C backend**: it is single-threaded today. CPU expert GEMV may need
  a thread pool or a measured fallback; that decision is phase 2, not assumed.
- **GPU/CPU transfer**: staging selected experts may cost more than CPU compute;
  measure before implementing.

## 9. Implementation files

```
breezetoken/quant/
  formats.py           # GGML type table
  _tables.py           # IQ grids generated by scripts/gen_qtables.py
  tensor.py            # QuantizedTensor descriptor / row-tile loading
  dequant.py           # vectorized decoders (all nine formats)
breezetoken/kernels/cpu/quant_gemv.py   # dequant+matmul contract, CPU
breezetoken/kernels/cuda/quant_gemv.py  # pinned staging + on-device dequant
breezetoken/model/moe.py                # softmax router + ExpertPool staging
breezetoken/model/qwen36_moe.py         # Qwen36Moe hybrid decoder
scripts/gen_qtables.py                  # ggml-common.h -> _tables.py
scripts/ggml_ref_dump.cpp             # differential dequant reference tool
tests/test_quant_dequant.py / test_quant_gemv.py / test_moe.py / test_qwen36_moe.py
```

The existing native 0.8B path (`model/{config,layers,prefill,graph,qwen35}.py`)
is unchanged and remains the reference implementation.
