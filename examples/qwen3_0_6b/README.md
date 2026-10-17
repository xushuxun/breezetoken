# Qwen3-0.6B on breezetoken — 移植自 TileFoundry 示例

从 TileFoundry `examples/qwen3_1_7b-tilelang`（MIT，见 `LICENSE.tilefoundry`）移植的
可运行推理引擎：整个 decode step 是一组 tilelang kernel + 单次 CUDA graph 捕获回放，
每步零 host 往返。原示例跑 Qwen3-1.7B（H200, 612 tok/s），此处跑本机 modelscope 缓存的
Qwen3-0.6B（RTX 4060 Laptop 8GB，~185 tok/s greedy）。

## 运行

```bash
uv run examples/qwen3_0_6b/run.py --prompt "中国大陆最南边的城市" --max-new-tokens 128
```

默认从 `~/.cache/modelscope/models/Qwen--Qwen3-0.6B` 读权重和 `tokenizer.json`，
可用 `--ckpt` 指定任意 Qwen3 dense 架构的 HF 目录。

## 移植改动（相对上游）

- **去掉 tilefoundry 依赖**：权重直读 safetensors（`safetensors`），config 直读
  `config.json`，tokenizer 用 `tokenizers` 库；上游的 HIR 参考/twin（`ref_src/`、
  `twin.py`）未拷贝
- **0.1.12 → 0.1.15 适配**：gamma 权重按 kernel 声明保持 bf16（上游 twin 全 f32）
- **SPLIT 256 → 128**：`attn_partial` 的 K/V shared 缓冲在上游配置下需 141KB
  动态 shared memory，超过 AD107 的 99KB opt-in 上限；128 档需要 ~73KB
- 新增依赖：`safetensors`、`tokenizers`

## 结构

- `kernels.py`：tilelang kernel 源码（原样拷贝：GEMV 系列、qk_rope_cache、
  attn_partial、lm_head+argmax、sample_step）
- `engine.py`：权重加载/repack、固定容量 KV cache、graph 捕获、generate
  （prefill/decode 分开计时）
- `run.py`：CLI 入口
