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

## engine_tileops.py：TileOPs 组件替换版

`engine.py` 的变体，模型组件替换为从 TileOPs（MIT）COPY 到 `tileops_lite/`
的 kernel（TileOPs 要求 Python <3.13 装不进本环境，故直接拷贝源码适配）：

| 组件 | 原实现（TileFoundry 移植） | 替换为（TileOPs） |
|---|---|---|
| RMSNorm | `rms_norm` | `tileops_lite RMSNormKernel` |
| residual + RMSNorm ×2 | `resid_rms_norm` | `tileops_lite FusedAddRMSNormKernel` |
| attention | `attn_partial` + `gemv_attn_combine` 融合 o_proj | `tileops_lite GQADecodeKernel`（split-K online-softmax），o_proj 变为独立 GEMV |

GEMV 系列（qkv/gate_up/down 含融合 silu）、qk_rope_cache、lm_head、embed、
采样保持 TileFoundry 移植版不变。

结构差异：TileOPs attention 按真实 KV 长度调用，context length 是 host 值，
无法进 CUDA graph，故该引擎**eager 执行**（无 graph 捕获）——这是用它换
组件的代价，计时对比如下（RTX 4060, greedy, 32 tokens）：

| 引擎 | decode tok/s | 说明 |
|---|---|---|
| engine.py（graph） | ~185 | 整步一次 graph 回放 |
| engine_tileops.py（eager） | ~100-107 | 每步 ~300 次 kernel launch |

三实现完整对比（`bench_3way.py`，subprocess 隔离 + 全路径预热，RTX 4060
Laptop，greedy 128 新 token，decode 速率已剔除 prefill）：

| 实现 | 短 prompt(5) prefill | decode | 长 prompt(233) prefill | decode |
|---|---|---|---|---|
| engine.py（TileFoundry kernel + graph） | 21 ms | **188.6 tok/s** | 1232 ms | **184.8 tok/s** |
| engine_tileops.py（TileOPs 组件 + graph） | 23 ms | **171.7 tok/s** | 1392 ms | **165.4 tok/s** |
| engine_hybrid.py（alternating 跨设备） | 1047 ms | 3.9 tok/s | 56.2 s | 4.1 tok/s |
| engine_cpu.py（tilelang CPU c 后端） | 1491 ms | 2.6 tok/s | 83.5 s | 2.9 tok/s |
| transformers（HF bf16, eager attn） | 1.9 ms | 27.7 tok/s | 8.8 ms | 27.6 tok/s |

（五实现对比含 CPU 与混合版，见下文"engine_cpu / engine_hybrid"节，跑法
`uv run --with transformers python examples/qwen3_0_6b/bench_3way.py`。）

## engine_cpu.py：tilelang CPU 推理版

全部模型组件是 tilelang `target="c"` 编译的 CPU kernel（`kernels_cpu.py`）：

- **f32 端到端**——C codegen 不支持 bfloat16
- **性能靠编译 flags**：默认生成标量代码 ~0.8 GFLOPS；`CC_FLAGS=["-O3",
  "-march=native", "-ffast-math"]` 后 gcc 自动向量化，~8 GFLOPS（单核带宽
  上限）。`target="llvm"` 在本安装的 TVM 构建中不可用（vendored TVM 未编
  LLVM；系统装 LLVM 不改变这一点）
- 数学库只有 `rsqrt` 内建，`exp` 等经 `T.call_extern("expf", ...)` 走 libm
- **注意**：`-ffast-math` 下 silu 必须对输入 clamp（`max(g, -80)`）——编译器
  重结合会先把 `expf` 溢出变成 `inf*0=NaN`（奇数步正常、偶数步全 NaN 的
  诡异现象就是这么来的；NaN 不污染 KV cache 所以下一步还能"自愈"）
- 实测 ~2.7 tok/s（0.6B 全量 CPU，计算受限）；对 HF 数值完全一致

## engine_hybrid.py：CPU-GPU 混合推理版

组合而非重写：持有 `EngineTileOps`（GPU 半）+ `EngineCPU`（CPU 半），驱动
两者的 `forward_layer(i, xn, resid)` 逐层执行。残差流以 **f32 主副本驻留
host**，每层边界按需迁移：GPU 层收 bf16 (xn, resid) 返回新张量再转回 f32。
每边界 2×H2D + 2×D2H（各 2KB）+ 一次同步，开销 ~100µs 量级。

device map：`all-gpu` / `all-cpu`（正确性对照）/ `alternating`（偶数层 GPU、
奇数层 CPU，最大真实跨设备流量）。

验证：`alternating` 与 `all-gpu`/`all-cpu` 输出一致且与 HF 同精度级别
（`compare_transformers.py --hybrid`）。

## 解读与下一步

- **decode**：graph 版最快（整步一次回放，无 launch 开销）；TileOPs eager 版
  每步 ~300 次 Python launch，约为 graph 版 55%；HF 最慢（通用算子 + 大 Python 开销）；
  CPU 版受 tilelang C 后端单线程限制 ~2.7 tok/s；混合版由最慢的那一半决定
- **prefill**：自研引擎都是 S=1 逐 token 设计，233 token prompt 逐步走
  （1.2s/2.2s）；HF 批量 prefill 约 10ms——下一个该补的是 batched prefill

验证：`uv run python examples/qwen3_0_6b/check_tileops_components.py`（组件级
对拍）与 `uv run --with transformers python examples/qwen3_0_6b/compare_transformers.py --tileops`
（对 HF 端到端，logits/top1/greedy 与原版引擎同精度级别）。

调试笔记：接入时的"层 14 悬崖"是 residual stream 传递 bug（新 norms 不再原地
写 `self.hid`，循环内每一层都误把 embedding 当残差基；residual stream 随层数
增长，前 13 层残差占比小未暴露，第 14 层爆发）。修复为在循环内局部传递
residual。
