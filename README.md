# breezetoken

基于 [tilelang](https://github.com/tile-ai/tilelang) 的本地 Qwen3.5 推理引擎：GGUF 读取、
CPU/CUDA kernel、OpenAI 兼容 HTTP API 和一个零依赖的网页聊天前端。

默认不要求用户区分 CPU/GPU：加载时根据显存、系统内存和模型形状自动把最长的
GPU 前缀与 CPU 后缀组合起来。0.8B 在 8GB 卡上直接全 GPU；更大的模型会退化为
混合推理而不是 OOM。全 GPU + greedy 时 decode 进一步走 CUDA graph；长 prompt
走按层批量 prefill（见 [docs/architecture.md](docs/architecture.md)）。

## 特性

- **模型**：Qwen3.5-0.8B（24 层 = 18 GDN + 6 GQA）原生 kernel。Qwen3.6-35B-A3B
  （40 层 = 30 GDN + 10 GQA，256 experts/top-8，IQ3_S）已实现原生混合执行：
  非专家权重 dequant 成 bf16 上 GPU，专家张量保持 mmap 量化字节、按 token
  把 top-8 专家 stage 到计算设备 dequant + matmul（greedy 与 llama.cpp
  逐 token 一致，decode 约 2 tok/s；细节见 [docs/qwen36_35b_plan.md](docs/qwen36_35b_plan.md)）。
- **权重**：GGUF v3 mmap 读取，Q8_0/f32，分块反量化以限制峰值内存；embedding 与
  lm_head 共用一份 `(H, V)` 权重。
- **tokenizer**：HuggingFace Rust `tokenizers` 从 GGUF 元数据重建；NFC、与
  tokenizer.json 一致的 Qwen 预分词正则、全部 special tokens。与 HF 逐 id 对拍。
- **chat**：直接渲染 GGUF 的 `tokenizer.chat_template`（Jinja2 沙箱），包含
  `<think>\n\n</think>\n\n` 空思考块、system / assistant / tool 消息。
- **自动放置**：`planner.py` 读 `torch.cuda.mem_get_info`/SM/shared memory、
  CPU 核数与可用内存，按层估算权重 + KV/GDN state；默认 GPU 前缀用
  `tilelang` bf16，剩下的 CPU 层用 `torch` f32。全模型放得下时不会强行拆到 CPU。
- **后端**：原生 `torch`/`tilelang`。`--backend/--device` 保留为调试覆盖，
  正常使用不需要传。
- **35B 量化 plan**：IQ3_S/IQ2_S/IQ4_XS + 256-expert MoE 的格式、内存模型、
  kernel 分层和验收标准见 [docs/qwen36_35b_plan.md](docs/qwen36_35b_plan.md)；
  `quant/`、`kernels/*/quant_gemv.py`、`model/qwen36_moe.py` 目前只有接口。
- **Kernel**：CPU/CUDA 的 GEMV、RMSNorm、SiLU、fused add+RMSNorm，以及 CUDA 的
  RoPE、GDN decode 和 GQA decode attention（GDN/attention 均移植自 TileOPs，
  已在 RTX 4060 上验证）。
- **执行图**：全 GPU、greedy 时把一整步 decode 捕获进 CUDA graph（固定容量
  attention + device position + 图内 argmax），复用后约 **126 tok/s**；混合放置、
  采样/top-p 或未覆盖形状自动回退 eager。
- **Batched prefill**：prompt ≥8 token 时按层批量投影/attention；GDN 的 delta
  递归仍按 token 顺序推进。128 token prompt 约 250ms，避免逐 token 重读全部权重。
- **流式**：增量 UTF-8 解码（不会在 token 边界产生 `�`）、stop 序列、EOS、
  断开取消；单 worker executor 串行化模型访问。
- **服务**：`/v1/chat/completions` 与 `/v1/completions`，流式/非流式、真实
  `usage`/`finish_reason`、OpenAI 形状错误、上下文与输出长度校验。
- **测试**：tokenizer/chat/GGUF/streaming/backend/model/server，默认不需要 GPU。

## 安装

需要 Python >= 3.12（仓库的 `.python-version` 是 3.14）。

```bash
uv sync --extra dev
```

如果 `$HOME` 不可写（CI/沙箱），给 tilelang 指定一个可写缓存目录：

```bash
export TILELANG_CACHE_DIR=$PWD/.tilelang-cache
```

## 快速开始

```bash
export BREEZETOKEN_MODEL=/path/to/Qwen3.5-0.8B-Q8_0.gguf

# 默认按 chat template 渲染；auto 自动选择全 GPU 或混合放置
uv run breezetoken --prompt "用一句话解释 RoPE" --max-new-tokens 64

# 原始 completion（不加 chat template）
uv run breezetoken --prompt "The capital of France is" --raw --max-new-tokens 4

# 35B-A3B IQ3_S 原生混合执行（非专家 bf16 上 GPU，专家量化字节 mmap 按 token stage）
uv run breezetoken \
  --model ~/.cache/modelscope/models/unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/master/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --prompt "你好" --max-new-tokens 32

# 只看自动放置结果（正常使用不需要传 backend/device）
uv run breezetoken --prompt "你好"

# 调试/对拍时强制单一后端
uv run breezetoken --device cuda --backend torch --prompt "你好"
uv run breezetoken --device cpu --backend tilelang --prompt "你好"
```

CLI 会把生成的文本写到 stdout，把模型、后端、加载时间、首 token 延迟和
tok/s 写到 stderr。

## OpenAI 兼容服务

```bash
uv run breezetoken-server --model "$BREEZETOKEN_MODEL" --port 8321
# 打开 http://127.0.0.1:8321/
```

```bash
curl -X POST http://127.0.0.1:8321/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":64}'
```

端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网页聊天 |
| GET | `/healthz` | 模型/后端/上下文摘要 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | chat，`stream` 可选 |
| POST | `/v1/completions` | 原始 completion，`stream` 可选 |

支持的参数：`max_tokens`、`temperature`、`top_p`、`seed`、`stop`、`stream`、
`stream_options.include_usage`；chat 额外支持 `tools` 与 `enable_thinking`。
多模态内容会被明确拒绝（当前引擎只支持文本）。

服务端行为：

- 模型加载与 warmup 在 FastAPI lifespan 中完成；warmup 会触发 kernel 编译；
- 请求在单 worker executor 中串行执行；客户端断开时设置取消标志，worker 在
  下一个 token 边界停止，不会与下一个请求重叠；
- `prompt_tokens + max_tokens` 超出 `max_context` 返回 400；
- `finish_reason` 根据实际停止原因返回 `stop`（EOS/stop 序列）或 `length`
  （`max_tokens`），`usage` 是实际生成 token 数；
- 消息正文中出现控制 token 字符串（如 `<|im_start|>`）会返回 400，避免 chat
  模板注入。

## 后端与性能

```python
from breezetoken.engine import Engine

engine = Engine.load("model.gguf", backend="auto", device="auto")
for token in engine.generate_stream(engine.encode("Hello"), 16):
    ...
```

| backend | device | 权重 dtype | 说明 |
|---|---|---|---|
| `torch` | CPU/CUDA | 默认 f32 | 纯 PyTorch 参考实现，用于正确性对拍 |
| `tilelang` | CPU | f32 | `target="c"`，GEMV/RMSNorm/SiLU kernel，单线程 |
| `tilelang` | CUDA | bf16 | bf16 GEMV + f32 norm/silu kernel |
| `auto` | 混合 | GPU bf16 / CPU f32 | **默认**；planner 自动选 GPU 前缀 + CPU 后缀 |

本机实测（WSL2，RTX 4060 Laptop 8GB，torch 2.14.1+cu130，短 prompt、greedy）：

| 配置 | 结果 |
|---|---|
| `torch` CPU | 约 **6–7 tok/s**（0.8B f32） |
| `tilelang` CPU | 约 **2.3 tok/s**（单线程 C 后端，首次 JIT 约 10s） |
| `auto` eager（全 GPU，greedy） | 约 **54 tok/s** decode（128 token；TTFT 约 87ms） |
| `auto` CUDA graph（同配置） | 约 **126 tok/s** decode（warmup 后复用；TTFT 约 96ms） |
| batched prefill（128 token） | 约 **250ms**（逐 token eager 约 2s+） |
| CUDA 显存（0.8B） | 峰值约 **1.66 GB** |
| 混合放置 | `max_gpu_layers=1` 的输出与全 GPU 一致，CPU 后缀显著变慢但不会 OOM |
| CUDA 正确性 | 16 token greedy 与 HF f32 **16/16** 一致 |
| 模型加载 | 约 **4–6s**（OS 页缓存命中） |

CUDA 首次运行需要 JIT 编译 18 个 kernel 形状，约 24s；之后走
`TILELANG_CACHE_DIR` 缓存。旧 Qwen3-0.6B 示例的 188 tok/s 是 0.6B + 全 kernel
融合 + CUDA graph 的成绩；当前 0.8B 为 126 tok/s，差距主要是每 token host 侧
同步（greedy token 读取）与尚未放进图的采样/控制流。

## 测试

```bash
uv run pytest

# 需要本机有 Qwen3.5 GGUF 的集成项（默认会在文件缺失时 skip）
BREEZETOKEN_TEST_MODEL=/path/to/Qwen3.5-0.8B-Q8_0.gguf uv run pytest

# 额外编译并运行 CPU tilelang kernel 对比
BREEZETOKEN_TEST_TILELANG=1 uv run pytest tests/test_model.py

# CUDA kernel 每次有 GPU 都会跑；完整 CUDA 模型需要显式开启
BREEZETOKEN_TEST_CUDA=1 uv run pytest tests/test_cuda.py

# 35B IQ3_S 原生路径：dequant 单测、quant GEMV、MoE router、模型级 greedy 对拍
uv run pytest tests/test_quant_dequant.py tests/test_quant_gemv.py tests/test_moe.py tests/test_qwen36_moe.py
```

测试默认把 tilelang 缓存放到临时目录；如果本机无 GPU，CUDA kernel 不会被编译。
`BREEZETOKEN_TEST_HF` 可指定 HF `tokenizer.json` 所在目录，用于 tokenizer 对拍。

## 架构

```
breezetoken/
├── gguf.py            # GGUF v3 + 分块 Q8_0/f32 反量化
├── tokenizer.py       # tokenizers BPE + NFC + special tokens + chat template
├── planner.py         # 硬件/模型画像 + GPU 前缀/CPU 后缀放置
├── backends.py        # Backend 协议 + TorchBackend + TilelangBackend + 工厂
├── quant/             # IQ/K-quant 格式表、IQ 网格表、QuantizedTensor、向量化 dequant
├── model/
│   ├── config.py      # Qwen35Config + KV/GDN state
│   ├── layers.py      # attention / GDN / dense MLP
│   ├── prefill.py     # batched prefill
│   ├── graph.py       # CUDA graph
│   ├── qwen35.py      # native facade/loading/generation
│   ├── moe.py         # 35B softmax router + ExpertPool 量化专家 stage/执行
│   └── qwen36_moe.py  # 35B 混合放置 decoder（GDN 16K/32V、attn 16Q/2KV + gate）
├── engine.py          # model + tokenizer + EOS/上下文/生成策略
├── streaming.py       # GenerationJob、StopBuffer、增量 UTF-8
├── kernels/
│   ├── cpu/{gemv,norm,quant_gemv}.py
│   └── cuda/{gemv,norm,gdn,rope,attention,quant_gemv,moe}.py
├── generate.py        # CLI
├── server.py          # FastAPI + OpenAI schema
└── web/chat.html      # vanilla 前端
```

依赖方向：`model → backends → kernels`；`engine → model/tokenizer`；
`server/generate → engine`。kernel 永不 import 模型或服务。

详细的 decode 流、内存与路线见 [docs/architecture.md](docs/architecture.md)。

## 已知限制

- 只做单序列 decode；prompt ≥8 才走 batched prefill，短 prompt 仍逐 token。
- 自动放置以连续 GPU 前缀为单位，边界只跨一次设备；不会为了多塞一层而做非连续
  放置，也不做逐 kernel 的 GPU/CPU 切分。
- CUDA 侧的 GEMV/RMSNorm/SiLU/fused-norm、RoPE、GDN decode 和 attention 都是
  tilelang kernel；attention kernel 针对 0.8B 的 `H=8/Hkv=2/D=256`，其他形状回退
  SDPA。
- CUDA graph 只在“全 GPU + greedy”时启用；混合放置、`temperature > 0` 或
  `top_p != 1` 回退 eager。图使用固定的 `max_context` 容量，长上下文会多算
  被 mask 的 K/V tile。
- batched prefill 的 attention 是因果 SDPA，GDN 递归逐 token 调 decode kernel；
  它不会把整个 prompt 的 GDN 扫描并行化，取消也只在 prefill 结束后响应。
- 文本模型；不支持 Qwen3.5/3.6 的 vision/audio 输入，也不使用 MTP/nextn 草稿层。
- 35B-A3B 原生 MoE/IQ3_S 尚未实现；当前只有 plan、格式表、descriptor 和
  kernel/model 接口，`Engine.load` 会抛 `NotImplementedError`。
- 实现前需要先通过 phase 1/2 的 IQ dequant 与 quantized GEMV 验收，不能直接
  把 15GB 量化权重 materialize 成 f32/bf16。
- `max_context` 默认 8192（GGUF 声明 262144，运行时截断）；更大的上下文需要先
  解决 KV/attention 的显存与 kernel 规划。
- GGUF 仅支持 Q8_0/f32；CPU tilelang 仅支持单线程 C 后端，不包含 site-packages
  的 TVM 替换 hack。

### 路线

1. 35B 原生 IQ3_S/MoE：按 [plan](docs/qwen36_35b_plan.md) 做 dequant → quant GEMV
   → 单层 GDN/attention → MoE → 全模型；
2. KV/prefix 复用（跨请求共享 prompt 前缀）；
3. 直接 Q8_0 GEMV，降低权重带宽；
4. 在独立环境验证 LLVM CPU 后端（不修改 site-packages）。

## 许可

仓库目前未声明项目级许可证。移植自 TileFoundry 的 CUDA GEMV、TileOPs 的 GDN
decode/GQA attention，以及 `examples/qwen3_0_6b/` 下的 TileOPs 代码遵循其 MIT
许可，见 [NOTICE](NOTICE) 与 `LICENSES/`。
