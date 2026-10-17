# breezetoken

基于 [tilelang](https://github.com/tile-ai/tilelang) 的本地 Qwen3.5 推理引擎：GGUF 读取、
CPU/CUDA kernel、OpenAI 兼容 HTTP API 和一个零依赖的网页聊天前端。

当前目标是**单序列 decode 正确、可服务、可测试**；批处理 prefill、CUDA graph 与
更大模型是后续路线（见 [docs/architecture.md](docs/architecture.md)）。

## 特性

- **模型**：Qwen3.5-0.8B（24 层 = 18 GDN + 6 GQA，partial RoPE、输出门控、tied
  embedding），文本单 token decode。
- **权重**：GGUF v3 mmap 读取，Q8_0/f32，分块反量化以限制峰值内存；embedding 与
  lm_head 共用一份 `(H, V)` 权重。
- **tokenizer**：HuggingFace Rust `tokenizers` 从 GGUF 元数据重建；NFC、与
  tokenizer.json 一致的 Qwen 预分词正则、全部 special tokens。与 HF 逐 id 对拍。
- **chat**：直接渲染 GGUF 的 `tokenizer.chat_template`（Jinja2 沙箱），包含
  `<think>\n\n</think>\n\n` 空思考块、system / assistant / tool 消息。
- **后端**：`torch`（纯参考实现，完全不 import tilelang）与 `tilelang`（CPU
  `target="c"` / CUDA bf16）。`auto` 在 CUDA 选 tilelang，在 CPU 选更快的 torch。
- **Kernel**：CPU/CUDA 的 GEMV、RMSNorm、SiLU；RoPE、attention（PyTorch SDPA +
  GQA）和 GDN 递归暂时保持 torch，便于对拍。
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

# 默认按 chat template 渲染，CPU auto 选 torch
uv run breezetoken --prompt "用一句话解释 RoPE" --max-new-tokens 64

# 原始 completion（不加 chat template）
uv run breezetoken --prompt "The capital of France is" --raw --max-new-tokens 4

# CUDA + tilelang kernel
uv run breezetoken --device cuda --backend tilelang --prompt "你好"

# 强制 CPU tilelang kernel（用于验证 kernel / 对比性能）
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
| `auto` | CUDA | bf16 | 优先 tilelang |
| `auto` | CPU | f32 | 选择 torch（单线程 C kernel 慢于 torch 的向量化 matmul） |

本机 CPU（WSL2，无 CUDA）短 prompt、greedy 实测：

- `torch` CPU：约 **6–7 tok/s**（0.8B f32）；
- `tilelang` CPU：约 **2.3 tok/s**（单线程 C 后端，首次 JIT 约 10s）；
- 模型加载约 **5s**（OS 页缓存命中时）。

GPU 数字需在目标机器上重新测量；本项目不再沿用旧 Qwen3-0.6B 示例的 188 tok/s
数据。

## 测试

```bash
uv run pytest

# 需要本机有 Qwen3.5 GGUF 的集成项（默认会在文件缺失时 skip）
BREEZETOKEN_TEST_MODEL=/path/to/Qwen3.5-0.8B-Q8_0.gguf uv run pytest

# 额外编译并运行 CPU tilelang kernel 对比
BREEZETOKEN_TEST_TILELANG=1 uv run pytest tests/test_model.py
```

测试默认把 tilelang 缓存放到临时目录；如果本机无 GPU，CUDA kernel 不会被编译。
`BREEZETOKEN_TEST_HF` 可指定 HF `tokenizer.json` 所在目录，用于 tokenizer 对拍。

## 架构

```
breezetoken/
├── gguf.py            # GGUF v3 + 分块 Q8_0/f32 反量化
├── tokenizer.py       # tokenizers BPE + NFC + special tokens + chat template
├── backends.py        # Backend 协议 + TorchBackend + TilelangBackend + 工厂
├── model/qwen35.py    # Qwen3.5 hybrid decoder，单 token forward
├── engine.py          # model + tokenizer + EOS/上下文/生成策略
├── streaming.py       # GenerationJob、StopBuffer、增量 UTF-8
├── kernels/
│   ├── cpu/{gemv,norm}.py
│   └── cuda/{gemv,norm}.py
├── generate.py        # CLI
├── server.py          # FastAPI + OpenAI schema
└── web/chat.html      # vanilla 前端
```

依赖方向：`model → backends → kernels`；`engine → model/tokenizer`；
`server/generate → engine`。kernel 永不 import 模型或服务。

详细的 decode 流、内存与路线见 [docs/architecture.md](docs/architecture.md)。

## 已知限制

- 只做单序列 decode；没有 batched prefill（长 prompt 逐 token 处理）。
- RoPE、attention、GDN 递归仍在 torch；CUDA graph 尚未接入（动态 Python 控制流
  与当前 ops 契约需要先改成固定 buffer + device `Pos`）。
- 文本模型；不支持 Qwen3.5 的 vision/audio 输入，也不支持 MTP/nextn 草稿层。
- `max_context` 默认 8192（GGUF 声明 262144，运行时截断）；更大的上下文需要先
  解决 KV/attention 的显存与 kernel 规划。
- GGUF 仅支持 Q8_0/f32；CPU tilelang 仅支持单线程 C 后端，不包含 site-packages
  的 TVM 替换 hack。

### 路线

1. 把 RoPE、attention、GDN 的 torch 算子收进 CPU/CUDA kernel；
2. 采用固定 buffer + device `Pos` + 全容量掩码 attention，接 CUDA graph；
3. batched prefill 与 KV/prefix 复用；
4. 直接 Q8_0 GEMV 以降低权重带宽；
5. 在独立环境验证 LLVM CPU 后端（不修改 site-packages）。

## 许可

仓库目前未声明项目级许可证。移植自 TileFoundry 的 CUDA GEMV 以及
`examples/qwen3_0_6b/` 下的 TileOPs 代码遵循其 MIT 许可，见
[NOTICE](NOTICE) 与 `LICENSES/`。
