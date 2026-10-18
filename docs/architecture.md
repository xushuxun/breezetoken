# breezetoken 架构

> 定位：单机、单序列、以 tilelang kernel 为核心的本地推理引擎。
> 当前版本：0.2.x。原生路径支持 Qwen3.5-0.8B（Q8_0/f32）；Qwen3.6-35B-A3B
> （IQ3_S、256 experts）只有原生 plan/scaffold，尚未实现。

## 1. 分层

```
                 ┌──────────────────────────────┐
CLI / HTTP  ───► │ engine.py                    │  prompt / chat / EOS / 取消
                 │  └─ Engine (native)          │
                 ├──────────────────────────────┤
                 │ model/qwen35.py              │  单 token 前向 + 生成循环
                 ├──────────────────────────────┤
                 │ backends.py                  │  Backend 协议 + 选择/缓存/fallback
                 │  ├─ TorchBackend             │  纯 torch 参考
                 │  └─ TilelangBackend          │  按 shape/dtype 选 kernel
                 ├──────────────────────────────┤
                 │ kernels/{cpu,cuda}           │  tilelang 源码
                 └──────────────────────────────┘
```

依赖单向：`model → backends → kernels`；`planner` 只依赖硬件和 `Qwen35Config`。
原生模型本身拆成 `config/layers/prefill/graph/qwen35` 五块：`layers` 只做
attention/GDN/MLP，`prefill` 只做批量 prompt，`graph` 只做 CUDA graph，
`qwen35.py`（436 行）只保留加载、forward 和 generation。

`qwen35moe`（35B-A3B）暂不进入原生执行：`Engine.load` 会抛
`NotImplementedError`，指向 [qwen36_35b_plan.md](qwen36_35b_plan.md)。量化格式、
tensor descriptor、quant GEMV 和 MoE 目前只有 scaffold；不会调用 llama.cpp。
`model` 不知道 tilelang，`kernels` 不知道模型；`torch` 后端不 import tilelang，因此
可以在 tilelang 不可用时仍作为参考实现。

没有单独保留 `ops/`、`backends/base.py` 或 `llvm_backend.py` 这类过渡层。后端接口
现在覆盖 decode/prefill 用到的七个操作：

```python
backend.linear(x, w)                 # (..., K) @ (K, N) -> (N,)
backend.linear_batch(x, w)           # (S, K) @ (K, N), prefill
backend.rms_norm(x, w, eps)          # 最后一维归一化
backend.silu_mul(gate, up)
backend.fused_add_rms_norm(a, b, w, eps) -> (h, xn)
backend.rope(x, cos, sin, pos, rope_dim)
backend.gdn_decode(q, k, v, g, beta, state, scale, eps)
```

## 2. Decode 流

`Engine.generate_stream(ids, max_new, ...)`：

1. 校验 prompt 非空、`len(ids) + max_new <= max_context`、采样参数；
2. `model.reset()`（GDN 递归/卷积状态清零）；
3. prefill：prompt <8 时逐个调用 `forward_token(token, pos, logits=?)`；≥8 时走
   `_prefill_batch`，一次算完，只有最后一个 prompt token 计算 248k 词表 logits；
4. 循环：采样 → 检查 EOS → yield；需要继续时把新 token 以 `pos = len(ids)+step`
   喂回模型；
5. 每个 prompt token 之间检查 `cancel`，客户端断开时在下一个 token 边界停止。

### 自动放置

`Qwen35(backend="auto", device="auto")` 是默认路径：

1. `Hardware.detect()` 读 CUDA 显存/SM/shared memory、CPU 核数和可用内存；
2. `planner.plan()` 按层估算权重（GPU bf16、CPU f32）+ KV/state，保留 1GiB
   CUDA workspace；
3. 选最长的连续 GPU 前缀；放不下就整段后缀落到 `TorchBackend` CPU；
4. head 单独判断：VRAM 够就上 GPU，否则留在 CPU；每层边界需要的下一层
   `norm_in` 会复制到当前层设备（4KB），所以 fused kernel 不跨设备；
5. 默认上下文 8192；内存不足时自动降到 4096/2048/1024，显式 `max_context`
   则视为硬约束。

0.8B / 8GB 会全 GPU；`max_gpu_layers=1` 可用于强制验证混合路径。CPU-only
环境走同一个 API，planner 返回全 CPU。

### Batched prefill

prompt 长度 ≥8 时走 `_prefill_batch`：

- embedding 用 `index_select` 一次取出整个 prompt；
- 每层投影用 `linear_batch`：GPU bf16 matmul，CPU f32 matmul；
- attention 用 `scaled_dot_product_attention(is_causal=True, enable_gqa=True)`，
  K/V 一次写满 cache；
- GDN 的 conv 用 `conv1d` 一次算完，q/k/v/beta/g 也是批量投影；delta 递归仍按
  token 调 `gdn_decode`，所以 state 更新与 decode 完全一致；
- 每层残差/norm/MLP 使用 rows=S 的 kernel 或 batched matmul。

128 token prompt 在本机约 250ms；短 prompt（<8）仍逐 token，以保证首 token
延迟和取消粒度。

### 前向

`forward_token` 依次执行 24 层：

- 当前层设备上 RMSNorm → attention（6 层）或 GDN（18 层）→ 残差；
- 当前层设备上 RMSNorm → MLP（gate/up + SiLU + down）→ 残差；
- 最后 `norm_f` + `[H, V]` 的 lm_head。

`h`/`xn` 只在放置边界 `.to(device)` 一次（连续前缀意味着每 token 至多一次）。
投影权重在 GPU 上是 bf16、CPU 上是 f32；norm 权重、KV cache、GDN state 和激活
保持 f32。CUDA 上 attention 走 `kernels/cuda/attention.py` 的 context-split + combine
kernel（f32/TF32、4 个 valid query row、16 splits），CPU 层和未覆盖的形状回退
`scaled_dot_product_attention(enable_gqa=True)` / 分组 einsum。旧实现里
`repeat_interleave` 物化 `(S, H, D)` 副本的问题不再存在。

## 3. Kernel 覆盖

| 组件 | CPU | CUDA | 备注 |
|---|---|---|---|
| GEMV | ✅ `kernels/cpu/gemv.py` | ✅ `kernels/cuda/gemv.py` | 投影；bf16 CUDA split-K |
| RMSNorm | ✅ 每行一个 block | ✅ 每行一个 block | f32；覆盖 flat / per-head / GDN |
| SiLU× | ✅ | ✅ | f32，两输入 |
| RoPE | torch | ✅ `kernels/cuda/rope.py` | partial RoPE；pos 为 device int32 |
| attention | torch SDPA | ✅ `kernels/cuda/attention.py` | GQA split+combine；覆盖 0.8B 形状，其他回退 SDPA |
| GDN 递归 | torch | ✅ `kernels/cuda/gdn.py` | TileOPs delta-rule 移植；原地 f32 state |
| fused add+RMSNorm | ✅ | ✅ | 残差与下一层 norm 合并为一次读 |
| lm_head | GEMV | GEMV | greedy 时 argmax 进 CUDA graph |

权重在加载期做两项融合：attention 的 `[q|k|v]` 是一次 GEMV，MLP 的
`[gate|up]` 也是一次 GEMV；它们只改变启动次数，不改变每个输出列的算术。

`TilelangBackend` 在 shape/dtype 没有 kernel 时回退 torch；例如 CUDA 上
`ssm_alpha/ssm_beta` 的 N=16 走 torch 小 matvec，GPU 上 K 不在 TILES 的投影也会
回退。自动放置的 CPU 层直接使用 `TorchBackend`；`--backend tilelang --device cpu`
仍保留用于单独验证 C kernel。

CUDA 路径已在 RTX 4060 Laptop（8GB，CUDA 13.0）上验证：kernel 单元测试、完整
模型 greedy 16/16 对齐 HF f32、HTTP 服务流式/非流式均通过。128 token greedy
eager 约 54 tok/s decode（TTFT 约 87ms）；warmup 捕获并复用 CUDA graph 后
约 126 tok/s（128 token，TTFT 约 96ms），峰值显存约 1.66GB；batched prefill
128 token 约 250ms（逐 token eager 约 2s+）；首次 JIT 约 24s（18 个基础 kernel
形状，prefill 还会按 prompt 长度编译 rows=S 的 norm/resid/silu）。

### CUDA graph

全 GPU + greedy 时，`generate_stream` 在第一个 token 之后走图：

- 固定容量 attention kernel（`gqa_decode_fixed`）读取完整 K/V cache + device
  `pos`，超出当前 `pos` 的 tile 被 mask；
- K/V 写入用 `index_copy_` + device index；RoPE 直接读 `pos` buffer；
- 整步 24 层、lm_head、argmax 和 token 写回都在图内；
- 图只在第一次 decode 捕获，之后跨请求复用；`max_context` 固定；
- 混合放置、非 greedy 或 attention 形状不匹配时回退 eager。

图路径的 token 读取需要在 host 侧 `.item()` 同步；这也是当前 126 tok/s 与
旧 188 tok/s 的主要差距（旧实现把更多采样/控制流也放进了图）。

## 3.5 35B-A3B 原生计划

Qwen3.6-35B-A3B 是 `qwen35moe`、IQ3_S、15GB、40 层、256 experts/top-8。原生
执行不能把权重 materialize 成 f32/bf16；需要：

- `quant/` 的 GGML 格式表、QuantizedTensor 行描述和参考 dequant；
- CPU/CUDA `quant_gemv`，把 dequant 融进 GEMV；
- GDN 16K/32V、attention 16Q/2KV + output gate；
- router/top-8/shared expert、CPU 专家 + GPU 非专家的分量级放置；
- 先做 dequant 单测，再做单层 MoE，再做全模型。

完整阶段、内存预算、风险和验收见
[docs/qwen36_35b_plan.md](qwen36_35b_plan.md)。当前 `quant/`、`kernels/*/quant_gemv.py`、
`model/qwen36_moe.py` 是接口和格式表，不包含可运行 kernel。

## 4. 内存

- **权重**：每个 2-D 权重以 `(K, N)` 连续布局保存（GGUF 原布局是 `(N, K)`，加载
  时转置一次）。tied embedding 只保留 `[H, V]` 一份；token 查表读一列并 `float()`
  复制 4KB，避免旧实现 `(V,H)+(H,V)` 两份。
- **反量化**：GGUF Q8_0 按 64MiB 行块反量化，避免整表 f32 临时副本把峰值内存翻倍。
- **KV cache**：每个 attention 层在自己的放置设备上分配
  `max_context × n_kv × head_dim` 的 f32；默认 `max_context=8192`（全部约
  200MB），不直接按 GGUF 的 262144 分配。
- **GDN state**：conv window + `[heads, state, state]` f32，每层约 1MB，跟随层设备。
- **RoPE**：每个使用到的设备各一份 `max_context × rope_dim` cos/sin 表和 device
  int32 position buffer。
- **放置预算**：GPU 层权重 bf16 + KV/state + head；CPU 层权重 f32 + KV/state。

## 5. 并发与取消

- `Engine` 只支持一个序列；server 用 `ThreadPoolExecutor(max_workers=1)` 串行执行。
- 每个请求有自己的 `threading.Event`；`GenerationJob` 把阻塞生成器桥接为 asyncio
  流，`aclose()` 设置取消并等待 worker。
- 客户端断开/命中 stop 时，HTTP 层只设置取消；worker 在下一个 token 边界退出。
  因为 executor 只有一个 worker，下一个请求一定在旧请求退出后开始，不会出现旧
  worker 继续改 KV/state 的竞态。
- 流式文本用增量 UTF-8 解码；`StopBuffer` 保留 `max(stop_len)-1` 个字符以检测跨
  token 边界的 stop。

## 6. Server 契约

- `POST /v1/chat/completions`：消息 → GGUF chat template → tokenizer；
- `POST /v1/completions`：原始 prompt；
- 非流式返回完整文本与真实 usage；流式为每个 token 发 delta，最后发
  `finish_reason`，再发 `[DONE]`；
- `prompt_tokens + max_tokens > max_context`、空 prompt、非法参数返回 400，错误体
  使用 OpenAI 形状 `{"error":{"message":...,"type":...}}`；
- 模型加载与 warmup 在 FastAPI lifespan 中完成，warmup 编译所有 kernel 形状。

## 7. 正确性约定（Qwen3.5）

- GGUF norm gamma 已预折叠为 `1 + w`，实现直接用 `x * gamma`；
- `ssm_a` 存的是 `-exp(A_log)`，所以 log-decay 是 `a * softplus(...)`；
- GGUF `block_count` 包含 MTP block，模型层数要减 `nextn_predict_layers`；
- partial RoPE 只旋转前 `rope_dim` 维；
- attention 的 `q_proj` 输出是每头 `[q | output-gate]`；
- text-only 输入下 mRoPE 退化为标准 RoPE（三个 position 网格相同）。

## 8. 与旧实现的差异

旧版 `examples/qwen3_0_6b/` 的四个引擎证明了 188 tok/s（0.6B、CUDA graph、全部
kernel 融合），但它是四套手写执行序列，且 `llvm_backend.py` 通过改 site-packages
的符号链接切换 TVM，曾在源码树被删除后使整个环境无法 import tilelang。

本版明确移除该 hack，改为：

- 后端接口显式化，`torch` 是可独立运行的参考实现；
- 规范化 tokenizer/chat template，修掉 special token、NFC、逐 token 解码问题；
- server 的取消/锁/usage/finish_reason 按实际状态返回；
- 不再保留不可达的 kernel（旧 CPU attention/rope 是 Qwen3 dense 布局，不能直接
  用于 qwen35，已删除）。

## 9. 路线

1. **KV/prefix 复用**：跨请求共享 prompt 前缀；
2. **原生 MoE/IQ3_S**：按 [plan](qwen36_35b_plan.md) 做 dequant → quant GEMV
   → 单层 GDN/attention → MoE → 全模型；
3. **采样图**：把 temperature/top-p 采样也放进图，减少 host 同步；
4. **平台**：在独立环境验证 LLVM CPU 后端；CUDA tile 参数仍以 RTX 4060 的
   99KB shared memory 为基准。
