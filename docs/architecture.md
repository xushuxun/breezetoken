# breezetoken 架构

> 定位：单机、单序列、以 tilelang kernel 为核心的本地推理引擎。
> 当前版本：0.2.x。文本模型 Qwen3.5-0.8B，GGUF Q8_0/f32。

## 1. 分层

```
                 ┌──────────────────────────────┐
CLI / HTTP  ───► │ engine.py                    │  prompt / chat / EOS / 取消
                 │  └─ model + tokenizer        │
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

依赖单向：`model → backends → kernels`。`model` 不知道 tilelang，`kernels` 不知道
模型；`torch` 后端不 import tilelang，因此可以在 tilelang 不可用时仍作为参考实现。

没有单独保留 `ops/`、`backends/base.py` 或 `llvm_backend.py` 这类过渡层。后端接口
只有三个操作：

```python
backend.linear(x, w)        # (..., K) @ (K, N) -> (N,)
backend.rms_norm(x, w, eps) # 最后一维归一化
backend.silu_mul(gate, up)
```

## 2. Decode 流

`Engine.generate_stream(ids, max_new, ...)`：

1. 校验 prompt 非空、`len(ids) + max_new <= max_context`、采样参数；
2. `model.reset()`（GDN 递归/卷积状态清零）；
3. prefill：逐个 prompt token 调用 `forward_token(token, pos, logits=?)`，
   只有最后一个 prompt token 计算 248k 词表的 logits；
4. 循环：采样 → 检查 EOS → yield；需要继续时把新 token 以 `pos = len(ids)+step`
   喂回模型；
5. 每个 prompt token 之间检查 `cancel`，客户端断开时在下一个 token 边界停止。

`forward_token` 依次执行 24 层：

- RMSNorm → attention（6 层）或 GDN（18 层）→ 残差；
- RMSNorm → MLP（gate/up + SiLU + down）→ 残差；
- 最后 `norm_f` + `[H, V]` 的 lm_head。

投影权重在 CUDA 上是 bf16，norm 权重、KV cache、GDN state 和激活保持 f32。
attention 使用 `torch.nn.functional.scaled_dot_product_attention(enable_gqa=True)`，
失败时退回按 KV head 分组的 einsum；这会避免旧实现里 `repeat_interleave` 物化
`(S, H, D)` 的副本。

## 3. Kernel 覆盖

| 组件 | CPU | CUDA | 备注 |
|---|---|---|---|
| GEMV | ✅ `kernels/cpu/gemv.py` | ✅ `kernels/cuda/gemv.py` | 投影；bf16 CUDA split-K |
| RMSNorm | ✅ 每行一个 block | ✅ 每行一个 block | f32；覆盖 flat / per-head / GDN |
| SiLU× | ✅ | ✅ | f32，两输入 |
| RoPE | torch | torch | partial RoPE |
| attention | torch SDPA | torch SDPA | GQA；无 CUDA attention kernel |
| GDN 递归 | torch | torch | 下一步 kernel 化的重点 |
| lm_head | GEMV | GEMV | 未融合 argmax |

`TilelangBackend` 在 shape/dtype 没有 kernel 时回退 torch；例如 CUDA 上
`ssm_alpha/ssm_beta` 的 N=16 走 torch 小 matvec，GPU 上 K 不在 TILES 的投影也会
回退。CPU 的 C 后端是单线程，`auto` 因此在 CPU 选择 torch；`--backend tilelang`
用于验证 kernel。

## 4. 内存

- **权重**：每个 2-D 权重以 `(K, N)` 连续布局保存（GGUF 原布局是 `(N, K)`，加载
  时转置一次）。tied embedding 只保留 `[H, V]` 一份；token 查表读一列并 `float()`
  复制 4KB，避免旧实现 `(V,H)+(H,V)` 两份。
- **反量化**：GGUF Q8_0 按 64MiB 行块反量化，避免整表 f32 临时副本把峰值内存翻倍。
- **KV cache**：每层 `max_context × n_kv × head_dim` 的 f32；默认
  `max_context=8192`（约 200MB），不直接按 GGUF 的 262144 分配。
- **GDN state**：conv window + `[heads, state, state]` f32，每层约 1MB。
- **RoPE**：`max_context × rope_dim` 的 cos/sin 表。

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

1. **Kernel 化**：RoPE、attention、GDN 的 torch 路径逐项搬到 CPU/CUDA；
2. **可捕获执行**：固定 buffer、device `Pos`、全容量掩码 attention、采样进图，
   把旧引擎的 CUDA graph 能力接回新架构；
3. **Prefill**：batched prefill 与 KV/prefix 复用；
4. **量化**：直接 Q8_0 GEMV，降低权重带宽；
5. **平台**：在独立虚拟环境验证 LLVM CPU 后端（通过 `TVM_LIBRARY_PATH`，不改
   site-packages）；GPU 端按目标卡重测 TILES。
