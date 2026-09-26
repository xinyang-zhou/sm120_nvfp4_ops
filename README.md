# SM120 NVFP4 Operator Stack

面向 NVIDIA SM120（GeForce RTX 50 系列）的 NVFP4 算子库，覆盖自定义矩阵乘、DS-V4 CSA sparse MLA decode/prefill 和端到端 Mixture-of-Experts：

```text
Custom CuTe NVFP4 GEMM
    ├── QK^T -> FP32 softmax/quant -> P@V -> Attention
    └── persistent CuTe Grouped GEMM
          -> routing + gate/up + SiLU/quant + down + reduce
          -> Fused MoE
```

单 GEMM 与 Grouped GEMM 都由仓库自有的 `__global__` kernel 实现，直接使用 CuTe tensor/partition/copy/gemm abstraction 组织 TMA、shared-memory pipeline 和 SM120 block-scaled MMA。CUTLASS `GemmUniversalAdapter` 版本仅作为 reference 保留，不是默认执行路径。

GEMM 使用 packed E2M1、UE4M3 scale、FP32 累加和 FP16 输出。新 sparse MLA decode/prefill 使用 BF16 Q/O，448 维 NVFP4 与 64 维 BF16 混合计算。

> 项目处于 research preview 阶段。既有 GEMM 等路径有 RTX 5090 实测；新 DS-V4 sparse MLA decode/prefill 仅完成本地源码开发，构建、正确性和性能待服务器验证，见 [Sparse MLA](docs/SPARSE_MLA.md)。

## Highlights

- 仓库自有 Custom CuTe 单 GEMM，不调用 CUTLASS `GemmUniversal` 或 `GemmUniversalAdapter`；
- 默认单 GEMM 按 M 自动分发：M=128 使用 Split-K=4、M=256 使用 Split-K=2，其他 M 使用通用 CuTe；
- 显式实现 384-thread producer/consumer warp specialization、三阶段 TMA pipeline 和 persistent CTA tile scheduling；
- FP16 epilogue 对不少于 64 行的问题使用 shared-memory staging 与 TMA store，小 M 保留低开销 predicated store；
- 使用 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` 原生 block-scaled Tensor Core 指令；
- 新 C++ CuTe sparse MLA decode 支持 DS-V4 的 64-head shared-KV、SWA＋Top-K 双缓存、混合精度 QK/PV、FP32 分母、sink 和 split 合并；原 dense prefill/decode 与 paged decode 已移除；
- sparse prefill 接收每个 query 独立的合法索引，一个 CTA 完整处理一行 Q，复用 decode 数学与精度规则，无 split workspace；
- 保留 CUTLASS Collective reference，并提供 Custom CuTe、CUTLASS、cuBLASLt 三方性能与正确性对照；
- Grouped GEMM 在一次 persistent launch 中调度多个动态 M expert；
- Fused MoE 包含路由重排、两次 Grouped GEMM、SiLU、动态 NVFP4 量化和 top-k reduce；
- 提供 DeepEP-compatible Expert-major 接口，直接复用 Dispatch 分组布局与调用方 Workspace；
- C++、PyTorch custom op 与 Python API 统一使用 `sm120_nvfp4` 命名空间；
- 提供 cuBLASLt 原生 grouped API 能力探测，避免把 GEMM loop 误标为 grouped API；
- 构建、测试、benchmark 和文档独立组织，便于后续接入 SGLang。

## Repository layout

```text
sm120_nvfp4_ops/
├── CMakeLists.txt
├── cmake/                     # CUDA/CUTLASS/Torch 依赖发现
├── include/sm120_nvfp4/       # C++ 公开接口
├── src/
│   ├── common/                # 单/Grouped GEMM 共用的 CuTe 配置
│   ├── gemm/
│   │   ├── cute_gemm.cu       # 通用 Custom CuTe kernel
│   │   ├── gemm_dispatch.cu    # 默认 M-shape 路径选择
│   │   ├── specialized/        # M=128 / M=256 固定 Split-K 路径
│   │   └── cutlass_reference.cu
│   ├── grouped_gemm/          # persistent expert GEMM
│   ├── attention/             # DS-V4 CSA sparse MLA decode/prefill
│   └── fused_moe/             # gather/activation/reduce 与 orchestration
├── bindings/                  # PyTorch custom-op registration
├── python/sm120_nvfp4/        # Python 友好接口
├── tests/                     # C++ 与 Python 正确性测试
├── benchmarks/                # GEMM、attention、MoE 性能与能力探测
├── scripts/                   # build/test/benchmark 一键脚本
└── docs/                      # API、架构、性能与 roadmap
```

本目录是项目唯一源码树；早期原型目录不属于当前仓库，也不参与构建、测试或性能数据生成。

实现边界：`src/gemm/cute_gemm.cu`、`src/gemm/specialized/` 和
`src/grouped_gemm/` 中的执行 kernel 与调度逻辑由本仓库实现；CuTe/CUTLASS
提供 MMA atom、Tensor、TMA layout/copy 等底层 primitive；
`src/gemm/cutlass_reference.cu` 只用于 reference，cuBLASLt 只用于 benchmark
baseline，不属于默认执行路径。

## Requirements

- Linux x86-64；
- NVIDIA SM120 GPU，已验证 RTX 5090；
- CUDA Toolkit 12.8+，已验证 CUDA 13.2；
- CUTLASS 4.2+，用于 CuTe headers、SM120 MMA atom、TMA layout 和 reference kernel；
- CMake 3.24+、C++17；
- 可选：PyTorch 2.7+，用于 Python extension。

```bash
export CUDA_ROOT=/path/to/cuda
export CUTLASS_ROOT=/path/to/cutlass
```

## Build and test

```bash
./scripts/build.sh
./scripts/test.sh
```

脚本会从当前 Python 环境自动发现 PyTorch；如果未设置 `CUTLASS_ROOT`，会尝试使用 FlashInfer 安装目录携带的 CUTLASS。

也可以直接使用 CMake：

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUTLASS_ROOT="$CUTLASS_ROOT" \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build -j4
```

生成可追溯的 GEMM/Grouped GEMM JSON 与 CSV：

```bash
mkdir -p benchmarks/results
./scripts/benchmark.sh 16 4096 8192 \
  --warmup 50 --iterations 500 --heuristics 16 \
  --json benchmarks/results/gemm_rtx5090_YYYY-MM-DD.json \
  --csv benchmarks/results/gemm_rtx5090_YYYY-MM-DD.csv

./scripts/benchmark_grouped.sh 4 16 4096 8192 \
  --warmup 50 --iterations 500 --heuristics 16 \
  --json benchmarks/results/grouped_gemm_rtx5090_YYYY-MM-DD.json \
  --csv benchmarks/results/grouped_gemm_rtx5090_YYYY-MM-DD.csv
```

JSON 是一次运行的完整记录；CSV 在文件已存在时追加数据行，适合 shape
sweep。字段包含设备/CUDA/cuBLASLt 版本、计时方法、输入分布、heuristic、
workspace、correctness 和原始命令。现有性能声明与原始产物状态见
[结果索引](benchmarks/results/README.md)。

不构建 PyTorch extension：

```bash
cmake -S . -B build \
  -DCUTLASS_ROOT="$CUTLASS_ROOT" \
  -DSM120_NVFP4_BUILD_TORCH=OFF
```

测试覆盖：

- Custom CuTe 单 GEMM CPU reference；
- sparse MLA 的混合精度参考、稀疏索引、双缓存、sink、split、buffer 复用及 CUDA Graph（待服务器运行）；
- sparse prefill 的逐 query 候选、ragged query 行、分块调用一致性、FlashInfer streaming prefill 对照（待服务器运行）；
- Custom CuTe、默认 `gemm` 与 CUTLASS reference 一致性；
- Grouped GEMM 不均匀 expert row count；
- Fused MoE 本地/远端路由和非均匀 activation scale；
- Expert-major 直连路径与完整路由路径的逐元素等价性；
- 输出 buffer 复用以及 dtype/shape 约束。

## Python API

```bash
export PYTHONPATH="$PWD/build/python:$PYTHONPATH"
```

```python
import sm120_nvfp4

# 默认路径在 M=128/M=256 使用专用 kernel，其余 M 回退通用 CuTe。
y = sm120_nvfp4.gemm(a, weight, a_scale, weight_scale)
# 显式通用基线，不经过 M-shape dispatcher。
y_cute = sm120_nvfp4.cute_gemm(a, weight, a_scale, weight_scale)

# 仅用于对照。
y_reference = sm120_nvfp4.cutlass_gemm(
    a, weight, a_scale, weight_scale
)

# DS-V4 CSA sparse MLA：BF16 Q，FlashInfer NVFP4 packed shared-KV。
decode_output, lse2 = sm120_nvfp4.sparse_mla_decode(
    query_bf16,              # [B,1,64,512]
    swa_cache,               # uint8 [P_swa,64,384]，页内 footer-scale 格式
    compressed_cache,        # uint8 [P_cmp,64,384]
    swa_indices,             # int32 [B,128]，物理 slot IDs
    compressed_indices,      # int32 [B,512]
    sink=attention_sink,     # 可选 FP32 [64]
)

# Prefill：T 是本轮所有请求的 query 总数，每行独立提供候选。
prefill_output, prefill_lse2 = sm120_nvfp4.sparse_mla_prefill(
    prefill_query_bf16,      # [T,64,512]，已完成 RoPE
    swa_cache, compressed_cache,
    prefill_swa_indices,     # int32 [T,Kswa]
    prefill_comp_indices,    # int32 [T,Kcompressed]
    swa_lengths=prefill_swa_lengths,             # 可选 int32 [T]
    compressed_lengths=prefill_comp_lengths,     # 可选 int32 [T]
    sink=attention_sink,
)

y_grouped = sm120_nvfp4.grouped_gemm(
    grouped_x,
    expert_weight,
    seqlens,
    cu_seqlens,
    grouped_x_scale,
    expert_weight_scale,
)

y_moe = sm120_nvfp4.fused_moe(
    x,
    x_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    topk_ids,
    topk_weights,
)

# DeepEP-style Expand Dispatch 已经按 Expert 排列 recv_x 时，
# 直接量化并进入两次 Grouped GEMM，不重复本地路由。
grouped_nvfp4, grouped_scale = sm120_nvfp4.quantize_expert(
    recv_x,
    seqlens,
    cu_seqlens,
    scale_m_pad=scale_m_pad,
)
expert_output = sm120_nvfp4.expert_moe(
    grouped_nvfp4,
    grouped_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    seqlens,
    cu_seqlens,
    workspace=reusable_workspace,
)
```

packed E2M1 tensor 可使用 `torch.uint8` 或 `torch.float4_e2m1fn_x2`。scale tensor 是包含 raw UE4M3 编码的 `torch.uint8`，使用 `Sm1xxBlockScaledConfig<16>` 物理布局。详细约束见 [API](docs/API.md)。

## Performance snapshot

平台：RTX 5090、CUDA 13.2、cuBLASLt 13.4、CUTLASS 4.2.1。以下是无
profiler 的单 GEMM CUDA Event 结果；cuBLASLt 数字是给定搜索范围内最快的
成功 heuristic，不代表所有 cuBLASLt 配置。所有路径使用相同随机 packed
E2M1 数据与 UE4M3 scale，并逐元素验证。

| M,N,K | Implementation | Latency | Throughput | vs selected cuBLASLt heuristic |
|---|---|---:|---:|---:|
| 16,4096,8192 | Custom CuTe | 25.2 us | 42.56 TFLOP/s | 104.34% |
| 16,4096,8192 | CUTLASS reference | 26.2 us | 41.01 TFLOP/s | 100.49% |
| 16,4096,8192 | cuBLASLt id 70 | 26.3 us | 40.81 TFLOP/s | 100% |
| 512,4096,8192 | Custom CuTe | 28.4 us | 1208.39 TFLOP/s | 105.50% |
| 512,4096,8192 | CUTLASS reference | 28.2 us | 1216.83 TFLOP/s | 106.24% |
| 512,4096,8192 | cuBLASLt id 70 | 30.0 us | 1145.40 TFLOP/s | 100% |

`M=16` 保留 scalar epilogue，以避免完整 TMA tile 的固定开销；`M=512`
的 TMA-store epilogue 将 Custom CuTe 从修改前的 35.0 us 降至 28.4 us。
完整测量方法、证据状态和历史 CUTLASS sweep 见
[Performance](docs/PERFORMANCE.md)。在新的结构化结果重新采集并入库前，
这些历史手工记录数字不应直接作为简历中的可追溯结果。

默认 dispatcher 在相同目标 shape 上相对显式通用 CuTe 的最终结果：

| M,N,K | Selected path | Generic CuTe | Dispatched | Speedup | Validation |
|---|---|---:|---:|---:|---:|
| 128,4096,8192 | M128 Split-K=4 | 26.21 us | 13.03 us | 2.01x | 0 mismatches |
| 256,4096,8192 | M256 Split-K=2 | 26.33 us | 18.10 us | 1.45x | 0 mismatches |

DS-V4 sparse MLA 的正确性及性能待服务器验证，测试与 A/B 命令见 [Sparse MLA](docs/SPARSE_MLA.md)。

双 GPU Expert-major 对接基准（GPU 1、2，`SYS` 跨 NUMA PCIe，
256 tokens/rank，hidden 4096，intermediate 2048，top-k 2，32 Expert）：

| 路由 | 直接交接计算 P50 | 重复路由计算 P50 | 计算段加速 | 端到端加速 |
|---|---:|---:|---:|---:|
| 均衡 | 0.2365 ms | 0.3002 ms | 1.27x | 1.04x |
| 80% Rank 偏斜 | 0.1506 ms | 0.2049 ms | 1.36x | 1.03x |

两条路径最大绝对误差均为 0。该结果使用
`torch.distributed` 参考通信验证接口与控制路径，不代表 DeepEP 原生
NVLink/RDMA Kernel 性能；详细定义和原始数据见
[Performance](docs/PERFORMANCE.md)。

## Current limitations and roadmap

- 仅支持 `compute_120a/sm_120a`；
- sparse MLA 固定 DS-V4 `Hq=64,Hkv=1,D=448+64`、64-token 页；decode 支持 split，prefill 固定每 query 一个 CTA；尚不包含 compressor/indexer 或完整 attention block；
- prefill 要求调用方提供每个 query 的合法索引并维持缓存生命周期，支持按 query 切分调用，尚未实现 chunked prefill 的压缩缓存状态管理；
- sparse decode/prefill 的构建、数值、竞争检查和性能均待服务器验证；
- Custom CuTe 当前只有 `128 x 128 x 128`、3-stage 配置；
- 默认 GEMM 已完成 M=128/M=256 专用路径和其他 M 的通用回退；两条专用路径需要由 workspace query 返回的 FP32 临时空间；
- Custom CuTe 的 FP32-output 路径及 `M < 64` 的 FP16 路径仍使用线程直接写回；其余 FP16 tile 使用 TMA store；
- Grouped GEMM 要求所有 group 共享 N/K，M 由 `seqlens` 给出；
- GEMM 的 scale 需要 SM1xx 物理布局；sparse MLA 使用独立的 footer-scale cache ABI；
- Grouped GEMM 与完整路由版 Fused MoE binding 仍会分配部分临时 tensor；Expert-major 路径支持调用方 Workspace 复用；
- 当前双 GPU 基准使用参考通信验证接口，不包含 DeepEP 原生 NVLink/RDMA Kernel；
- cuBLASLt 当前对测试的 NVFP4 pointer-array grouped 配置没有可用原生算法。

GEMM 阶段收尾状态：

- [x] M=128 Split-K=4 专用路径、正确性与稳定性能验证；
- [x] M=256 Split-K=2 专用路径、`half2` reduction 与固定 task mapping；
- [x] 默认 host-side shape dispatch、workspace contract 与通用回退；
- [x] 显式通用 CuTe/CUTLASS reference 接口保持独立。

项目后续重点：

- [x] 为 Expert-major 路径增加调用方 Workspace 和 MoE 元数据复用；
- [x] 增加 Grouped GEMM 与双 GPU Fused MoE 对接的可复现 benchmark；
- [x] 实现 DS-V4 C++ CuTe sparse MLA core，并准备参考测试与 benchmark；
- [ ] 在服务器编译、验证新 sparse MLA，并基于 profile 优化；
- [ ] 接入 SGLang MoE runner 与 DeepEP 原生 dispatcher。

## Documentation

- [DS-V4 CSA Sparse MLA](docs/SPARSE_MLA.md)

- [Architecture](docs/ARCHITECTURE.md)
- [C++ and Python API](docs/API.md)
- [Performance](docs/PERFORMANCE.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT License。CUTLASS、CUDA、PyTorch 等依赖分别遵循其自身许可证。
