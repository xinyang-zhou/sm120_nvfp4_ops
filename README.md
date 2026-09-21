# SM120 NVFP4 Operator Stack

面向 NVIDIA SM120（GeForce RTX 50 系列）的 NVFP4 高性能算子库，覆盖自定义矩阵乘、dense prefill/decode attention 与端到端 Mixture-of-Experts：

```text
Custom CuTe NVFP4 GEMM
    ├── QK^T -> FP32 softmax/quant -> P@V -> Attention
    └── persistent CuTe Grouped GEMM
          -> routing + gate/up + SiLU/quant + down + reduce
          -> Fused MoE
```

单 GEMM 与 Grouped GEMM 都由仓库自有的 `__global__` kernel 实现，直接使用 CuTe tensor/partition/copy/gemm abstraction 组织 TMA、shared-memory pipeline 和 SM120 block-scaled MMA。CUTLASS `GemmUniversalAdapter` 版本仅作为 reference 保留，不是默认执行路径。

输入使用 packed E2M1，scale 使用 UE4M3，累加为 FP32，输出为 FP16。

> 项目目前处于 research preview 阶段：核心 kernel 和正确性测试已在 RTX 5090 上运行；单 GEMM 的通用、M=128 与 M=256 路径已经收尾，后续工作集中在 attention/MoE 集成。

## Highlights

- 仓库自有 Custom CuTe 单 GEMM，不调用 CUTLASS `GemmUniversal` 或 `GemmUniversalAdapter`；
- 默认单 GEMM 按 M 自动分发：M=128 使用 Split-K=4、M=256 使用 Split-K=2，其他 M 使用通用 CuTe；
- 显式实现 384-thread producer/consumer warp specialization、三阶段 TMA pipeline 和 persistent CTA tile scheduling；
- FP16 epilogue 对不少于 64 行的问题使用 shared-memory staging 与 TMA store，小 M 保留低开销 predicated store；
- 使用 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` 原生 block-scaled Tensor Core 指令；
- dense prefill/decode attention 的 `QK^T` 和 `P@V` 都使用原生 NVFP4 MMA，softmax 保持 FP32，并校正量化后概率行和；
- decode 支持单 token、GQA/MQA、GPU `kv_lengths`、paged KV block table
  和调用方 workspace 复用；
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
│   ├── attention/             # dense prefill/decode QK/softmax/PV
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
- dense prefill 非因果、右对齐 causal mask、非均匀双-head FP32 reference；
- dense decode GQA、动态 KV 长度、buffer 复用和非均匀 FP32 reference；
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

# V 需要预先按 [B,H,Dv,N/2] 转置并量化。
attention_output = sm120_nvfp4.attention_prefill(
    query_nvfp4,
    key_nvfp4,
    value_transposed_nvfp4,
    query_scale,
    key_scale,
    value_scale,
    causal=True,
)

# 单 token dense decode；Hq 必须是 Hkv 的整数倍。
decode_output = sm120_nvfp4.attention_decode(
    decode_query_nvfp4,
    key_cache_nvfp4,
    value_cache_transposed_nvfp4,
    decode_query_scale,
    key_cache_scale,
    value_cache_scale,
    kv_lengths=kv_lengths_cuda_int32,
)

# 单 token paged decode；物理页大小 S 支持 32、64、128。
paged_decode_output = sm120_nvfp4.attention_paged_decode(
    decode_query_nvfp4,
    paged_key_cache_nvfp4,              # [P,Hkv,S,D/2]
    paged_value_cache_transposed_nvfp4, # [P,Hkv,Dv,S/2]
    decode_query_scale,
    paged_key_scale,
    paged_value_scale,
    block_table_cuda_int32,             # [B,max_blocks]
    kv_lengths_cuda_int32,              # [B]
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

单 token decode attention（`Hq=32`、`Hkv=8`、`N=1024`、
`D=Dv=128`，30 次 warmup、300 次 CUDA Event 计时）：

| Batch | Page size | Dense | Paged | Paged / Dense |
|---:|---:|---:|---:|---:|
| 1 | 32 | 18.28 us | 31.39 us | 1.72x |
| 1 | 64 | 18.10 us | 30.46 us | 1.68x |
| 1 | 128 | 17.96 us | 29.94 us | 1.67x |
| 8 | 32 | 22.89 us | 109.81 us | 4.80x |
| 8 | 64 | 22.90 us | 107.45 us | 4.69x |
| 8 | 128 | 22.87 us | 106.43 us | 4.65x |

两条路径使用相同随机 NVFP4 数据、预分配 output/workspace，所有输出逐元素
完全一致。复现脚本、测量说明和原始 JSON 见 [Performance](docs/PERFORMANCE.md)。

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
- attention 支持 dense prefill，以及单 token dense/paged decode；要求 `D % 32 == 0`、padded `N % 32 == 0`、`Dv % 8 == 0`；
- decode 支持 GQA/MQA、动态有效长度、paged KV cache 和 split-K/LSE combine；尚未支持 MTP 或动态 device task map；
- prefill 当前仍会 materialize FP32 logits 和 NVFP4 probability workspace；两种 decode 路径均融合 QK、在线 softmax 与 PV，不落地完整 logits/probability；
- Custom CuTe 当前只有 `128 x 128 x 128`、3-stage 配置；
- 默认 GEMM 已完成 M=128/M=256 专用路径和其他 M 的通用回退；两条专用路径需要由 workspace query 返回的 FP32 临时空间；
- Custom CuTe 的 FP32-output 路径及 `M < 64` 的 FP16 路径仍使用线程直接写回；其余 FP16 tile 使用 TMA store；
- Grouped GEMM 要求所有 group 共享 N/K，M 由 `seqlens` 给出；
- scale 必须预先转换为 SM1xx 物理布局；
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
- [x] 增加 dense/paged decode attention 的可复现 benchmark；
- [ ] 专门化 paged KV gather，降低 block-table 间接寻址开销；
- [ ] 接入 SGLang MoE runner 与 DeepEP 原生 dispatcher。

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [C++ and Python API](docs/API.md)
- [Performance](docs/PERFORMANCE.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT License。CUTLASS、CUDA、PyTorch 等依赖分别遵循其自身许可证。
