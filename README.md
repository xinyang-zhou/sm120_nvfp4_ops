# SM120 NVFP4 Operator Stack

面向 NVIDIA SM120（GeForce RTX 50 系列）的 NVFP4 高性能算子库，覆盖从自定义单矩阵乘法到端到端 Mixture-of-Experts 的完整计算链路：

```text
Custom CuTe NVFP4 GEMM
    -> persistent CuTe Grouped GEMM
    -> routing + gate/up + SiLU/quant + down + reduce
    -> Fused MoE
```

单 GEMM 与 Grouped GEMM 都由仓库自有的 `__global__` kernel 实现，直接使用 CuTe tensor/partition/copy/gemm abstraction 组织 TMA、shared-memory pipeline 和 SM120 block-scaled MMA。CUTLASS `GemmUniversalAdapter` 版本仅作为 reference 保留，不是默认执行路径。

输入使用 packed E2M1，scale 使用 UE4M3，累加为 FP32，输出为 FP16。

> 项目目前处于 research preview 阶段：核心 kernel 和正确性测试已在 RTX 5090 上运行，workspace 管理、epilogue 和 shape autotuning 仍会继续演进。

## Highlights

- 仓库自有 Custom CuTe 单 GEMM，不调用 CUTLASS `GemmUniversal` 或 `GemmUniversalAdapter`；
- 显式实现 384-thread producer/consumer warp specialization、三阶段 TMA pipeline 和 persistent CTA tile scheduling；
- 使用 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` 原生 block-scaled Tensor Core 指令；
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
│   │   ├── cute_gemm.cu       # 仓库自有 Custom CuTe kernel
│   │   └── cutlass_reference.cu
│   ├── grouped_gemm/          # persistent expert GEMM
│   └── fused_moe/             # gather/activation/reduce 与 orchestration
├── bindings/                  # PyTorch custom-op registration
├── python/sm120_nvfp4/        # Python 友好接口
├── tests/                     # C++ 与 Python 正确性测试
├── benchmarks/                # 三方 GEMM 对比与 grouped API 探测
├── scripts/                   # build/test/benchmark 一键脚本
└── docs/                      # API、架构、性能与 roadmap
```

原始开发目录 `sm120/`、`sm120_group_gemm/` 和 `sm120_fuse_moe/` 暂时保留用于实验回溯；公开仓库应以本目录为唯一源码树。

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

不构建 PyTorch extension：

```bash
cmake -S . -B build \
  -DCUTLASS_ROOT="$CUTLASS_ROOT" \
  -DSM120_NVFP4_BUILD_TORCH=OFF
```

测试覆盖：

- Custom CuTe 单 GEMM CPU reference；
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

# 默认路径就是 Custom CuTe。
y = sm120_nvfp4.gemm(a, weight, a_scale, weight_scale)
y_cute = sm120_nvfp4.cute_gemm(a, weight, a_scale, weight_scale)

# 仅用于对照。
y_reference = sm120_nvfp4.cutlass_gemm(
    a, weight, a_scale, weight_scale
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

平台：RTX 5090、CUDA 13.2、cuBLASLt 13.4、CUTLASS 4.2.1。所有路径使用相同随机 packed E2M1 数据与 UE4M3 scale，并逐元素验证。

| M,N,K | Implementation | Latency | Throughput | vs cuBLASLt |
|---|---|---:|---:|---:|
| 16,4096,8192 | Custom CuTe | 25.2 us | 42.56 TFLOP/s | 104.34% |
| 16,4096,8192 | CUTLASS reference | 26.2 us | 41.01 TFLOP/s | 100.49% |
| 16,4096,8192 | cuBLASLt id 70 | 26.3 us | 40.81 TFLOP/s | 100% |
| 512,4096,8192 | Custom CuTe | 34.8 us | 987.05 TFLOP/s | 85.56% |
| 512,4096,8192 | CUTLASS reference | 28.3 us | 1215.47 TFLOP/s | 105.36% |
| 512,4096,8192 | cuBLASLt id 70 | 29.8 us | 1153.62 TFLOP/s | 100% |

`M=16` 说明手写 mainloop 能降低通用 adapter 的固定开销；`M=512` 暴露了当前 scalar/predicated epilogue 与单一 tile 配置的不足。完整测量方法和历史 CUTLASS sweep 见 [Performance](docs/PERFORMANCE.md)。

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
- Custom CuTe 当前只有 `128 x 128 x 128`、3-stage 配置；
- Custom CuTe epilogue 仍由线程直接写 global memory，尚未使用向量化 copy/TMA store；
- Grouped GEMM 要求所有 group 共享 N/K，M 由 `seqlens` 给出；
- scale 必须预先转换为 SM1xx 物理布局；
- Grouped GEMM 与完整路由版 Fused MoE binding 仍会分配部分临时 tensor；Expert-major 路径支持调用方 Workspace 复用；
- 当前双 GPU 基准使用参考通信验证接口，不包含 DeepEP 原生 NVLink/RDMA Kernel；
- cuBLASLt 当前对测试的 NVFP4 pointer-array grouped 配置没有可用原生算法。

下一步重点：

- [ ] 为 `M=16/32/64` 与中大 M 分别增加 tile/stage specialization；
- [ ] 用 vectorized shared-memory epilogue 或 TMA store 替换 scalar store；
- [ ] 增加 host-side shape dispatch 与离线 autotuning；
- [x] 为 Expert-major 路径增加调用方 Workspace 和 MoE 元数据复用；
- [x] 增加 Grouped GEMM 与双 GPU Fused MoE 对接的可复现 benchmark；
- [ ] 接入 SGLang MoE runner 与 DeepEP 原生 dispatcher。

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [C++ and Python API](docs/API.md)
- [Performance](docs/PERFORMANCE.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT License。CUTLASS、CUDA、PyTorch 等依赖分别遵循其自身许可证。
