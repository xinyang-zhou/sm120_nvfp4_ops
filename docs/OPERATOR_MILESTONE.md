# 算子正确性与 benchmark 交付边界

2026-09-27：按最新项目范围，本仓库交付 SM120 kernel、正确性测试和 benchmark，
后续作为 SGLang 的 DS-V4 算子后端。独立模型加载、请求管理、完整 Transformer
执行程序不作为本轮目标。完整 block 和单层对照不是算子验收的前置条件。
后续 profile 优化及 SGLang adapter 独立推进。

本轮代码仅在 WSL 开发和静态检查。所有构建、测试、benchmark、sanitizer 和
profiler 采集都在服务器执行；下面的验收条件不是已通过的结果。

## 目标算子

| 算子 | 实现/接口 | 正确性与 benchmark |
|---|---|---|
| 单 GEMM | 通用 CuTe、M128 Split-K=4、M256 Split-K=2 | 现有 C++/Python tests；CuTe/CUTLASS/cuBLASLt 对照与 specialized dispatcher benchmark |
| Grouped GEMM | 动态 expert 行数、persistent CuTe | 现有 tests、cuBLASLt loop 基线；不把 loop 称为原生 grouped API |
| Expert quantize / MoE | NVFP4 量化、gate/up、SiLU/量化、down、路由/reduce | 现有 tests，加单 GPU `benchmark_moe_ops.py`；无需 DeepEP 或第二张 GPU |
| CSA decode / prefill core | 双缓存、给定 Top-K 与 SWA、sink、NVFP4/BF16 混合 QK/PV | 独立矩阵 reference、原有边界/graph suite、新 `benchmark_attention_ops.py` |
| HCA decode / prefill core | 同一 attention kernel，索引覆盖全部可见压缩条目 | ratio=128 的因果候选 fixture、压缩边界/请求隔离测试及同一 benchmark |
| KV pack / append | `sparse_mla_pack_cache`，原生 CUDA 量化/写入 | 独立字节级 ABI、page tail、scattered append、Graph 动态输入测试和独立 benchmark |

HCA 在此只表示“所有可见压缩条目＋SWA”的 core 工作负载，不声称已实现 HCA
compressor 或完整 HCA block。CSA 的 Top-K 同样在计时前生成。已有 stateful CSA
block 保留作参考实验，不是核心 suite 的依赖，也不继续承担生产请求生命周期。
compressor/indexer、norm/RoPE 的专用 CUDA 重写本轮后置，未来按 SGLang 实际
替换范围决定；外围可以复用框架算子。

主 attention 使用 NVFP4/BF16 实验合同；MoE 使用本仓库的 E2M1/UE4M3 合同。
这些 benchmark 不证明与官方模型精度等价，也不包含 checkpoint 模型质量结论。
权重格式、scale 布局和输出类型的框架适配仍属于后续交付。

## 一次运行

在服务器拉取包含本次改动的确定 commit，设置 CUDA_ROOT/CUTLASS_ROOT 后：

```bash
set -euo pipefail
mkdir -p artifacts/operator_build
SM120_NVFP4_BUILD_TESTS=ON SM120_NVFP4_BUILD_BENCHMARKS=ON \
  bash scripts/build.sh 2>&1 | tee artifacts/operator_build/build.log
CUDA_VISIBLE_DEVICES=0 bash scripts/benchmark_operators.sh \
  artifacts/operators_run_01 --flashinfer
```

输出目录必须不存在。`--flashinfer` 要求已有可导入的干净源码安装，revision 为
`37b4d30eac39b89f198b893dd11914bd76f5fcf8` 或
`ea728cb558c32a3c58ec8fbd5a154ff676b9ab70`。省略它可以先跑独立 reference，
但该轮不能标为完成 FlashInfer A/B。脚本不下载模型或安装框架。

执行顺序：C++ GEMM gate → Python operator gates → GEMM/grouped 三次独立
运行 → dispatcher 对照 → attention/cache/MoE。任何失败立即停止，保留日志。
已有 GEMM C++ benchmark 使用整段 CUDA-event 均值，specialized 输出保存为
三轮日志；它们不提供单次调用 P95，不能用三轮均值代替。新的 Python benchmark
每 case 默认三轮、每轮 100 个独立 event 样本，保存原始值与 P50/P95。

`correctness.json` 分别记录 pass/fail/skip；非 SM120、缺扩展、非可选测试跳过
都会失败。默认未启用的 FlashInfer 用例单独记为 optional skip。这里不运行
`test_dsv4_csa_block.py`，无需 safetensors 或模型权重。

## 单独运行和扩展矩阵

```bash
export PYTHONPATH="$PWD/build/python${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/validate_operators.py --flashinfer --output artifacts/gates_01.json

python3 benchmarks/benchmark_attention_ops.py \
  --modes decode prefill --kinds csa hca \
  --batches 1 4 8 16 64 --prefill-batches 1 4 --query-lengths 7 128 \
  --context-length 32768 --mixed-lengths --flashinfer \
  --output artifacts/attention_32k_01.json

python3 benchmarks/benchmark_sparse_mla_cache.py \
  --rows 1 63 64 65 128 1024 --flashinfer --output artifacts/cache_01.json

python3 benchmarks/benchmark_moe_ops.py \
  --rows 16 128 512 --experts 8 --hidden 4096 --intermediate 2048 \
  --routing balanced skewed --output artifacts/moe_01.json
```

attention 默认 CUDA Graph、256 MiB L2 干扰（在计时外）；`--eager` 单独测 eager
GPU event 区间，`--l2-flush-mib 0` 单独测热缓存。对照两侧的设置保持相同。
`--selection contiguous` 可替代 CSA 随机选择；HCA 始终按逻辑顺序读取全部可见条目。
原始 history、每 query 的位置/有效长度、实际压缩数量与分配容量分别记录。
跨请求物理地址独立，prefill fixture 中每 query 都有因果合法索引；缓存向量仍为
合成 post-compression/post-RoPE 数据。上下文按需扫描 0、1K、8K、32K、128K。

FlashInfer attention 对照使用固定版本的 allocation-free native 入口，显式匹配
CPB/候选顺序，双方 output/LSE/scratch 都预分配。prefill 双方一 query 一个 CTA。
这不是公共 auto planner 的性能结果。CSA decode 的 public-dispatch 对照已统一到
`benchmark_attention_ops.py --modes decode --kinds csa --flashinfer --flashinfer-dispatch public`，
实际 plan 会写入 JSON；该公共 API 不返回 LSE，因此只与本库比较输出，本库 LSE 仍对照独立参考。
不需要本库扩展的独立基线和 profiling 工具位于
`benchmarks/tools/benchmark_flashinfer_csa_decode.py`，两种结果不混成同一对照口径。
FlashInfer native decode 的保守 scratch 分配与本库紧凑 split 分配分别记录。

cache 的 pack 表示向预分配 cache 的顺序量化写入；append 使用不重复随机 slot，
检查未写入的行保留。FlashInfer 也调用预分配 append API，双方计时均不含分配。
目标 slot 唯一是本库输入合同；没有把 FlashInfer 的重复 slot 处理能力归到本库。

MoE fixture 使用随机 BF16 激活、expert 内常量权重及不同 expert scales，从而
通过独立解析参考检查完整计算；它不代表真实权重性能/质量分布。分别报告输入量化、
预量化 expert 计算、量化＋计算、量化＋重复路由。重复路由对照使用相同展开行，
top-k=1，是控制路径消融，不是 FlashInfer MoE baseline，也不包含网络通信。
既有双 GPU handoff benchmark 继续作为单独扩展。

## 验收和数据边界

- 必须先过正确性，再计时。attention gate 同时检查输出、RMSE 与 FP32 LSE；
  cache 要求所有 packed bytes 完全一致；MoE 对照解析输出并检查路由路径等价。
- attention benchmark 的初始门槛沿用现有合成 workload：`rtol=.05, atol=.02`、
  RMSE≤.005，LSE `rtol=2e-5, atol=5e-4`。FlashInfer 对照 `rtol=.07, atol=.03`、
  RMSE≤.005。MoE 解析参考 `rtol=.02, atol=.005`、RMSE≤.003。
  原有 unit suite 保留自己的更严格门槛；失败要定位，不能为得到性能表放宽阈值。
- JSON 包含实际工作区状态、源码 hashes、commit、命令、设备、精度、缓存/索引/
  workspace 字节数与原始样本。峰值 allocation 明确包含准备/reference/测量临时量，
  不是 kernel 自身的显存需求。每完成一个 case 即写入；异常保留失败 case 和 traceback。
- 计时包含 attention 内 Q/P/V 转换与 merge，排除 compressor/indexer/projection。
  不将 core queries/s 标成模型 tokens/s。当前不报告 block/layer/serving 收益。

新增 cache kernel 及 core 调度边界还需要 sanitizer：

```bash
set -o pipefail
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python3 tests/python/test_sparse_mla_cache.py \
  2>&1 | tee artifacts/cache_memcheck.log
compute-sanitizer --tool racecheck --error-exitcode 1 \
  python3 tests/python/test_sparse_mla_cache.py \
  2>&1 | tee artifacts/cache_racecheck.log
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python3 tests/python/test_sparse_mla_workloads.py \
  2>&1 | tee artifacts/attention_workloads_memcheck.log
```

sanitizer 零错误是验收要求；core 原有 racecheck 命令仍见 `SPARSE_MLA.md`。
失败时保留 build/test/sanitizer stdout/stderr、JSON、commit 与工作区源码。
服务器结果返回前，本轮状态只能标为“实现与验证入口已准备，GPU 验收待执行”。
