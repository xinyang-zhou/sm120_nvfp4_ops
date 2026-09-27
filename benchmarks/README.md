# Benchmark 导航

主目录只保留五个常用算子入口。辅助代码、专项诊断和集成实验分开存放；
`results/` 只保留公开结果索引，最新原始数据和本机报告放入 Git 忽略的 `temp/`。
原始文件可能包含个人路径、设备 UUID 和工作区状态，不直接提交。

## 常用入口

| 文件 | 测量对象 / 对照 |
|---|---|
| `benchmark_gemm_vs_cublaslt.cu` | 单 GEMM：通用 CuTe / CUTLASS reference / cuBLASLt |
| `benchmark_grouped_gemm_vs_cublaslt.cu` | 等长专家 grouped GEMM / 多次独立 cuBLASLt 调用（不是原生 grouped API） |
| `benchmark_attention_ops.py` | CSA/HCA、decode/prefill；独立参考与可选 FlashInfer 对照 |
| `benchmark_sparse_mla_cache.py` | BF16→NVFP4 cache 的顺序 pack / 随机 slot append；逐字节参考与可选 FlashInfer |
| `benchmark_moe_ops.py` | 单卡量化、expert 计算、量化+计算、重复路由对照；无跨卡通信 |

C++ 文件编译后从 `build/` 运行，构建目标名不变。Python 入口需要相应的
`sm120_nvfp4` 扩展。仅包含 attention 的独立构建不能用于运行 GEMM/MoE Python 算子。

## Attention：统一两种对照口径

从项目根目录运行，GPU 通过可见设备限制为一张；下面的 JSON 路径必须未存在。
`--batches` 表示普通 decode 的请求数，每个请求一个 query token。

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/build/python${PYTHONPATH:+:$PYTHONPATH}"

# 默认 native：同输入、同 CPB（每个 CTA 的候选块数），覆盖 CSA/HCA。
python3 benchmarks/benchmark_attention_ops.py \
  --modes decode --kinds csa hca --batches 64 512 1024 \
  --context-length 32768 --flashinfer --output artifacts/attention_native_new.json

# public：同输入，但由 FlashInfer 公共 planner 自行选择调度；仅 CSA decode。
python3 benchmarks/benchmark_attention_ops.py \
  --modes decode --kinds csa --batches 64 512 1024 \
  --context-length 32768 --flashinfer --flashinfer-dispatch public \
  --output artifacts/attention_public_new.json

# Prefill 的请求数和每请求 query 长度分开指定。
python3 benchmarks/benchmark_attention_ops.py \
  --modes prefill --kinds csa hca --prefill-batches 1 4 --query-lengths 7 128 \
  --flashinfer --output artifacts/attention_prefill_new.json
```

native 对照检查全部输出及 LSE；public API 不返回 LSE，只检查全部输出，
本库 LSE 仍与独立参考比较。public 记录实际 plan，并检查计时前后 plan 不变。
public 对照不强制匹配本库 CPB。两种口径分别保存，不混成一个“加速比”。

旧 `benchmark_sparse_mla.py` 的 CSA decode/public A/B 功能已合并进上述入口，
旧文件移除，不保留转发脚本。统一 fixture 使用按时间排序的随机唯一 CSA 候选，
与旧脚本的随机候选顺序不同；迁移前后的结果不能当成逐样本复现。
默认参数没有改成 64/512/1024，请像上例显式指定目标 batch。

## 专项工具：tools/

- [独立 FlashInfer 基线及 profiling](tools/README_FLASHINFER_CSA_BASELINE.md)：
  `tools/benchmark_flashinfer_csa_decode.py` 不需要本库扩展，可用 `--profile-once`
  抓取一次调用；公共 API、BF16 精度诊断和原有计时逻辑保留。
- `tools/benchmark_gemm_specialized.cu`：默认 GEMM dispatcher 与通用 CuTe 的内部比较。
  可执行文件仍为 `build/benchmark_gemm_specialized`。
- `tools/probe_cublaslt_grouped.cu`：固定配置下的 grouped cuBLASLt 算法能力探测，
  不是性能或完整正确性测试；可执行文件仍为 `build/probe_cublaslt_grouped`。

## 集成实验：integration/

- `integration/benchmark_deepep_handoff.py`：恰好两张 GPU，
  `torch.distributed` 参考 dispatch→expert→combine，不是原生 DeepEP 性能。

该脚本只迁移目录，没有改动计算或集成逻辑；单卡流程不启动双卡实验。

## 公共工具和结果边界

`common/operator_benchmark_utils.py` 统一 Python 计时、环境/源码 hash、误差和 JSON；
`common/flashinfer_public.py` 共用固定版本的 public plan 检查；
`common/benchmark_io.hpp` 共用 C++ JSON/CSV 输出。

Python 算子主入口默认用 CUDA Graph、3×100 样本、计时外 256 MiB L2 扰动，
输出 P50/P95；C++ GEMM 程序报告多次调用的平均时间，不能直接混用统计口径。
合成算子结果不是完整模型 token/s。公开性能摘要见主 README 与 `docs/PERFORMANCE.md`；
旧 DeepEP 性能产物已移出源码树，最新原始 attention 结果仅在本机 `temp/` 保留。

## 旧路径迁移

| 旧路径（均相对 benchmarks/） | 新位置 / 替代 |
|---|---|
| `benchmark_sparse_mla.py` | `benchmark_attention_ops.py --modes decode --kinds csa`；原 `--flashinfer` 对应额外指定 `--flashinfer-dispatch public` |
| `benchmark_flashinfer_csa_decode.py` | `tools/benchmark_flashinfer_csa_decode.py` |
| `benchmark_gemm_specialized.cu`、`probe_cublaslt_grouped.cu` | `tools/`，二进制名不变 |
| `benchmark_deepep_handoff.py` | `integration/` |
| `operator_benchmark_utils.py`、`benchmark_io.hpp` | `common/` |

历史 `artifacts/` 中的复现脚本不回写；若引用上述旧路径，按此表迁移。
