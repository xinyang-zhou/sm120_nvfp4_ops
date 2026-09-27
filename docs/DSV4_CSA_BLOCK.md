# DeepSeek-V4-Flash CSA attention block

2026-09-27 范围调整：本文保留已有 block 的参考/验证设计。当前优先交付独立
kernel、正确性与 benchmark，见 [算子里程碑](OPERATOR_MILESTONE.md)。
本 block 不作为未来 SGLang 后端的请求运行时，也不再是算子验收的前置条件。

本次实现覆盖 CSA 候选生成、状态化 prefill/decode 和完整 attention 分支。
这里新增的 block、cache append 和测试已完成源码开发与静态检查，仍需在 SM120
服务器验证。本 block 验证入口不采集性能数据；独立算子 benchmark 见算子里程碑。

## 固定参考与边界

模型参考固定为 `deepseek-ai/DeepSeek-V4-Flash` 的
[`60d8d70770c6776ff598c94bb586a859a38244f1`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1)。
具体依据是 [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py)
中的 `Compressor`、`Indexer`、`Attention` 和 attention-side mHC，以及
[inference/kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py)
中的量化与 Sinkhorn。文件 SHA-256 和完整官方配置记录在
[dsv4_flash_reference.json](dsv4_flash_reference.json)。默认选取第 2 层（从 0 开始）的 CSA。

固定 attention 几何为 `Hq=64,Hkv=1,D=448+64,SWA=128,TopK=512,compress_ratio=4`，
indexer 为 `64×128`。默认 hidden size=4096、Q rank=1024、O rank=1024、O groups=8。
单 GPU、单 attention 层；支持不同请求的独立状态和不同 prefix 长度。
HCA、CSA2、FFN、整个 Transformer、TP/CP、多卡和 serving 调度不属于这个 CSA block。
`forward_hc` 包含完整 attention 分支的 mHC 前后混合，但不会运行 FFN 分支。

## 数据流与精度

```text
hidden -> attention RMSNorm -> x
   ├─ wq_a -> learned q_norm -> wq_b -> head normalization -> RoPE -> Q
   ├─ wkv -> learned kv_norm -> RoPE -> NVFP4 SWA cache
   ├─ attention compressor -> norm -> block-start RoPE -> NVFP4 compressed cache
   └─ indexer compressor -> norm -> block-start RoPE -> Hadamard -> MXFP4 keys
          qr -> indexer wq_b -> RoPE -> Hadamard -> MXFP4 query
          x -> head weights; weighted ReLU QK -> causal Top-K
   -> joint SWA + CSA + sink CuTe attention
   -> inverse RoPE -> grouped wo_a -> wo_b -> output
```

attention core 保持 [数学合同](attention_decode_math.tex) 的 NVFP4/BF16 混合规则：主缓存 non-RoPE
采用每 16 通道 E2M1+E4M3，RoPE 保留 BF16；Q/P 在线量化，V 沿候选轴重新量化，
分母与累加为 FP32，sink 加入一次。block 的 decode 固定 `chunks_per_cta=10`，
和 prefill 一样采用单 CTA 完整候选流，避免 query 分块改变 split 舍入规则。

indexer 是独立的 **MXFP4** 合同：每 32 元素共享一个 2 的整数次幂 scale，
RNE E2M1、归一化 Hadamard，缓存保存 packed payload 与 E8M0 编码。
它不复用主 attention 的 E4M3/16 scale。评分按官方 BF16 dot、ReLU、BF16 head
weight 乘法和 head 求和的顺序进行。GPU 上维护最多 512 个候选；score tile
默认是 32 queries × 64 heads × 256 compressed entries，不落地完整长序列评分矩阵。

精确同分时按较小逻辑 slot 优先，选中集合最终按逻辑位置排序；不足 512 的部分填 -1。
这明确了官方 `torch.topk` 未约定的同分顺序，并固定 NVFP4 候选分组。
FlashInfer A/B 接收相同排序后的候选，不能拿另一种候选顺序下的输出宣称逐位等价。

外围投影、norm、压缩、RoPE、Hadamard、Top-K 与编排使用 PyTorch GPU 算子。
核心 QK/PV 仍使用 C++ CuTe；新增 `cache.cu` 负责原生 NVFP4 打包/追加，
复用主 kernel 的同一个 `quantize16`。原量化计算语句未改变。

权重加载遵循官方格式：BF16/FP32 直接投影；FP8 `.weight` + `.scale` 使用 per-128
激活量化和 per-128×128 权重 scale，在 FP32 中累加各 K 块后输出 BF16。
这个 FP8 路径用 PyTorch GEMM 实现，优先用于集成验证。`wo_a` 和官方参考一样，
将 checkpoint 中的 FP8 权重解量化到 BF16 后做 grouped projection。
compressor 的投影和 softmax pooling 使用 FP32，pooling 结果转 BF16 后做 learned norm；
query head norm 保留官方 BF16 表达式，learned RMSNorm 在 FP32 中计算。

**这里仍是原生 NVFP4 attention 实验路径。** 官方参考的主 KV FP8 模拟被既有 NVFP4
cache/QK/PV 合同取代，没有先做 FP8 再做 NVFP4。加载真实权重不等于已经证明原模型
质量等价；block 测试验证相同 NVFP4 合同下的实现误差，原模型精度差异需独立评估。

## 压缩、可见性与缓存生命周期

- CSA compressor 对每个 token 生成两个 half 的 KV 和 gate，加上 `ape[position % 4]`。
  一个新条目使用前一完整块的 first half 与当前完整块的 second half，共最多 8 个 token，
  在每个通道上独立进行 gated softmax pooling；首块缺失的前窗权重为零。
- 压缩条目 j 使用 RoPE 位置 `4*j`，从 query 位置 `4*j+3` 开始可见。
  query p 的有效压缩条目数是 `floor((p+1)/4)`；不读取未来条目。
- attention compressor 和 indexer compressor 各自保留前一完整块，以及 0～3 个
  未完成 token 的投影和 gate；从任意非对齐 query chunk 继续都沿用这些状态。
- SWA 状态只保留 128 个 slot。每个 query chunk 构造“旧环形窗口＋本轮 KV”的临时池，
  按 query 的绝对位置生成索引。attention 完成读取后才将本轮最后至多 128 个 KV
  按 `position % 128` 提交回环形缓存。复制的是 packed 行，不二次量化。
- 每个请求的两个主缓存和 indexer 缓存独立，避免跨请求引用。`state.fork()` 深复制
  prefix 缓存、完整块和未完成尾状态；`reset()` 清空整个请求状态。

持久缓存随设定 context capacity 增长：主压缩缓存约 384 字节/4 tokens，indexer
约 68 字节/4 tokens，另有固定 SWA 页和小型压缩状态。瞬时 Q/O、KV 和打分存储
受 `query_chunk_size` 与 indexer tile 大小约束。诊断 `trace` 会主动保留中间 tensor，
长序列正常运行应省略它。独立测试 reference 为易于审阅可以使用完整前缀矩阵，
不属于生产 forward。

## 使用方式

```python
import torch
from sm120_nvfp4 import DSV4CSAAttentionBlock

# 仅加载官方本地 snapshot 中第 2 层 attention 及其 mHC 权重。
# 需要 safetensors，以及可读取 checkpoint FP8/E8M0 dtype 的 PyTorch。
block = DSV4CSAAttentionBlock.from_checkpoint(
    "/path/to/DeepSeek-V4-Flash/snapshot", layer_id=2,
    device="cuda", query_chunk_size=128,
)
a, b = block.new_state(max_seq_len=8192), block.new_state(max_seq_len=8192)

# hidden: BF16 [T,4096]，输入到本层 attention norm 前的值。
# 多请求按 request 顺序拼接，长度是 CPU Python 整数。
y = block.prefill(hidden, [a, b], query_lengths=[len_a, len_b])
y_next = block.decode(next_hidden, [a, b])  # next_hidden: BF16 [2,4096]

# 非对齐 chunk 会保留 compressor residual state。
y_chunk = block.prefill(more_hidden, [a], [more_hidden.shape[0]])
branch = a.fork()
branch.reset()

# 若上游仍是 [T,4,4096] 的 mHC 流，使用 attention 分支的完整前后混合。
y_hc = block.forward_hc(hc_hidden, [new_request_state], [hc_hidden.shape[0]])
```

不加载 checkpoint 的 `DSV4CSAAttentionBlock()` 使用初始化权重，仅用于开发 fixture，
不是预训练模型。输入必须是有限 BF16 hidden states；Q/KV/indexer 都由 block 内部生成。
原始 `sparse_mla_decode` / `sparse_mla_prefill` API 继续可独立使用。

`backend="cute"` 为默认值。`backend="flashinfer"` 在同一编排中仅替换 attention core，
用于同输入、同缓存、同候选、同精度的实现 A/B；使用已审阅 FlashInfer revision
`37b4d30eac39b89f198b893dd11914bd76f5fcf8` 或 `ea728cb558c32a3c58ec8fbd5a154ff676b9ab70`。
显式传入 reference callable 可用于诊断，不会自动回退到它。

状态属于创建它的 block 实例和 CUDA stream。不同状态可分别处理；同一状态不得并发使用、
换 stream、修改权重后继续复用或手工更改内部计数。创建新状态或 reset 后再开启新序列。
异常发生后，该请求状态标为 failed，需 reset 后重试。请求长度/容量错误会在修改状态前拒绝。
全 block 使用 host 请求长度并有动态分配，**明确拒绝 CUDA Graph capture**；
fixed-index core 的 CUDA Graph 支持保持不变。当前按请求和 query chunk 发起计算，
没有宣称 serving 级批处理调度已优化。

## 服务器验证

以下命令只在 SM120 服务器运行。先设置已有构建需要的 `CUDA_ROOT`、`CUTLASS_ROOT`。
新 pack kernel 和共享 quantizer 需要重新构建，已通过的 decode/prefill suite 也做回归。

```bash
git switch main
git pull --ff-only
mkdir -p artifacts/dsv4_csa
set -euo pipefail
git rev-parse HEAD | tee artifacts/dsv4_csa/commit.txt
nvidia-smi > artifacts/dsv4_csa/nvidia-smi.txt
python3 -m pip install safetensors
SM120_NVFP4_BUILD_BENCHMARKS=OFF bash scripts/build.sh \
  2>&1 | tee artifacts/dsv4_csa/build.log
export PYTHONPATH="$PWD/build/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import torch; assert torch.cuda.get_device_capability() == (12, 0)'
python3 -m unittest discover -s tests/python -p 'test_sparse_mla*.py' -v \
  2>&1 | tee artifacts/dsv4_csa/core_regression.log
python3 -m unittest discover -s tests/python -p test_dsv4_csa_block.py -v \
  2>&1 | tee artifacts/dsv4_csa/block_tests.log
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python3 tests/python/test_dsv4_csa_block.py DSV4CSABlockTest.test_native_cache_pack_append_and_footer \
  2>&1 | tee artifacts/dsv4_csa/cache_memcheck.log
compute-sanitizer --tool racecheck --error-exitcode 1 \
  python3 tests/python/test_dsv4_csa_block.py DSV4CSABlockTest.test_prefill_chunks_then_decode_state_and_outputs \
  2>&1 | tee artifacts/dsv4_csa/state_racecheck.log
python3 scripts/validate_dsv4_block.py --tokens 137 --decode-tail 5 \
  --output artifacts/dsv4_csa/initialized_block.json \
  2>&1 | tee artifacts/dsv4_csa/initialized_block.log
```

真实 checkpoint 与 FlashInfer 对照：

```bash
export DSV4_CHECKPOINT=/path/to/pinned/DeepSeek-V4-Flash/snapshot
SPARSE_MLA_FLASHINFER_TEST=1 python3 -m unittest discover \
  -s tests/python -p test_dsv4_csa_block.py -v \
  2>&1 | tee artifacts/dsv4_csa/checkpoint_and_flashinfer.log
python3 scripts/validate_dsv4_block.py --checkpoint "$DSV4_CHECKPOINT" \
  --layer-id 2 --tokens 137 --decode-tail 5 --flashinfer \
  --output artifacts/dsv4_csa/checkpoint_block.json \
  2>&1 | tee artifacts/dsv4_csa/checkpoint_block.log
```

若有真实 layer input，用 `--hidden input.pt` 替代合成 hidden；文件内容为
`torch.save` 保存的 BF16 `[T,4096]` tensor，位于 attention norm 之前、mHC pre 之后。
下载/存放权重由调用方完成；loader 只读取单层所需 tensor，不联网加载整个模型。
JSON 区分初始化/预训练权重以及合成/真实 hidden 输入，不能将合成 fixture 当模型质量结论。

验收门槛：所有非可选测试通过，sanitizer 零错误。测试包含 native cache ABI、MXFP4
舍入、Hadamard、重叠压缩、partial tail、超过 512 个候选时的 Top-K 与可见性、独立
全前缀 block reference、SWA 跨 128 边界、prefill→decode、ragged 请求、fork/reset、
mHC、FP8 投影、官方 shard 键名/scale 加载，以及默认 4096/1024 projection 几何。
reference 从 hidden 重新构造压缩窗口与候选，不读取 production 选出的索引。

初始 block 数值门槛为逐元素 `rtol=.06,atol=.015` 且 RMSE≤.003；状态的 packed bytes
逐位相同，FP32 压缩尾状态按 `rtol=atol=2e-5` 对照。Top-K 选中 logical slots 按上述
同分规则精确匹配。JSON 保存 max-abs、RMSE、relative L2、非有限值数量、候选差异和
状态差异。门槛是待服务器检验的实现验收条件，不是已经获得的结果。

失败时保留完整 build/test/sanitizer stdout/stderr、commit、环境和 JSON。
缺真实 checkpoint 时该项可选测试会跳过；“可选项跳过”不能被描述为真实权重已验证。
本次没有增加或运行性能 benchmark。
