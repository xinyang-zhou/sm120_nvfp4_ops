#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>
#include <torch/library.h>
#include "sm120_nvfp4/sparse_mla.hpp"

namespace {

void check(const torch::Tensor& t, int device, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.get_device() == device && t.is_contiguous(),
              name, " must be contiguous on query's CUDA device");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has an incorrect dtype");
}

bool overlaps(const torch::Tensor& a, const torch::Tensor& b) {
  if (!a.numel() || !b.numel()) return false;
  auto x = reinterpret_cast<std::uintptr_t>(a.const_data_ptr());
  auto y = reinterpret_cast<std::uintptr_t>(b.const_data_ptr());
  return x < y + b.nbytes() && y < x + a.nbytes();
}

int pages(const torch::Tensor& cache, int device, const char* name) {
  check(cache, device, torch::kUInt8, name);
  bool valid = (cache.dim() == 3 && cache.size(1) == 64 && cache.size(2) == 384) ||
               (cache.dim() == 4 && cache.size(3) == 384 &&
                ((cache.size(1) == 1 && cache.size(2) == 64) ||
                 (cache.size(1) == 64 && cache.size(2) == 1)));
  TORCH_CHECK(valid, name, " must have FlashInfer packed shape [P,64,384], [P,1,64,384], or [P,64,1,384]");
  TORCH_CHECK(cache.size(0) <= std::numeric_limits<int>::max() / 64,
              name, " has too many pages for int32 slot IDs");
  return static_cast<int>(cache.size(0));
}

int capacity(const torch::Tensor& indices, int device, int batch, const char* name) {
  check(indices, device, torch::kInt32, name);
  TORCH_CHECK((indices.dim() == 2 || (indices.dim() == 3 && indices.size(1) == 1)) &&
                  indices.size(0) == batch,
              name, " must have shape [B,K] or [B,1,K]");
  TORCH_CHECK(indices.size(-1) <= 1048576, name, " capacity exceeds 1048576");
  return static_cast<int>(indices.size(-1));
}

std::tuple<torch::Tensor, torch::Tensor> sparse_mla_decode_torch(
    const torch::Tensor& query, const torch::Tensor& swa_cache,
    const torch::Tensor& compressed_cache, const torch::Tensor& swa_indices,
    const torch::Tensor& compressed_indices,
    std::optional<torch::Tensor> swa_lengths,
    std::optional<torch::Tensor> compressed_lengths,
    std::optional<torch::Tensor> sink, std::int64_t chunks_per_cta,
    double softmax_scale, double lse_scale,
    std::optional<torch::Tensor> output, std::optional<torch::Tensor> lse,
    std::optional<torch::Tensor> workspace) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  int device = query.get_device();
  c10::cuda::CUDAGuard guard(device);
  check(query, device, torch::kBFloat16, "query");
  TORCH_CHECK((query.dim() == 3 || (query.dim() == 4 && query.size(1) == 1)) &&
                  query.size(-2) == 64 && query.size(-1) == 512,
              "query must be BF16 [B,64,512] or [B,1,64,512]");
  TORCH_CHECK(query.size(0) > 0 && query.size(0) <= 1048576,
              "batch must be in [1,1048576]");
  TORCH_CHECK(chunks_per_cta >= 0 && chunks_per_cta <= std::numeric_limits<int>::max(),
              "chunks_per_cta must be a nonnegative int32");
  TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0 &&
                  std::isfinite(static_cast<float>(softmax_scale)) &&
                  static_cast<float>(softmax_scale) > 0,
              "softmax_scale must be finite and positive in FP32");
  TORCH_CHECK(std::isfinite(lse_scale) && lse_scale > 0 &&
                  std::isfinite(static_cast<float>(lse_scale)) &&
                  static_cast<float>(lse_scale) > 0,
              "lse_scale must be finite and positive in FP32");
  sm120_nvfp4::SparseMlaDecodeParams p;
  p.batch = static_cast<int>(query.size(0));
  p.swa_pages = pages(swa_cache, device, "swa_cache");
  p.compressed_pages = pages(compressed_cache, device, "compressed_cache");
  p.swa_candidates = capacity(swa_indices, device, p.batch, "swa_indices");
  p.compressed_candidates = capacity(compressed_indices, device, p.batch, "compressed_indices");
  p.query = reinterpret_cast<const __nv_bfloat16*>(query.const_data_ptr<at::BFloat16>());
  p.swa_cache = swa_cache.const_data_ptr<std::uint8_t>();
  p.compressed_cache = compressed_cache.const_data_ptr<std::uint8_t>();
  p.swa_indices = swa_indices.const_data_ptr<int>();
  p.compressed_indices = compressed_indices.const_data_ptr<int>();
  std::vector<torch::Tensor> inputs{query, swa_cache, compressed_cache, swa_indices, compressed_indices};
  auto lengths = [&](const std::optional<torch::Tensor>& t, const char* name) -> const int* {
    if (!t) return nullptr;
    check(*t, device, torch::kInt32, name);
    TORCH_CHECK(t->dim() == 1 && t->size(0) == p.batch, name, " must have shape [B]");
    inputs.push_back(*t);
    return t->const_data_ptr<int>();
  };
  p.swa_lengths = lengths(swa_lengths, "swa_lengths");
  p.compressed_lengths = lengths(compressed_lengths, "compressed_lengths");
  if (sink) {
    check(*sink, device, torch::kFloat32, "sink");
    TORCH_CHECK(sink->dim() == 1 && sink->size(0) == 64, "sink must be FP32 [64]");
    p.sink = sink->const_data_ptr<float>();
    inputs.push_back(*sink);
  }
  p.chunks_per_cta = static_cast<int>(chunks_per_cta);
  p.softmax_scale = static_cast<float>(softmax_scale);
  p.lse_scale = static_cast<float>(lse_scale);

  torch::Tensor result = output ? *output : torch::empty(query.sizes(), query.options());
  check(result, device, torch::kBFloat16, "output");
  TORCH_CHECK(result.sizes() == query.sizes(), "output must have query's shape");
  torch::Tensor result_lse = lse ? *lse : torch::empty({p.batch, 64}, query.options().dtype(torch::kFloat32));
  check(result_lse, device, torch::kFloat32, "lse");
  TORCH_CHECK(result_lse.sizes() == torch::IntArrayRef({p.batch, 64}), "lse must have shape [B,64]");
  std::size_t required = sm120_nvfp4::sparse_mla_decode_workspace_size(
      p.batch, p.swa_candidates, p.compressed_candidates, p.chunks_per_cta);
  torch::Tensor scratch = workspace ? *workspace : torch::empty(
      {static_cast<std::int64_t>(required)}, query.options().dtype(torch::kUInt8));
  check(scratch, device, torch::kUInt8, "workspace");
  TORCH_CHECK(static_cast<std::size_t>(scratch.numel()) >= required, "workspace needs ", required, " bytes");
  std::vector<torch::Tensor> writes{result, result_lse, scratch};
  for (std::size_t i = 0; i < writes.size(); ++i) {
    for (const auto& input : inputs)
      TORCH_CHECK(!overlaps(writes[i], input), "output/lse/workspace must not overlap inputs");
    for (std::size_t j = 0; j < i; ++j)
      TORCH_CHECK(!overlaps(writes[i], writes[j]), "output/lse/workspace must not overlap each other");
  }
  p.output = reinterpret_cast<__nv_bfloat16*>(result.mutable_data_ptr<at::BFloat16>());
  p.lse = result_lse.mutable_data_ptr<float>();
  auto status = sm120_nvfp4::sparse_mla_decode_sm120(
      p, scratch.mutable_data_ptr(), required, at::cuda::getCurrentCUDAStream(device).stream());
  TORCH_CHECK(status == sm120_nvfp4::GemmStatus::kSuccess, "SM120 sparse MLA failed: ",
              sm120_nvfp4::gemm_status_string(status));
  C10_CUDA_CHECK(cudaGetLastError());
  return {result, result_lse};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sm120_nvfp4, m) {
  m.def("sparse_mla_decode(Tensor query, Tensor swa_cache, Tensor compressed_cache, "
        "Tensor swa_indices, Tensor compressed_indices, Tensor? swa_lengths, "
        "Tensor? compressed_lengths, Tensor? sink, int chunks_per_cta, "
        "float softmax_scale, float lse_scale, Tensor(a!)? output, Tensor(b!)? lse, "
        "Tensor(c!)? workspace) -> (Tensor(a!), Tensor(b!))");
  m.impl("sparse_mla_decode", torch::kCUDA, &sparse_mla_decode_torch);
}
