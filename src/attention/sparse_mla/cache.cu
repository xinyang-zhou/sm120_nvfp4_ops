#include "sm120_nvfp4/sparse_mla.hpp"

#include <limits>
#include <cuda_runtime.h>
#include "quantization.cuh"

namespace sm120_nvfp4 {
namespace {

__global__ void pack_cache_kernel(const __nv_bfloat16* values, const int* slots,
                                  std::uint8_t* cache, int pages) {
  int source_row = blockIdx.x, group = threadIdx.x;
  int slot = slots[source_row];
  if (slot < 0 || static_cast<std::int64_t>(slot) >= static_cast<std::int64_t>(pages) * 64) return;
  auto* page = cache + static_cast<std::int64_t>(slot / 64) * (64 * 384);
  auto* data = page + (slot % 64) * 352;
  auto* scales = page + 64 * 352 + (slot % 64) * 32;
  const auto* input = values + static_cast<std::int64_t>(source_row) * 512;
  if (group < 28) {
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = __bfloat162float(input[group * 16 + i]);
    sparse_mla::quantize16(x, data + group * 8, scales + group);
  } else {
    scales[group] = 0;
  }
  auto* rope = reinterpret_cast<__nv_bfloat16*>(data + 224);
  rope[group] = input[448 + group];
  rope[group + 32] = input[480 + group];
}

}  // namespace

GemmStatus sparse_mla_pack_cache_sm120(int rows, const __nv_bfloat16* values,
    const std::int32_t* slots, std::uint8_t* cache, int pages, cudaStream_t stream) {
  if (rows < 0 || rows > 1048576 || pages < 0 || pages > std::numeric_limits<int>::max() / 64 ||
      (rows && (!values || !slots)) || (pages && !cache) ||
      reinterpret_cast<std::uintptr_t>(values) % 2 ||
      reinterpret_cast<std::uintptr_t>(slots) % 4 ||
      reinterpret_cast<std::uintptr_t>(cache) % 16) return GemmStatus::kInvalidArgument;
  if (!rows) return GemmStatus::kSuccess;
  int device;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    return GemmStatus::kCudaError;
  if (properties.major != 12 || properties.minor != 0) return GemmStatus::kUnsupportedDevice;
  pack_cache_kernel<<<rows, 32, 0, stream>>>(values, slots, cache, pages);
  return cudaPeekAtLastError() == cudaSuccess ? GemmStatus::kSuccess : GemmStatus::kCudaError;
}

}  // namespace sm120_nvfp4
