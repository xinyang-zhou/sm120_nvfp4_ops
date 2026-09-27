#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace sm120_nvfp4::sparse_mla {

__device__ __forceinline__ void quantize16(const float (&x)[16],
                                          std::uint8_t* dst,
                                          std::uint8_t* scale_dst) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
  __nv_fp8_e4m3 scale(amax / 6.f);
  *scale_dst = scale.__x;
  float sf = static_cast<float>(scale);
  float inv = sf == 0.f ? 0.f : 1.f / sf;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __nv_fp4x2_e2m1 pair(make_float2(x[2 * i] * inv, x[2 * i + 1] * inv));
    dst[i] = pair.__x;
  }
}

}  // namespace sm120_nvfp4::sparse_mla
