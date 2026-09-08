#ifndef SM120_FUSE_MOE_SCALE_LAYOUT_CUH_
#define SM120_FUSE_MOE_SCALE_LAYOUT_CUH_

#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

namespace sm120_nvfp4 {
namespace fused_moe {

constexpr int kNvfp4SFVectorSize = 16;
constexpr int kNvfp4ScaleMAlignment = 128;

CUTE_HOST_DEVICE constexpr int64_t ceil_div_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

CUTE_HOST_DEVICE constexpr int64_t scale_k_padded(int k) {
  return ceil_div_i64(ceil_div_i64(k, kNvfp4SFVectorSize), 4) * 4;
}

CUTE_HOST_DEVICE constexpr int64_t sfa_group_elements(int m_pad, int k) {
  return static_cast<int64_t>(m_pad) * scale_k_padded(k);
}

CUTE_HOST_DEVICE constexpr int align_up_128(int value) {
  return static_cast<int>(ceil_div_i64(value, kNvfp4ScaleMAlignment) *
                          kNvfp4ScaleMAlignment);
}

// Return the byte offset of the scale for logical (row, k_block * 16) in
// CUTLASS's K-major Sm1xxBlockScaledConfig<16> physical layout.
CUTE_HOST_DEVICE int64_t sfa_offset(
    int row, int k_block, int m_pad, int logical_k) {
  using ScaleConfig =
      cutlass::detail::Sm1xxBlockScaledConfig<kNvfp4SFVectorSize>;
  auto layout = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m_pad, 1, logical_k, 1));
  return static_cast<int64_t>(
      layout(cute::make_coord(row, k_block * kNvfp4SFVectorSize, 0)));
}

}  // namespace fused_moe
}  // namespace sm120_nvfp4

#endif  // SM120_FUSE_MOE_SCALE_LAYOUT_CUH_
