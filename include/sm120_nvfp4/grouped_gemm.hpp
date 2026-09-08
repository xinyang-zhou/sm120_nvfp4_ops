#ifndef SM120_NVFP4_GROUPED_GEMM_HPP_
#define SM120_NVFP4_GROUPED_GEMM_HPP_

#include <cuda_runtime_api.h>

namespace sm120_nvfp4 {
namespace grouped_gemm {

// x/w contain two E2M1 values per byte. x_scale/w_scale contain raw UE4M3
// bytes in CUTLASS Sm1xxBlockScaledConfig<16> physical layout.
// tmas_ptr is per-call scratch of at least (3 * num_group + 2) * 128 bytes;
// tiles_ptr and cu_tiles_ptr hold num_group and num_group + 1 ints. The
// update_tma argument is retained for ABI compatibility and is ignored because
// TensorMap base addresses and shapes must be refreshed for every invocation.
void group_gemm_nvfp4_async(
    void *y_ptr, const void *x_ptr, const void *w_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *x_scale_ptr, const void *w_scale_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    int num_group, int m, int n, int k, int m_scale_pad,
    int num_seq_per_group_avg, bool update_tma, cudaStream_t stream);

}  // namespace grouped_gemm
}  // namespace sm120_nvfp4

#endif  // SM120_NVFP4_GROUPED_GEMM_HPP_
