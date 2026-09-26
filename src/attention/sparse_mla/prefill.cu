#include "sm120_nvfp4/sparse_mla.hpp"

#include <limits>

namespace sm120_nvfp4 {

GemmStatus sparse_mla_prefill_sm120(const SparseMlaPrefillParams& params,
                                  cudaStream_t stream) {
  // Reuse exactly the decode numerical loop, with one complete candidate
  // stream per query. Explicitly force the direct-output schedule even for
  // T<=64, where decode's default would otherwise introduce split rounding.
  SparseMlaDecodeParams launch;
  static_cast<SparseMlaCommonParams&>(launch) = params;
  launch.batch = params.num_queries;
  launch.chunks_per_cta = std::numeric_limits<int>::max();
  return sparse_mla_decode_sm120(launch, nullptr, 0, stream);
}

}  // namespace sm120_nvfp4
