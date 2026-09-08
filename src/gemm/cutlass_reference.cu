#include "sm120_nvfp4/gemm.hpp"

#include <cstddef>

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/version.h"

#if CUTLASS_VERSION < 420
#error "SM120 NVFP4 GEMM requires CUTLASS 4.2 or newer"
#endif

#if __CUDACC_VER_MAJOR__ < 12 || \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)
#error "SM120 NVFP4 GEMM requires CUDA 12.8 or newer"
#endif

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#error "Compile the CUTLASS reference target for compute_120a/sm_120a to enable SM120 block-scaled MMA"
#endif

namespace sm120_nvfp4 {
namespace detail {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = void;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int kAlignmentA = kInputAlignmentElements;
constexpr int kAlignmentB = kInputAlignmentElements;
constexpr int kAlignmentC = kOutputAlignmentElements;
constexpr int kAlignmentD = kOutputAlignmentElements;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, kAlignmentC,
        ElementD, LayoutD, kAlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA, kAlignmentA,
        ElementB, LayoutB, kAlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

Gemm::Arguments make_arguments(
    int m, int n, int k,
    const void* a, const void* b,
    const void* sfa, const void* sfb,
    half* c) {
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(
      make_shape(m, n, k, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(
      make_shape(m, n, k, 1));

  using Payload = cutlass::float_e2m1_t;
  using Scale = cutlass::float_ue4m3_t;

  return Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<const Payload*>(a), stride_a,
       reinterpret_cast<const Payload*>(b), stride_b,
       reinterpret_cast<const Scale*>(sfa), layout_sfa,
       reinterpret_cast<const Scale*>(sfb), layout_sfb},
      {{1.0f, 0.0f},
       nullptr, stride_c,
       reinterpret_cast<ElementD*>(c), stride_d}};
}

bool valid_shape(int m, int n, int k) {
  return m > 0 && n > 0 && k > 0 &&
         (k % kInputAlignmentElements) == 0 &&
         (n % kOutputAlignmentElements) == 0;
}

GemmStatus map_cutlass_status(cutlass::Status status, bool from_can_implement) {
  if (status == cutlass::Status::kSuccess) {
    return GemmStatus::kSuccess;
  }
  if (from_can_implement || status == cutlass::Status::kErrorInvalidProblem ||
      status == cutlass::Status::kErrorMisalignedOperand) {
    return GemmStatus::kCutlassNotSupported;
  }
  if (status == cutlass::Status::kErrorWorkspaceNull) {
    return GemmStatus::kInsufficientWorkspace;
  }
  return GemmStatus::kCutlassError;
}

}  // namespace detail

 std::size_t nvfp4_cutlass_gemm_workspace_size_sm120(int m, int n, int k) {
  if (!detail::valid_shape(m, n, k)) {
    return 0;
  }
  auto arguments = detail::make_arguments(
      m, n, k, nullptr, nullptr, nullptr, nullptr, nullptr);
  return detail::Gemm::get_workspace_size(arguments);
}

GemmStatus nvfp4_cutlass_gemm_sm120(
    int m, int n, int k,
    const void* a, const void* b,
    const void* sfa, const void* sfb,
    half* c,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream) {
  if (!detail::valid_shape(m, n, k) || a == nullptr || b == nullptr ||
      sfa == nullptr || sfb == nullptr || c == nullptr) {
    return GemmStatus::kInvalidArgument;
  }

  int device = 0;
  cudaDeviceProp props{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&props, device) != cudaSuccess) {
    return GemmStatus::kCudaError;
  }
  if (props.major != 12 || props.minor != 0) {
    return GemmStatus::kUnsupportedDevice;
  }

  auto arguments = detail::make_arguments(m, n, k, a, b, sfa, sfb, c);
  std::size_t required_workspace = detail::Gemm::get_workspace_size(arguments);
  if (required_workspace > workspace_bytes ||
      (required_workspace != 0 && workspace == nullptr)) {
    return GemmStatus::kInsufficientWorkspace;
  }

  detail::Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return detail::map_cutlass_status(status, true);
  }
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return detail::map_cutlass_status(status, false);
  }
  status = gemm.run(stream);
  return detail::map_cutlass_status(status, false);
}

}  // namespace sm120_nvfp4
