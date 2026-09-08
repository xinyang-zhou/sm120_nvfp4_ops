#include "sm120_nvfp4/gemm.hpp"

namespace sm120_nvfp4 {

const char* gemm_status_string(GemmStatus status) {
  switch (status) {
    case GemmStatus::kSuccess:
      return "success";
    case GemmStatus::kInvalidArgument:
      return "invalid argument or unsupported alignment";
    case GemmStatus::kUnsupportedDevice:
      return "current device is not SM120";
    case GemmStatus::kInsufficientWorkspace:
      return "workspace is null or too small";
    case GemmStatus::kCutlassNotSupported:
      return "CUTLASS cannot implement this problem shape/alignment";
    case GemmStatus::kCutlassError:
      return "CUTLASS initialize or launch failed";
    case GemmStatus::kCudaError:
      return "CUDA runtime error";
  }
  return "unknown status";
}

}  // namespace sm120_nvfp4
