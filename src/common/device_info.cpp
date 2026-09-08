#include "common/device_info.hpp"

#include <cuda_runtime_api.h>

namespace sm120_nvfp4 {

int get_sm_count() {
  int device = 0;
  int value = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device);
  return value;
}

int get_sm_major_version() {
  int device = 0;
  int value = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, device);
  return value;
}

}  // namespace sm120_nvfp4
