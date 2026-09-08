find_package(CUDAToolkit REQUIRED)

set(CUTLASS_ROOT "$ENV{CUTLASS_ROOT}" CACHE PATH "Path to CUTLASS 4.2+")

if(NOT CUTLASS_ROOT)
  set(_bundled_cutlass "${PROJECT_SOURCE_DIR}/third_party/cutlass")
  if(EXISTS "${_bundled_cutlass}/include/cutlass/version.h")
    set(CUTLASS_ROOT "${_bundled_cutlass}")
  endif()
endif()

if(NOT EXISTS
   "${CUTLASS_ROOT}/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp")
  message(FATAL_ERROR
    "CUTLASS_ROOT must point to CUTLASS 4.2+ with SM120 block-scaled support. "
    "Pass -DCUTLASS_ROOT=/path/to/cutlass or set CUTLASS_ROOT.")
endif()
