cmake_minimum_required(VERSION 3.24)

project(sm120_nvfp4_splitk_experiment LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR "SM120 NVFP4 kernels require CUDA 12.8 or newer")
endif()

set(CMAKE_CUDA_ARCHITECTURES OFF)
find_package(CUDAToolkit REQUIRED)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(CUTLASS_ROOT "" CACHE PATH "CUTLASS 4.2+ source root")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cute/tensor.hpp")
  message(FATAL_ERROR "Set CUTLASS_ROOT to a CUTLASS 4.2+ source tree")
endif()

add_executable(sm120_nvfp4_splitk_benchmark
  benchmark.cu
  ../splitk_gemm.cu
  ../cute_gemm.cu
  ../status.cpp)

target_include_directories(sm120_nvfp4_splitk_benchmark PRIVATE
  "${PROJECT_ROOT}/include"
  "${PROJECT_ROOT}/src"
  "${CUTLASS_ROOT}/include"
  "${CUTLASS_ROOT}/tools/util/include")

target_compile_options(sm120_nvfp4_splitk_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20013,20015>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)

target_link_libraries(sm120_nvfp4_splitk_benchmark PRIVATE CUDA::cudart)
