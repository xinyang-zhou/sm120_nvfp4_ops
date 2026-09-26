cmake_minimum_required(VERSION 3.24)

project(sm120_nvfp4_ops VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR "SM120 NVFP4 kernels require CUDA 12.8 or newer")
endif()

# CMake does not model the architecture-specific 120a target reliably yet.
# Disable its default SM75 code generation and pass the exact gencode below.
set(CMAKE_CUDA_ARCHITECTURES OFF)

option(SM120_NVFP4_BUILD_TORCH "Build the PyTorch extension" ON)
option(SM120_NVFP4_BUILD_TESTS "Build C++ tests" ON)
option(SM120_NVFP4_BUILD_BENCHMARKS "Build benchmark executables" ON)

include(cmake/Dependencies.cmake)

function(sm120_nvfp4_configure_target target)
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${CUTLASS_ROOT}/include
      ${CUTLASS_ROOT}/tools/util/include)
  target_compile_options(${target}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20013,20015>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

add_library(sm120_nvfp4_common STATIC src/common/device_info.cpp)
target_include_directories(sm120_nvfp4_common PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(sm120_nvfp4_common PUBLIC CUDA::cudart)
set_target_properties(sm120_nvfp4_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(sm120_nvfp4_gemm STATIC
  src/gemm/cute_gemm.cu
  src/gemm/gemm_dispatch.cu
  src/gemm/specialized/m128_gemm.cu
  src/gemm/specialized/m256_gemm.cu
  src/gemm/status.cpp
  src/gemm/cutlass_reference.cu)
sm120_nvfp4_configure_target(sm120_nvfp4_gemm)
target_link_libraries(sm120_nvfp4_gemm PUBLIC CUDA::cudart)

add_library(sm120_nvfp4_grouped_gemm STATIC src/grouped_gemm/grouped_gemm.cu)
sm120_nvfp4_configure_target(sm120_nvfp4_grouped_gemm)
target_link_libraries(
  sm120_nvfp4_grouped_gemm PUBLIC sm120_nvfp4_common CUDA::cudart)

add_library(sm120_nvfp4_attention STATIC
  src/attention/sparse_mla/decode.cu)
sm120_nvfp4_configure_target(sm120_nvfp4_attention)
target_link_libraries(
  sm120_nvfp4_attention PUBLIC sm120_nvfp4_gemm CUDA::cudart)

add_library(sm120_nvfp4_fused_moe STATIC
  src/fused_moe/fused_moe.cu
  src/fused_moe/count_and_gather.cu
  src/fused_moe/activation.cu
  src/fused_moe/reduce.cu)
sm120_nvfp4_configure_target(sm120_nvfp4_fused_moe)
target_link_libraries(
  sm120_nvfp4_fused_moe PUBLIC sm120_nvfp4_grouped_gemm CUDA::cudart)

add_library(sm120_nvfp4 INTERFACE)
target_link_libraries(sm120_nvfp4 INTERFACE
  sm120_nvfp4_gemm sm120_nvfp4_grouped_gemm sm120_nvfp4_attention
  sm120_nvfp4_fused_moe)

if(SM120_NVFP4_BUILD_TORCH)
  set(TORCH_ROOT "$ENV{TORCH_ROOT}" CACHE PATH "PyTorch package root")
  set(TORCH_CXX11_ABI "" CACHE STRING "PyTorch CXX11 ABI value")
  include(cmake/TorchPython.cmake)
  sm120_nvfp4_discover_torch()
  if(NOT EXISTS "${TORCH_ROOT}/include/torch/library.h")
    message(FATAL_ERROR "TORCH_ROOT must point to the installed Python torch package root")
  endif()
  add_library(_C MODULE
    bindings/gemm_ops.cu
    bindings/sparse_mla_ops.cpp
    bindings/grouped_gemm_ops.cpp
    bindings/fused_moe_ops.cpp)
  sm120_nvfp4_configure_target(_C)
  target_include_directories(_C PRIVATE
    ${TORCH_ROOT}/include
    ${TORCH_ROOT}/include/torch/csrc/api/include)
  target_link_directories(_C PRIVATE ${TORCH_ROOT}/lib)
  target_link_libraries(_C PRIVATE sm120_nvfp4
    torch torch_cpu torch_cuda c10 c10_cuda)
  target_compile_definitions(_C PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
  set_target_properties(_C PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    BUILD_RPATH "${TORCH_ROOT}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/sm120_nvfp4")
  add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${PROJECT_SOURCE_DIR}/python/sm120_nvfp4/__init__.py
      ${CMAKE_BINARY_DIR}/python/sm120_nvfp4/__init__.py
    COMMENT "Syncing the Python package into the build tree")
  install(TARGETS _C LIBRARY DESTINATION sm120_nvfp4)
  install(DIRECTORY python/sm120_nvfp4/ DESTINATION sm120_nvfp4
    FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
endif()

include(CTest)
if(SM120_NVFP4_BUILD_TESTS)
  add_executable(test_sm120_nvfp4_gemm tests/cpp/test_gemm.cu)
  sm120_nvfp4_configure_target(test_sm120_nvfp4_gemm)
  target_link_libraries(test_sm120_nvfp4_gemm PRIVATE sm120_nvfp4_gemm)
  add_test(NAME sm120_nvfp4_gemm COMMAND test_sm120_nvfp4_gemm)
endif()

if(SM120_NVFP4_BUILD_BENCHMARKS)
  add_executable(
    benchmark_gemm_specialized benchmarks/benchmark_gemm_specialized.cu)
  sm120_nvfp4_configure_target(benchmark_gemm_specialized)
  target_link_libraries(
    benchmark_gemm_specialized PRIVATE sm120_nvfp4_gemm)

  add_executable(
    benchmark_gemm_vs_cublaslt benchmarks/benchmark_gemm_vs_cublaslt.cu)
  sm120_nvfp4_configure_target(benchmark_gemm_vs_cublaslt)
  target_link_libraries(
    benchmark_gemm_vs_cublaslt PRIVATE sm120_nvfp4_gemm CUDA::cublasLt)

  add_executable(
    benchmark_grouped_gemm_vs_cublaslt
    benchmarks/benchmark_grouped_gemm_vs_cublaslt.cu)
  sm120_nvfp4_configure_target(benchmark_grouped_gemm_vs_cublaslt)
  target_link_libraries(
    benchmark_grouped_gemm_vs_cublaslt
    PRIVATE sm120_nvfp4_grouped_gemm CUDA::cublasLt)

  add_executable(
    probe_cublaslt_grouped benchmarks/probe_cublaslt_grouped.cu)
  target_link_libraries(probe_cublaslt_grouped PRIVATE CUDA::cublasLt CUDA::cudart)
  target_compile_options(probe_cublaslt_grouped PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a>)
endif()

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS
  sm120_nvfp4_common
  sm120_nvfp4_gemm
  sm120_nvfp4_attention
  sm120_nvfp4_grouped_gemm
  sm120_nvfp4_fused_moe
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
