#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "sm120_nvfp4/gemm.hpp"

namespace {

struct Options {
  int m = 16;
  int n = 4096;
  int k = 8192;
  int device = 0;
  int warmup = 50;
  int iterations = 500;
  int heuristic_count = 16;
  std::size_t workspace_bytes = 64ull << 20;
};

const char* cublas_status_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "not supported";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "license error";
    default:
      return "unknown status";
  }
}

void check_cuda(cudaError_t status, const char* expression, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA error at line ") + std::to_string(line) + " for " +
        expression + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* expression, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string("cuBLASLt error at line ") + std::to_string(line) +
        " for " + expression + ": " + cublas_status_string(status) + " (" +
        std::to_string(static_cast<int>(status)) + ")");
  }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __LINE__)
#define CUBLAS_CHECK(expression) check_cublas((expression), #expression, __LINE__)

int parse_positive_int(const char* value, const char* name) {
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " must be a positive integer");
  }
  return static_cast<int>(parsed);
}

int parse_nonnegative_int(const char* value, const char* name) {
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        std::string(name) + " must be a non-negative integer");
  }
  return static_cast<int>(parsed);
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [M N K] [options]\n"
      << "\nOptions:\n"
      << "  --device D          CUDA device index (default: 0)\n"
      << "  --warmup N          warmup launches (default: 50)\n"
      << "  --iterations N      measured launches (default: 500)\n"
      << "  --heuristics N      cuBLASLt candidates to request (default: 16)\n"
      << "  --workspace-mib N   cuBLASLt workspace limit (default: 64)\n"
      << "  --help              show this message\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  std::vector<int> dimensions;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    auto next_value = [&](const char* name) -> const char* {
      if (++i >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[i];
    };
    if (arg == "--device") {
      options.device = parse_nonnegative_int(next_value("--device"), "device");
    } else if (arg == "--warmup") {
      options.warmup = parse_nonnegative_int(next_value("--warmup"), "warmup");
    } else if (arg == "--iterations") {
      options.iterations =
          parse_positive_int(next_value("--iterations"), "iterations");
    } else if (arg == "--heuristics") {
      options.heuristic_count =
          parse_positive_int(next_value("--heuristics"), "heuristics");
    } else if (arg == "--workspace-mib") {
      int mib = parse_nonnegative_int(
          next_value("--workspace-mib"), "workspace-mib");
      options.workspace_bytes = static_cast<std::size_t>(mib) << 20;
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::invalid_argument("unknown option: " + arg);
    } else {
      dimensions.push_back(parse_positive_int(argv[i], "dimension"));
    }
  }
  if (!dimensions.empty() && dimensions.size() != 3) {
    throw std::invalid_argument("provide either no dimensions or exactly M N K");
  }
  if (dimensions.size() == 3) {
    options.m = dimensions[0];
    options.n = dimensions[1];
    options.k = dimensions[2];
  }
  if ((options.k % sm120_nvfp4::kInputAlignmentElements) != 0 ||
      (options.n % sm120_nvfp4::kOutputAlignmentElements) != 0) {
    throw std::invalid_argument("K must be a multiple of 32 and N a multiple of 8");
  }
  return options;
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) {
      CUDA_CHECK(cudaMalloc(&data_, bytes_));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  void* data() { return data_; }
  std::size_t size() const { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

class Event {
 public:
  Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() { cudaEventDestroy(event_); }
  operator cudaEvent_t() const { return event_; }

 private:
  cudaEvent_t event_{};
};

template <class Launch>
float time_launches(Launch&& launch, cudaStream_t stream, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Event start;
  Event stop;
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  return elapsed_ms / static_cast<float>(iterations);
}

struct LtDescriptors {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  ~LtDescriptors() {
    if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
    if (d != nullptr) cublasLtMatrixLayoutDestroy(d);
    if (c != nullptr) cublasLtMatrixLayoutDestroy(c);
    if (b != nullptr) cublasLtMatrixLayoutDestroy(b);
    if (a != nullptr) cublasLtMatrixLayoutDestroy(a);
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    if (handle != nullptr) cublasLtDestroy(handle);
  }
};

int algorithm_id(const cublasLtMatmulAlgo_t& algorithm) {
  int id = -1;
  std::size_t written = 0;
  cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
      &algorithm, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written);
  return status == CUBLAS_STATUS_SUCCESS ? id : -1;
}

double tflops(int m, int n, int k, float milliseconds) {
  double operations = 2.0 * static_cast<double>(m) * n * k;
  return operations / (static_cast<double>(milliseconds) * 1.0e9);
}

void fill_payload(std::vector<std::uint8_t>& values, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(0, 255);
  for (std::uint8_t& value : values) {
    value = static_cast<std::uint8_t>(distribution(generator));
  }
}

struct ValidationStats {
  std::size_t mismatches = 0;
  float max_absolute_error = 0.0f;
  float max_relative_error = 0.0f;
};

ValidationStats compare_outputs(
    const std::vector<half>& cute, const std::vector<half>& cublas,
    int m, int n) {
  ValidationStats stats;
  constexpr float absolute_tolerance = 0.5f;
  constexpr float relative_tolerance = 1.0e-3f;
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      float lhs = __half2float(
          cute[static_cast<std::size_t>(row) * n + column]);
      float rhs = __half2float(
          cublas[static_cast<std::size_t>(column) * m + row]);
      float absolute_error = std::fabs(lhs - rhs);
      float denominator = std::max({1.0f, std::fabs(lhs), std::fabs(rhs)});
      float relative_error = absolute_error / denominator;
      stats.max_absolute_error =
          std::max(stats.max_absolute_error, absolute_error);
      stats.max_relative_error =
          std::max(stats.max_relative_error, relative_error);
      if (!std::isfinite(lhs) || !std::isfinite(rhs) ||
          absolute_error > absolute_tolerance + relative_tolerance * denominator) {
        ++stats.mismatches;
        if (stats.mismatches <= 5) {
          std::cerr << "Mismatch at (" << row << ',' << column << "): cute="
                    << lhs << ", cuBLASLt=" << rhs << '\n';
        }
      }
    }
  }
  return stats;
}

ValidationStats compare_row_major_outputs(
    const std::vector<half>& lhs_values,
    const std::vector<half>& rhs_values, int m, int n) {
  ValidationStats stats;
  constexpr float absolute_tolerance = 0.5f;
  constexpr float relative_tolerance = 1.0e-3f;
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      std::size_t offset = static_cast<std::size_t>(row) * n + column;
      float lhs = __half2float(lhs_values[offset]);
      float rhs = __half2float(rhs_values[offset]);
      float absolute_error = std::fabs(lhs - rhs);
      float denominator = std::max({1.0f, std::fabs(lhs), std::fabs(rhs)});
      float relative_error = absolute_error / denominator;
      stats.max_absolute_error = std::max(stats.max_absolute_error, absolute_error);
      stats.max_relative_error = std::max(stats.max_relative_error, relative_error);
      if (!std::isfinite(lhs) || !std::isfinite(rhs) ||
          absolute_error > absolute_tolerance + relative_tolerance * denominator) {
        ++stats.mismatches;
      }
    }
  }
  return stats;
}

int run(const Options& options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));
  if (properties.major != 12 || properties.minor != 0) {
    throw std::runtime_error("the selected device is not SM120");
  }

  std::cout << "SM120 NVFP4 GEMM: Custom CuTe vs CUTLASS reference vs cuBLASLt\n"
            << "Device: " << properties.name << " (SM" << properties.major
            << properties.minor << ")\n"
            << "Problem: M=" << options.m << ", N=" << options.n
            << ", K=" << options.k << '\n'
            << "Timing: warmup=" << options.warmup
            << ", iterations=" << options.iterations
            << ", cuBLASLt heuristic candidates=" << options.heuristic_count
            << ", workspace_limit=" << (options.workspace_bytes >> 20)
            << " MiB\n";

  const std::size_t a_bytes =
      sm120_nvfp4::packed_fp4_bytes(options.m, options.k);
  const std::size_t b_bytes =
      sm120_nvfp4::packed_fp4_bytes(options.n, options.k);
  const std::size_t output_elements =
      static_cast<std::size_t>(options.m) * options.n;
  const std::size_t sfa_bytes =
      sm120_nvfp4::scale_a_elements(options.m, options.n, options.k);
  const std::size_t sfb_bytes =
      sm120_nvfp4::scale_b_elements(options.m, options.n, options.k);

  std::vector<std::uint8_t> host_a(a_bytes);
  std::vector<std::uint8_t> host_b(b_bytes);
  fill_payload(host_a, 0x120a);
  fill_payload(host_b, 0xc0b1a5u);
  // Raw UE4M3 0x38 is 1.0. Filling the padded physical layouts makes the
  // buffers valid for both CUTLASS Sm1xxBlockScaledConfig and cuBLASLt.
  std::vector<std::uint8_t> host_sfa(sfa_bytes, 0x38);
  std::vector<std::uint8_t> host_sfb(sfb_bytes, 0x38);

  DeviceBuffer device_a(a_bytes);
  DeviceBuffer device_b(b_bytes);
  DeviceBuffer device_sfa(sfa_bytes);
  DeviceBuffer device_sfb(sfb_bytes);
  DeviceBuffer cute_output(output_elements * sizeof(half));
  DeviceBuffer cutlass_output(output_elements * sizeof(half));
  DeviceBuffer cublas_output(output_elements * sizeof(half));
  DeviceBuffer cublas_workspace(options.workspace_bytes);

  std::size_t cutlass_workspace_bytes =
      sm120_nvfp4::nvfp4_cutlass_gemm_workspace_size_sm120(
          options.m, options.n, options.k);
  DeviceBuffer cutlass_workspace(cutlass_workspace_bytes);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  try {
    CUDA_CHECK(cudaMemcpyAsync(
        device_a.data(), host_a.data(), a_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_b.data(), host_b.data(), b_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_sfa.data(), host_sfa.data(), sfa_bytes,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_sfb.data(), host_sfb.data(), sfb_bytes,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(
        cute_output.data(), 0, cute_output.size(), stream));
    CUDA_CHECK(cudaMemsetAsync(
        cutlass_output.data(), 0, cutlass_output.size(), stream));
    CUDA_CHECK(cudaMemsetAsync(
        cublas_output.data(), 0, cublas_output.size(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    LtDescriptors lt;
    CUBLAS_CHECK(cublasLtCreate(&lt.handle));
    CUBLAS_CHECK(cublasLtMatmulDescCreate(
        &lt.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        lt.operation, CUBLASLT_MATMUL_DESC_TRANSA,
        &trans_a, sizeof(trans_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        lt.operation, CUBLASLT_MATMUL_DESC_TRANSB,
        &trans_b, sizeof(trans_b)));

    cublasLtMatmulMatrixScale_t scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        lt.operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
        &scale_mode, sizeof(scale_mode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        lt.operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
        &scale_mode, sizeof(scale_mode)));
    const void* a_scale_pointer = device_sfa.data();
    const void* b_scale_pointer = device_sfb.data();
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        lt.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &a_scale_pointer, sizeof(a_scale_pointer)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        lt.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &b_scale_pointer, sizeof(b_scale_pointer)));

    // Packed row-major [M,K] is column-major [K,M], so cuBLASLt's required
    // TN form consumes exactly the same A and B bytes as the cute kernel.
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &lt.a, CUDA_R_4F_E2M1, options.k, options.m, options.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &lt.b, CUDA_R_4F_E2M1, options.k, options.n, options.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &lt.c, CUDA_R_16F, options.m, options.n, options.m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &lt.d, CUDA_R_16F, options.m, options.n, options.m));

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&lt.preference));
    std::uint64_t workspace_limit = options.workspace_bytes;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit, sizeof(workspace_limit)));

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(
        static_cast<std::size_t>(options.heuristic_count));
    int heuristic_results = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt.handle, lt.operation, lt.a, lt.b, lt.c, lt.d, lt.preference,
        options.heuristic_count, heuristics.data(), &heuristic_results));
    if (heuristic_results == 0) {
      throw std::runtime_error("cuBLASLt returned no NVFP4 algorithm");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch_cublas = [&](const cublasLtMatmulAlgo_t* algorithm) {
      CUBLAS_CHECK(cublasLtMatmul(
          lt.handle, lt.operation, &alpha,
          device_a.data(), lt.a, device_b.data(), lt.b, &beta,
          cublas_output.data(), lt.c, cublas_output.data(), lt.d,
          algorithm, cublas_workspace.data(), cublas_workspace.size(), stream));
    };

    int selected_index = -1;
    float best_tune_ms = std::numeric_limits<float>::infinity();
    int tune_iterations = std::max(10, std::min(50, options.iterations / 4));
    for (int i = 0; i < heuristic_results; ++i) {
      if (heuristics[i].state != CUBLAS_STATUS_SUCCESS ||
          heuristics[i].workspaceSize > options.workspace_bytes) {
        continue;
      }
      float candidate_ms = time_launches(
          [&] { launch_cublas(&heuristics[i].algo); }, stream, 3,
          tune_iterations);
      if (candidate_ms < best_tune_ms) {
        best_tune_ms = candidate_ms;
        selected_index = i;
      }
    }
    if (selected_index < 0) {
      throw std::runtime_error("none of the cuBLASLt heuristic algorithms ran");
    }

    auto launch_cute = [&] {
      sm120_nvfp4::GemmStatus status = sm120_nvfp4::nvfp4_cute_gemm_sm120(
          options.m, options.n, options.k,
          device_a.data(), device_b.data(), device_sfa.data(),
          device_sfb.data(), static_cast<half*>(cute_output.data()),
          stream);
      if (status != sm120_nvfp4::GemmStatus::kSuccess) {
        throw std::runtime_error(
            std::string("cute GEMM failed: ") +
            sm120_nvfp4::gemm_status_string(status));
      }
    };

    auto launch_cutlass = [&] {
      sm120_nvfp4::GemmStatus status =
          sm120_nvfp4::nvfp4_cutlass_gemm_sm120(
              options.m, options.n, options.k, device_a.data(),
              device_b.data(), device_sfa.data(), device_sfb.data(),
              static_cast<half*>(cutlass_output.data()),
              cutlass_workspace.data(), cutlass_workspace.size(), stream);
      if (status != sm120_nvfp4::GemmStatus::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS reference GEMM failed: ") +
            sm120_nvfp4::gemm_status_string(status));
      }
    };

    float cute_ms = time_launches(
        launch_cute, stream, options.warmup, options.iterations);
    float cutlass_ms = time_launches(
        launch_cutlass, stream, options.warmup, options.iterations);
    float cublas_ms = time_launches(
        [&] { launch_cublas(&heuristics[selected_index].algo); }, stream,
        options.warmup, options.iterations);

    // Refresh both outputs once after timing before comparing their different
    // physical output layouts (cute row-major, cuBLASLt column-major).
    launch_cute();
    launch_cutlass();
    launch_cublas(&heuristics[selected_index].algo);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<half> host_cute(output_elements);
    std::vector<half> host_cutlass(output_elements);
    std::vector<half> host_cublas(output_elements);
    CUDA_CHECK(cudaMemcpy(
        host_cutlass.data(), cutlass_output.data(), cutlass_output.size(),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        host_cute.data(), cute_output.data(), cute_output.size(),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        host_cublas.data(), cublas_output.data(), cublas_output.size(),
        cudaMemcpyDeviceToHost));
    ValidationStats validation =
        compare_outputs(host_cute, host_cublas, options.m, options.n);
    ValidationStats cutlass_validation = compare_row_major_outputs(
        host_cute, host_cutlass, options.m, options.n);

    double cutlass_tflops =
        tflops(options.m, options.n, options.k, cutlass_ms);
    double cute_tflops =
        tflops(options.m, options.n, options.k, cute_ms);
    double cublas_tflops =
        tflops(options.m, options.n, options.k, cublas_ms);
    double percent_of_cublas = 100.0 * cute_tflops / cublas_tflops;
    double cutlass_percent_of_cublas =
        100.0 * cutlass_tflops / cublas_tflops;

    std::cout << std::fixed << std::setprecision(4)
              << "\nSelected cuBLASLt algorithm: id="
              << algorithm_id(heuristics[selected_index].algo)
              << ", heuristic_index=" << selected_index
              << ", workspace=" << heuristics[selected_index].workspaceSize
              << " bytes, returned_candidates=" << heuristic_results << '\n'
              << "CuTe vs cuBLASLt validation: mismatches=" << validation.mismatches
              << ", max_abs_error=" << validation.max_absolute_error
              << ", max_rel_error=" << validation.max_relative_error << '\n'
              << "CuTe vs CUTLASS validation: mismatches="
              << cutlass_validation.mismatches
              << ", max_abs_error=" << cutlass_validation.max_absolute_error
              << ", max_rel_error=" << cutlass_validation.max_relative_error
              << '\n'
              << "Custom CuTe: " << cute_ms << " ms, "
              << cute_tflops << " TFLOP/s\n"
              << "CUTLASS ref: " << cutlass_ms << " ms, "
              << cutlass_tflops << " TFLOP/s\n"
              << "cuBLASLt:     " << cublas_ms << " ms, "
              << cublas_tflops << " TFLOP/s\n"
              << "CuTe/cuBLASLt: " << percent_of_cublas
              << "% of cuBLASLt (" << (cublas_ms / cute_ms)
              << "x cuBLASLt throughput)\n"
              << "CUTLASS/cuBLASLt: " << cutlass_percent_of_cublas
              << "% (" << (cublas_ms / cutlass_ms)
              << "x cuBLASLt throughput)\n"
              << "CSV_HEADER,M,N,K,cute_ms,cute_tflops,cutlass_ms,cutlass_tflops,cutlass_percent_of_cublas,cublaslt_ms,"
                 "cublaslt_tflops,percent_of_cublas,algo_id,cute_vs_cublas_mismatches,cute_vs_cutlass_mismatches\n"
              << "CSV," << options.m << ',' << options.n << ',' << options.k
              << ',' << cute_ms << ',' << cute_tflops << ',' << cutlass_ms
              << ',' << cutlass_tflops << ',' << cutlass_percent_of_cublas
              << ',' << cublas_ms
              << ',' << cublas_tflops << ',' << percent_of_cublas << ','
              << algorithm_id(heuristics[selected_index].algo) << ','
              << validation.mismatches << ',' << cutlass_validation.mismatches << '\n';

    CUDA_CHECK(cudaStreamDestroy(stream));
    return validation.mismatches == 0 && cutlass_validation.mismatches == 0
               ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (...) {
    cudaStreamDestroy(stream);
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
