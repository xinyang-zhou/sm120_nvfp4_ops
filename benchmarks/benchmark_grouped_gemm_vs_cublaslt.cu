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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_io.hpp"
#include "sm120_nvfp4/gemm.hpp"
#include "sm120_nvfp4/grouped_gemm.hpp"

namespace {

struct Options {
  int groups = 4;
  int group_m = 16;
  int n = 4096;
  int k = 8192;
  int device = 0;
  int warmup = 50;
  int iterations = 500;
  int heuristic_count = 16;
  std::size_t workspace_bytes = 64ull << 20;
  std::string json_path;
  std::string csv_path;
  std::string command;
};

const char* cublas_status_string(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
    case CUBLAS_STATUS_LICENSE_ERROR: return "license error";
    default: return "unknown status";
  }
}

void check_cuda(cudaError_t status, const char* expression, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA error at line ") + std::to_string(line) +
        " for " + expression + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* expression, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string("cuBLASLt error at line ") + std::to_string(line) +
        " for " + expression + ": " + cublas_status_string(status) +
        " (" + std::to_string(static_cast<int>(status)) + ")");
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
      << "Usage: " << program << " [GROUPS M_PER_GROUP N K] [options]\n\n"
      << "The current benchmark uses equal rows per expert. It compares one\n"
      << "repository-owned persistent grouped launch with a loop of GROUPS\n"
      << "independent cuBLASLt GEMMs; it is not a native grouped-cuBLASLt baseline.\n\n"
      << "Options:\n"
      << "  --device D          CUDA device index (default: 0)\n"
      << "  --warmup N          warmup full-workload launches (default: 50)\n"
      << "  --iterations N      measured full-workload launches (default: 500)\n"
      << "  --heuristics N      cuBLASLt candidates to request (default: 16)\n"
      << "  --workspace-mib N   shared cuBLASLt workspace limit (default: 64)\n"
      << "  --json PATH         write one self-contained JSON result\n"
      << "  --csv PATH          append one flat result row to CSV\n"
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
    } else if (arg == "--json") {
      options.json_path = next_value("--json");
    } else if (arg == "--csv") {
      options.csv_path = next_value("--csv");
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::invalid_argument("unknown option: " + arg);
    } else {
      dimensions.push_back(parse_positive_int(argv[i], "dimension"));
    }
  }
  if (!dimensions.empty() && dimensions.size() != 4) {
    throw std::invalid_argument(
        "provide either no dimensions or exactly GROUPS M_PER_GROUP N K");
  }
  if (dimensions.size() == 4) {
    options.groups = dimensions[0];
    options.group_m = dimensions[1];
    options.n = dimensions[2];
    options.k = dimensions[3];
  }
  if (options.groups > 256) {
    throw std::invalid_argument("GROUPS must be at most 256");
  }
  if ((options.k % sm120_nvfp4::kInputAlignmentElements) != 0 ||
      (options.n % sm120_nvfp4::kOutputAlignmentElements) != 0) {
    throw std::invalid_argument("K must be a multiple of 32 and N a multiple of 8");
  }
  if (options.group_m > std::numeric_limits<int>::max() / options.groups) {
    throw std::invalid_argument("GROUPS * M_PER_GROUP exceeds int32 range");
  }
  return options;
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) CUDA_CHECK(cudaMalloc(&data_, bytes_));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
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
float time_launches(
    Launch&& launch, cudaStream_t stream, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Event start;
  Event stop;
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  return elapsed_ms / static_cast<float>(iterations);
}

struct LtDescriptors {
  cublasLtHandle_t handle = nullptr;
  std::vector<cublasLtMatmulDesc_t> operations;
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
    for (cublasLtMatmulDesc_t operation : operations) {
      if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    }
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

double tflops(int groups, int group_m, int n, int k, float milliseconds) {
  const double operations =
      2.0 * static_cast<double>(groups) * group_m * n * k;
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
    const std::vector<half>& grouped, const std::vector<half>& cublas,
    int groups, int group_m, int n) {
  ValidationStats stats;
  constexpr float absolute_tolerance = 0.5f;
  constexpr float relative_tolerance = 1.0e-3f;
  const std::size_t group_elements =
      static_cast<std::size_t>(group_m) * n;
  for (int group = 0; group < groups; ++group) {
    for (int row = 0; row < group_m; ++row) {
      for (int column = 0; column < n; ++column) {
        const std::size_t grouped_offset =
            (static_cast<std::size_t>(group) * group_m + row) * n + column;
        const std::size_t cublas_offset =
            static_cast<std::size_t>(group) * group_elements +
            static_cast<std::size_t>(column) * group_m + row;
        const float lhs = __half2float(grouped[grouped_offset]);
        const float rhs = __half2float(cublas[cublas_offset]);
        const float absolute_error = std::fabs(lhs - rhs);
        const float denominator = std::max({1.0f, std::fabs(lhs), std::fabs(rhs)});
        const float relative_error = absolute_error / denominator;
        stats.max_absolute_error =
            std::max(stats.max_absolute_error, absolute_error);
        stats.max_relative_error =
            std::max(stats.max_relative_error, relative_error);
        if (!std::isfinite(lhs) || !std::isfinite(rhs) ||
            absolute_error > absolute_tolerance + relative_tolerance * denominator) {
          ++stats.mismatches;
          if (stats.mismatches <= 5) {
            std::cerr << "Mismatch at group=" << group << ", row=" << row
                      << ", column=" << column << ": grouped=" << lhs
                      << ", cuBLASLt=" << rhs << '\n';
          }
        }
      }
    }
  }
  return stats;
}

void write_structured_results(
    const Options& options, const cudaDeviceProp& properties,
    int cuda_runtime_version, int cuda_driver_version,
    std::size_t cublaslt_version, int heuristic_results,
    int selected_index, int selected_algorithm_id,
    std::size_t selected_workspace_bytes, int tune_iterations,
    int m_scale_pad, float grouped_ms, float cublas_ms,
    const ValidationStats& validation) {
  using sm120_nvfp4_benchmark::append_csv_row;
  using sm120_nvfp4_benchmark::csv_escape;
  using sm120_nvfp4_benchmark::json_escape;
  using sm120_nvfp4_benchmark::json_number;
  using sm120_nvfp4_benchmark::utc_timestamp;
  using sm120_nvfp4_benchmark::write_text_file;

  const std::string timestamp = utc_timestamp();
  const bool passed = validation.mismatches == 0;
  const int total_m = options.groups * options.group_m;
  const double grouped_tflops = tflops(
      options.groups, options.group_m, options.n, options.k, grouped_ms);
  const double cublas_tflops = tflops(
      options.groups, options.group_m, options.n, options.k, cublas_ms);
  const double relative_throughput = cublas_ms / grouped_ms;

  if (!options.json_path.empty()) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(9)
         << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"benchmark\": \"grouped_gemm\",\n"
         << "  \"timestamp_utc\": \"" << timestamp << "\",\n"
         << "  \"status\": \"" << (passed ? "PASS" : "FAIL") << "\",\n"
         << "  \"command\": \"" << json_escape(options.command) << "\",\n"
         << "  \"environment\": {\n"
         << "    \"device_index\": " << options.device << ",\n"
         << "    \"device_name\": \"" << json_escape(properties.name) << "\",\n"
         << "    \"compute_capability\": \"" << properties.major << '.'
         << properties.minor << "\",\n"
         << "    \"sm_count\": " << properties.multiProcessorCount << ",\n"
         << "    \"cuda_runtime_version\": " << cuda_runtime_version << ",\n"
         << "    \"cuda_driver_api_version\": " << cuda_driver_version << ",\n"
         << "    \"cublaslt_version\": " << cublaslt_version << "\n"
         << "  },\n"
         << "  \"shape\": {\"groups\": " << options.groups
         << ", \"m_per_group\": " << options.group_m
         << ", \"total_m\": " << total_m << ", \"n\": " << options.n
         << ", \"k\": " << options.k
         << ", \"m_scale_pad\": " << m_scale_pad << "},\n"
         << "  \"measurement\": {\n"
         << "    \"warmup\": " << options.warmup << ",\n"
         << "    \"iterations\": " << options.iterations << ",\n"
         << "    \"timer\": \"CUDA events\",\n"
         << "    \"stream\": \"dedicated non-default stream\",\n"
         << "    \"input_distribution\": \"mt19937 uniform packed bytes; seeds 0x120a/0xc0b1a5; UE4M3 scales 0x38\",\n"
         << "    \"persistent_scope\": \"metadata setup plus one grouped compute launch\",\n"
         << "    \"baseline_scope\": \"one cuBLASLt GEMM launch per expert; no output concatenation\",\n"
         << "    \"baseline_uses_native_grouped_cublaslt\": false\n"
         << "  },\n"
         << "  \"cublaslt_search\": {\n"
         << "    \"requested_candidates\": " << options.heuristic_count << ",\n"
         << "    \"returned_candidates\": " << heuristic_results << ",\n"
         << "    \"selected_index\": " << selected_index << ",\n"
         << "    \"selected_algorithm_id\": " << selected_algorithm_id << ",\n"
         << "    \"tune_warmup\": 3,\n"
         << "    \"tune_iterations\": " << tune_iterations << ",\n"
         << "    \"workspace_limit_bytes\": " << options.workspace_bytes << ",\n"
         << "    \"selected_workspace_bytes\": "
         << selected_workspace_bytes << "\n"
         << "  },\n"
         << "  \"results\": {\n"
         << "    \"persistent_grouped\": {\"latency_ms\": "
         << json_number(grouped_ms) << ", \"tflops\": "
         << json_number(grouped_tflops) << "},\n"
         << "    \"cublaslt_loop\": {\"latency_ms\": "
         << json_number(cublas_ms) << ", \"tflops\": "
         << json_number(cublas_tflops) << "},\n"
         << "    \"relative_throughput\": "
         << json_number(relative_throughput) << "\n"
         << "  },\n"
         << "  \"correctness\": {\n"
         << "    \"absolute_tolerance\": 0.5,\n"
         << "    \"relative_tolerance\": 0.001,\n"
         << "    \"mismatches\": " << validation.mismatches << ",\n"
         << "    \"max_abs_error\": "
         << json_number(validation.max_absolute_error) << ",\n"
         << "    \"max_rel_error\": "
         << json_number(validation.max_relative_error) << "\n"
         << "  }\n"
         << "}\n";
    write_text_file(options.json_path, json.str());
  }

  if (!options.csv_path.empty()) {
    const std::string header =
        "timestamp_utc,benchmark,status,command,device_index,device_name,"
        "compute_capability,sm_count,cuda_runtime_version,cuda_driver_api_version,"
        "cublaslt_version,groups,m_per_group,total_m,n,k,m_scale_pad,warmup,"
        "iterations,timer,stream,input_distribution,baseline,"
        "baseline_uses_native_grouped_cublaslt,"
        "heuristics_requested,heuristics_returned,selected_index,algo_id,"
        "algo_workspace_bytes,workspace_limit_bytes,persistent_grouped_ms,"
        "persistent_grouped_tflops,cublaslt_loop_ms,cublaslt_loop_tflops,"
        "relative_throughput,mismatches,max_abs_error,max_rel_error";
    std::ostringstream row;
    row << std::fixed << std::setprecision(9)
        << timestamp << ",grouped_gemm," << (passed ? "PASS" : "FAIL") << ','
        << csv_escape(options.command) << ',' << options.device << ','
        << csv_escape(properties.name) << ',' << properties.major << '.'
        << properties.minor << ',' << properties.multiProcessorCount << ','
        << cuda_runtime_version << ',' << cuda_driver_version << ','
        << cublaslt_version << ',' << options.groups << ',' << options.group_m
        << ',' << total_m << ',' << options.n << ',' << options.k << ','
        << m_scale_pad << ',' << options.warmup << ',' << options.iterations
        << ",CUDA events,dedicated non-default stream,"
        << csv_escape("mt19937 uniform packed bytes; seeds 0x120a/0xc0b1a5; UE4M3 scales 0x38")
        << ',' << csv_escape("one cuBLASLt GEMM launch per expert; no output concatenation")
        << ",false," << options.heuristic_count << ',' << heuristic_results << ','
        << selected_index << ',' << selected_algorithm_id << ','
        << selected_workspace_bytes << ',' << options.workspace_bytes << ','
        << grouped_ms << ',' << grouped_tflops << ',' << cublas_ms << ','
        << cublas_tflops << ',' << relative_throughput << ','
        << validation.mismatches << ',' << validation.max_absolute_error << ','
        << validation.max_relative_error;
    append_csv_row(options.csv_path, header, row.str());
  }
}

int run(const Options& options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));
  if (properties.major != 12 || properties.minor != 0) {
    throw std::runtime_error("the selected device is not SM120");
  }
  int cuda_runtime_version = 0;
  int cuda_driver_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));
  const std::size_t cublaslt_version = cublasLtGetVersion();

  const int total_m = options.groups * options.group_m;
  const int m_scale_pad =
      sm120_nvfp4::align_up(options.group_m, sm120_nvfp4::kScaleMNAlignment);
  const std::size_t x_bytes =
      sm120_nvfp4::packed_fp4_bytes(total_m, options.k);
  const std::size_t weight_group_bytes =
      sm120_nvfp4::packed_fp4_bytes(options.n, options.k);
  const std::size_t weight_bytes =
      static_cast<std::size_t>(options.groups) * weight_group_bytes;
  const std::size_t output_group_elements =
      static_cast<std::size_t>(options.group_m) * options.n;
  const std::size_t output_elements =
      static_cast<std::size_t>(options.groups) * output_group_elements;
  const std::size_t sfa_group_elements =
      sm120_nvfp4::scale_a_elements(options.group_m, options.n, options.k);
  const std::size_t sfb_group_elements =
      sm120_nvfp4::scale_b_elements(options.group_m, options.n, options.k);
  const std::size_t tma_bytes =
      static_cast<std::size_t>(options.groups * 3 + 2) * 128;

  std::cout
      << "SM120 NVFP4 Grouped GEMM: persistent grouped vs cuBLASLt loop\n"
      << "Device: " << properties.name << " (SM" << properties.major
      << properties.minor << ")\n"
      << "Problem: groups=" << options.groups
      << ", M_per_group=" << options.group_m << ", total_M=" << total_m
      << ", N=" << options.n << ", K=" << options.k << '\n'
      << "Baseline: " << options.groups
      << " independent cuBLASLt launches; not native grouped cuBLASLt\n"
      << "Timing: warmup=" << options.warmup
      << ", iterations=" << options.iterations
      << ", cuBLASLt heuristic candidates=" << options.heuristic_count
      << ", workspace_limit=" << (options.workspace_bytes >> 20) << " MiB\n";

  std::vector<std::uint8_t> host_x(x_bytes);
  std::vector<std::uint8_t> host_weight(weight_bytes);
  fill_payload(host_x, 0x120a);
  fill_payload(host_weight, 0xc0b1a5u);
  std::vector<std::uint8_t> host_sfa(
      static_cast<std::size_t>(options.groups) * sfa_group_elements, 0x38);
  std::vector<std::uint8_t> host_sfb(
      static_cast<std::size_t>(options.groups) * sfb_group_elements, 0x38);
  std::vector<int> host_seqlens(options.groups, options.group_m);
  std::vector<int> host_cu_seqlens(options.groups + 1, 0);
  for (int group = 0; group < options.groups; ++group) {
    host_cu_seqlens[group + 1] = (group + 1) * options.group_m;
  }

  DeviceBuffer device_x(x_bytes);
  DeviceBuffer device_weight(weight_bytes);
  DeviceBuffer device_sfa(host_sfa.size());
  DeviceBuffer device_sfb(host_sfb.size());
  DeviceBuffer device_seqlens(host_seqlens.size() * sizeof(int));
  DeviceBuffer device_cu_seqlens(host_cu_seqlens.size() * sizeof(int));
  DeviceBuffer grouped_output(output_elements * sizeof(half));
  DeviceBuffer cublas_output(output_elements * sizeof(half));
  DeviceBuffer tma_descriptors(tma_bytes);
  DeviceBuffer tiles(static_cast<std::size_t>(options.groups) * sizeof(int));
  DeviceBuffer cu_tiles(
      static_cast<std::size_t>(options.groups + 1) * sizeof(int));
  DeviceBuffer cublas_workspace(options.workspace_bytes);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  try {
    CUDA_CHECK(cudaMemcpyAsync(
        device_x.data(), host_x.data(), host_x.size(),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_weight.data(), host_weight.data(), host_weight.size(),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_sfa.data(), host_sfa.data(), host_sfa.size(),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_sfb.data(), host_sfb.data(), host_sfb.size(),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_seqlens.data(), host_seqlens.data(), device_seqlens.size(),
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_cu_seqlens.data(), host_cu_seqlens.data(),
        device_cu_seqlens.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(
        grouped_output.data(), 0, grouped_output.size(), stream));
    CUDA_CHECK(cudaMemsetAsync(
        cublas_output.data(), 0, cublas_output.size(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    LtDescriptors lt;
    CUBLAS_CHECK(cublasLtCreate(&lt.handle));
    lt.operations.resize(options.groups, nullptr);
    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    for (int group = 0; group < options.groups; ++group) {
      CUBLAS_CHECK(cublasLtMatmulDescCreate(
          &lt.operations[group], CUBLAS_COMPUTE_32F, CUDA_R_32F));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
          lt.operations[group], CUBLASLT_MATMUL_DESC_TRANSA,
          &trans_a, sizeof(trans_a)));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
          lt.operations[group], CUBLASLT_MATMUL_DESC_TRANSB,
          &trans_b, sizeof(trans_b)));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
          lt.operations[group], CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
          &scale_mode, sizeof(scale_mode)));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
          lt.operations[group], CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
          &scale_mode, sizeof(scale_mode)));
      const void* a_scale_pointer =
          static_cast<const std::uint8_t*>(device_sfa.data()) +
          static_cast<std::size_t>(group) * sfa_group_elements;
      const void* b_scale_pointer =
          static_cast<const std::uint8_t*>(device_sfb.data()) +
          static_cast<std::size_t>(group) * sfb_group_elements;
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
          lt.operations[group], CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
          &a_scale_pointer, sizeof(a_scale_pointer)));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
          lt.operations[group], CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
          &b_scale_pointer, sizeof(b_scale_pointer)));
    }

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &lt.a, CUDA_R_4F_E2M1, options.k, options.group_m, options.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &lt.b, CUDA_R_4F_E2M1, options.k, options.n, options.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &lt.c, CUDA_R_16F, options.group_m, options.n, options.group_m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &lt.d, CUDA_R_16F, options.group_m, options.n, options.group_m));
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&lt.preference));
    const std::uint64_t workspace_limit = options.workspace_bytes;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit, sizeof(workspace_limit)));

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(
        static_cast<std::size_t>(options.heuristic_count));
    int heuristic_results = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt.handle, lt.operations[0], lt.a, lt.b, lt.c, lt.d, lt.preference,
        options.heuristic_count, heuristics.data(), &heuristic_results));
    if (heuristic_results == 0) {
      throw std::runtime_error("cuBLASLt returned no NVFP4 algorithm");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto launch_cublas_loop = [&](const cublasLtMatmulAlgo_t* algorithm) {
      for (int group = 0; group < options.groups; ++group) {
        const void* group_x =
            static_cast<const std::uint8_t*>(device_x.data()) +
            static_cast<std::size_t>(group) *
                sm120_nvfp4::packed_fp4_bytes(options.group_m, options.k);
        const void* group_weight =
            static_cast<const std::uint8_t*>(device_weight.data()) +
            static_cast<std::size_t>(group) * weight_group_bytes;
        void* group_output =
            static_cast<std::uint8_t*>(cublas_output.data()) +
            static_cast<std::size_t>(group) * output_group_elements * sizeof(half);
        CUBLAS_CHECK(cublasLtMatmul(
            lt.handle, lt.operations[group], &alpha,
            group_x, lt.a, group_weight, lt.b, &beta,
            group_output, lt.c, group_output, lt.d, algorithm,
            cublas_workspace.data(), cublas_workspace.size(), stream));
      }
    };

    int selected_index = -1;
    float best_tune_ms = std::numeric_limits<float>::infinity();
    const int tune_iterations =
        std::max(10, std::min(50, options.iterations / 4));
    for (int index = 0; index < heuristic_results; ++index) {
      if (heuristics[index].state != CUBLAS_STATUS_SUCCESS ||
          heuristics[index].workspaceSize > options.workspace_bytes) {
        continue;
      }
      const float candidate_ms = time_launches(
          [&] { launch_cublas_loop(&heuristics[index].algo); },
          stream, 3, tune_iterations);
      if (candidate_ms < best_tune_ms) {
        best_tune_ms = candidate_ms;
        selected_index = index;
      }
    }
    if (selected_index < 0) {
      throw std::runtime_error("none of the cuBLASLt heuristic algorithms ran");
    }

    auto launch_grouped = [&] {
      sm120_nvfp4::grouped_gemm::group_gemm_nvfp4_async(
          grouped_output.data(), device_x.data(), device_weight.data(),
          device_seqlens.data(), device_cu_seqlens.data(),
          device_sfa.data(), device_sfb.data(), tma_descriptors.data(),
          tiles.data(), cu_tiles.data(), options.groups, total_m,
          options.n, options.k, m_scale_pad, options.group_m, true, stream);
    };

    const float grouped_ms = time_launches(
        launch_grouped, stream, options.warmup, options.iterations);
    const float cublas_ms = time_launches(
        [&] { launch_cublas_loop(&heuristics[selected_index].algo); },
        stream, options.warmup, options.iterations);

    launch_grouped();
    launch_cublas_loop(&heuristics[selected_index].algo);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<half> host_grouped(output_elements);
    std::vector<half> host_cublas(output_elements);
    CUDA_CHECK(cudaMemcpy(
        host_grouped.data(), grouped_output.data(), grouped_output.size(),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        host_cublas.data(), cublas_output.data(), cublas_output.size(),
        cudaMemcpyDeviceToHost));
    const ValidationStats validation = compare_outputs(
        host_grouped, host_cublas, options.groups, options.group_m, options.n);

    const double grouped_tflops = tflops(
        options.groups, options.group_m, options.n, options.k, grouped_ms);
    const double cublas_tflops = tflops(
        options.groups, options.group_m, options.n, options.k, cublas_ms);
    const double relative_throughput = cublas_ms / grouped_ms;
    const int selected_algorithm_id =
        algorithm_id(heuristics[selected_index].algo);
    const std::size_t selected_workspace_bytes =
        heuristics[selected_index].workspaceSize;

    std::cout << std::fixed << std::setprecision(4)
              << "\nSelected cuBLASLt algorithm: id=" << selected_algorithm_id
              << ", heuristic_index=" << selected_index
              << ", workspace=" << selected_workspace_bytes
              << " bytes, returned_candidates=" << heuristic_results << '\n'
              << "Validation: mismatches=" << validation.mismatches
              << ", max_abs_error=" << validation.max_absolute_error
              << ", max_rel_error=" << validation.max_relative_error << '\n'
              << "Persistent grouped: " << grouped_ms << " ms, "
              << grouped_tflops << " TFLOP/s\n"
              << "cuBLASLt loop:     " << cublas_ms << " ms, "
              << cublas_tflops << " TFLOP/s\n"
              << "Relative throughput: " << relative_throughput << "x\n"
              << "CSV_HEADER,groups,m_per_group,total_m,N,K,grouped_ms,"
                 "grouped_tflops,cublaslt_loop_ms,cublaslt_loop_tflops,"
                 "relative_throughput,algo_id,mismatches\n"
              << "CSV," << options.groups << ',' << options.group_m << ','
              << total_m << ',' << options.n << ',' << options.k << ','
              << grouped_ms << ',' << grouped_tflops << ',' << cublas_ms << ','
              << cublas_tflops << ',' << relative_throughput << ','
              << selected_algorithm_id << ',' << validation.mismatches << '\n';

    write_structured_results(
        options, properties, cuda_runtime_version, cuda_driver_version,
        cublaslt_version, heuristic_results, selected_index,
        selected_algorithm_id, selected_workspace_bytes, tune_iterations,
        m_scale_pad, grouped_ms, cublas_ms, validation);
    if (!options.json_path.empty()) {
      std::cout << "JSON result: " << options.json_path << '\n';
    }
    if (!options.csv_path.empty()) {
      std::cout << "CSV result: " << options.csv_path << '\n';
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return validation.mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (...) {
    cudaStreamDestroy(stream);
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    options.command = sm120_nvfp4_benchmark::command_line(argc, argv);
    return run(options);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
