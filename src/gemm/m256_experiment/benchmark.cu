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

#include "gemm/m256_gemm.hpp"
#include "sm120_nvfp4/gemm.hpp"

namespace {

void check_cuda(cudaError_t status, const char* expression, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA error at line ") + std::to_string(line) +
        " for " + expression + ": " + cudaGetErrorString(status));
  }
}
#define CUDA_CHECK(expression) check_cuda((expression), #expression, __LINE__)

int parse_positive_int(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return static_cast<int>(value);
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
float time_launches(
    Launch&& launch, cudaStream_t stream, int warmup, int iterations) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    launch();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Event start;
  Event stop;
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  return elapsed_ms / static_cast<float>(iterations);
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
    const std::vector<half>& baseline,
    const std::vector<half>& splitk) {
  ValidationStats stats;
  constexpr float kAbsoluteTolerance = 0.5f;
  constexpr float kRelativeTolerance = 1.0e-3f;
  for (std::size_t index = 0; index < baseline.size(); ++index) {
    const float lhs = __half2float(baseline[index]);
    const float rhs = __half2float(splitk[index]);
    const float absolute_error = std::fabs(lhs - rhs);
    const float denominator =
        std::max({1.0f, std::fabs(lhs), std::fabs(rhs)});
    const float relative_error = absolute_error / denominator;
    stats.max_absolute_error =
        std::max(stats.max_absolute_error, absolute_error);
    stats.max_relative_error =
        std::max(stats.max_relative_error, relative_error);
    if (!std::isfinite(lhs) || !std::isfinite(rhs) ||
        absolute_error >
            kAbsoluteTolerance + kRelativeTolerance * denominator) {
      ++stats.mismatches;
    }
  }
  return stats;
}

double tflops(int m, int n, int k, float milliseconds) {
  const double operations = 2.0 * static_cast<double>(m) * n * k;
  return operations / (static_cast<double>(milliseconds) * 1.0e9);
}

void check_status(sm120_nvfp4::GemmStatus status, const char* path) {
  if (status != sm120_nvfp4::GemmStatus::kSuccess) {
    throw std::runtime_error(
        std::string(path) + " failed: " +
        sm120_nvfp4::gemm_status_string(status));
  }
}

int run(
    int m, int n, int k, int split_k,
    int warmup, int iterations) {
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 12 || properties.minor != 0) {
    throw std::runtime_error("the selected device is not SM120");
  }

  const std::size_t a_bytes = sm120_nvfp4::packed_fp4_bytes(m, k);
  const std::size_t b_bytes = sm120_nvfp4::packed_fp4_bytes(n, k);
  const std::size_t sfa_bytes = sm120_nvfp4::scale_a_elements(m, n, k);
  const std::size_t sfb_bytes = sm120_nvfp4::scale_b_elements(m, n, k);
  const std::size_t output_elements = static_cast<std::size_t>(m) * n;
  const std::size_t splitk_workspace_bytes =
      sm120_nvfp4::m256_experiment::workspace_size(
          1, m, n, k, split_k);
  if (splitk_workspace_bytes == 0) {
    throw std::invalid_argument("this path requires M=256 and split_k=2");
  }

  std::vector<std::uint8_t> host_a(a_bytes);
  std::vector<std::uint8_t> host_b(b_bytes);
  fill_payload(host_a, 0x120a);
  fill_payload(host_b, 0xc0b1a5u);
  std::vector<std::uint8_t> host_sfa(sfa_bytes, 0x38);
  std::vector<std::uint8_t> host_sfb(sfb_bytes, 0x38);

  DeviceBuffer device_a(a_bytes);
  DeviceBuffer device_b(b_bytes);
  DeviceBuffer device_sfa(sfa_bytes);
  DeviceBuffer device_sfb(sfb_bytes);
  DeviceBuffer baseline_output(output_elements * sizeof(half));
  DeviceBuffer splitk_output(output_elements * sizeof(half));
  DeviceBuffer splitk_workspace(splitk_workspace_bytes);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  try {
    CUDA_CHECK(cudaMemcpyAsync(
        device_a.data(), host_a.data(), a_bytes,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_b.data(), host_b.data(), b_bytes,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_sfa.data(), host_sfa.data(), sfa_bytes,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        device_sfb.data(), host_sfb.data(), sfb_bytes,
        cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto launch_baseline = [&] {
      check_status(
          sm120_nvfp4::nvfp4_cute_gemm_sm120(
              m, n, k, device_a.data(), device_b.data(),
              device_sfa.data(), device_sfb.data(),
              static_cast<half*>(baseline_output.data()), stream),
          "baseline");
    };
    auto launch_splitk = [&] {
      check_status(
          sm120_nvfp4::m256_experiment::launch(
              1, m, n, k, split_k,
              device_a.data(), device_b.data(),
              device_sfa.data(), device_sfb.data(),
              static_cast<half*>(splitk_output.data()),
              splitk_workspace.data(), splitk_workspace.size(), stream),
          "M=256 specialized path");
    };

    const float baseline_ms = time_launches(
        launch_baseline, stream, warmup, iterations);
    const float splitk_ms = time_launches(
        launch_splitk, stream, warmup, iterations);

    launch_baseline();
    launch_splitk();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<half> host_baseline(output_elements);
    std::vector<half> host_splitk(output_elements);
    CUDA_CHECK(cudaMemcpy(
        host_baseline.data(), baseline_output.data(), baseline_output.size(),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        host_splitk.data(), splitk_output.data(), splitk_output.size(),
        cudaMemcpyDeviceToHost));
    const ValidationStats validation =
        compare_outputs(host_baseline, host_splitk);

    std::cout << std::fixed << std::setprecision(6)
              << "Device: " << properties.name << '\n'
              << "Shape: M=" << m << ", N=" << n << ", K=" << k
              << ", split_k=" << split_k << '\n'
              << "Workspace: " << splitk_workspace_bytes << " bytes ("
              << static_cast<double>(splitk_workspace_bytes) / (1 << 20)
              << " MiB)\n"
              << "Validation: mismatches=" << validation.mismatches
              << ", max_abs_error=" << validation.max_absolute_error
              << ", max_rel_error=" << validation.max_relative_error << '\n'
              << "Baseline: " << baseline_ms << " ms, "
              << tflops(m, n, k, baseline_ms) << " TFLOP/s\n"
              << "M=256 specialized total: " << splitk_ms << " ms, "
              << tflops(m, n, k, splitk_ms) << " TFLOP/s\n"
              << "Speedup: " << baseline_ms / splitk_ms << "x\n";

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
    int m = 256;
    int n = 4096;
    int k = 8192;
    int split_k = 2;
    int warmup = 500;
    int iterations = 500;
    if (argc != 1 && argc != 7) {
      std::cerr << "Usage: " << argv[0]
                << " [M N K SPLIT_K WARMUP ITERATIONS]\n";
      return EXIT_FAILURE;
    }
    if (argc == 7) {
      m = parse_positive_int(argv[1], "M");
      n = parse_positive_int(argv[2], "N");
      k = parse_positive_int(argv[3], "K");
      split_k = parse_positive_int(argv[4], "SPLIT_K");
      warmup = parse_positive_int(argv[5], "WARMUP");
      iterations = parse_positive_int(argv[6], "ITERATIONS");
    }
    return run(m, n, k, split_k, warmup, iterations);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
