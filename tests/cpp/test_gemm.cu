#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "sm120_nvfp4/gemm.hpp"

namespace {

constexpr int kTestM = 16;
constexpr int kTestN = 4096;
constexpr int kTestK = 8192;
constexpr float kScaleA = 2.0f;
constexpr float kScaleB = 1.0f;
constexpr int kCheckRows = 2;
constexpr int kCheckCols = 8;

static_assert((kTestK % sm120_nvfp4::kInputAlignmentElements) == 0);
static_assert((kTestN % sm120_nvfp4::kOutputAlignmentElements) == 0);

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t status_ = (call);                                                    \
    if (status_ != cudaSuccess) {                                                    \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "          \
                << cudaGetErrorString(status_) << '\n';                             \
      std::exit(EXIT_FAILURE);                                                       \
    }                                                                               \
  } while (0)

std::uint8_t encode_e2m1(float value) {
  return static_cast<std::uint8_t>(
      cutlass::float_e2m1_t(value).raw() & 0x0f);
}

float decode_e2m1(std::uint8_t nibble) {
  auto bits = cutlass::float_e2m1_t::bitcast(
      static_cast<std::uint8_t>(nibble & 0x0f));
  return static_cast<float>(cutlass::float_e2m1_t(bits));
}

std::uint8_t encode_ue4m3(float value) {
  return static_cast<std::uint8_t>(
      cutlass::float_ue4m3_t(value).raw());
}

void set_packed(
    std::vector<std::uint8_t>& data, int row, int col,
    int stride_bytes, std::uint8_t nibble) {
  std::uint8_t& byte =
      data[static_cast<std::size_t>(row) * stride_bytes + (col >> 1)];
  if ((col & 1) == 0) {
    byte = static_cast<std::uint8_t>((byte & 0xf0) | (nibble & 0x0f));
  } else {
    byte = static_cast<std::uint8_t>(
        (byte & 0x0f) | ((nibble & 0x0f) << 4));
  }
}

float get_packed(
    const std::vector<std::uint8_t>& data, int row, int col,
    int stride_bytes) {
  std::uint8_t byte =
      data[static_cast<std::size_t>(row) * stride_bytes + (col >> 1)];
  return decode_e2m1((col & 1) ? (byte >> 4) : (byte & 0x0f));
}

float make_a_value(int m, int k) {
  return (((m + k) & 7) == 0) ? 1.0f : 0.0f;
}

float make_b_value(int n, int k) {
  return (((k + 3 * n) & 15) == 0) ? 1.0f : 0.0f;
}

}  // namespace

int main() {
  using sm120_nvfp4::GemmStatus;

  std::cout << "SM120 hardware NVFP4 Tensor Core GEMM test\n"
            << "Problem: A[" << kTestM << ',' << kTestK << "] @ B^T["
            << kTestK << ',' << kTestN << "]\n";

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  std::cout << "Device: " << props.name << " (SM" << props.major
            << props.minor << ")\n";
  if (props.major != 12 || props.minor != 0) {
    std::cerr << "This test requires an SM120 GPU.\n";
    return 1;
  }

  constexpr int a_stride_bytes = sm120_nvfp4::ceil_div(kTestK, 2);
  constexpr int b_stride_bytes = sm120_nvfp4::ceil_div(kTestK, 2);
  constexpr std::size_t a_bytes =
      sm120_nvfp4::packed_fp4_bytes(kTestM, kTestK);
  constexpr std::size_t b_bytes =
      sm120_nvfp4::packed_fp4_bytes(kTestN, kTestK);
  constexpr std::size_t c_elements =
      static_cast<std::size_t>(kTestM) * kTestN;
  constexpr std::size_t sfa_elements =
      sm120_nvfp4::scale_a_elements(kTestM, kTestN, kTestK);
  constexpr std::size_t sfb_elements =
      sm120_nvfp4::scale_b_elements(kTestM, kTestN, kTestK);

  std::vector<std::uint8_t> h_a(a_bytes, 0);
  std::vector<std::uint8_t> h_b(b_bytes, 0);
  std::vector<std::uint8_t> h_sfa(
      sfa_elements, encode_ue4m3(kScaleA));
  std::vector<std::uint8_t> h_sfb(
      sfb_elements, encode_ue4m3(kScaleB));
  std::vector<half> h_c(c_elements);

  for (int m = 0; m < kTestM; ++m) {
    for (int k = 0; k < kTestK; ++k) {
      set_packed(
          h_a, m, k, a_stride_bytes, encode_e2m1(make_a_value(m, k)));
    }
  }
  for (int n = 0; n < kTestN; ++n) {
    for (int k = 0; k < kTestK; ++k) {
      set_packed(
          h_b, n, k, b_stride_bytes, encode_e2m1(make_b_value(n, k)));
    }
  }

  std::uint8_t* d_a = nullptr;
  std::uint8_t* d_b = nullptr;
  std::uint8_t* d_sfa = nullptr;
  std::uint8_t* d_sfb = nullptr;
  half* d_c = nullptr;
  void* d_workspace = nullptr;
  cudaStream_t stream = nullptr;

  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), a_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), b_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sfa), sfa_elements));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sfb), sfb_elements));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c),
                        c_elements * sizeof(half)));

  std::size_t workspace_bytes =
      sm120_nvfp4::nvfp4_gemm_workspace_size_sm120(
          kTestM, kTestN, kTestK);
  if (workspace_bytes != 0) {
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
  }

  CUDA_CHECK(cudaMemcpyAsync(
      d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      d_sfa, h_sfa.data(), sfa_elements, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      d_sfb, h_sfb.data(), sfb_elements, cudaMemcpyHostToDevice, stream));

  GemmStatus status = sm120_nvfp4::nvfp4_gemm_sm120(
      kTestM, kTestN, kTestK,
      d_a, d_b, d_sfa, d_sfb, d_c,
      d_workspace, workspace_bytes, stream);
  if (status != GemmStatus::kSuccess) {
    std::cerr << "GEMM launch failed: "
              << sm120_nvfp4::gemm_status_string(status) << '\n';
    return 2;
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(
      h_c.data(), d_c, c_elements * sizeof(half), cudaMemcpyDeviceToHost));

  int failures = 0;
  float max_abs_error = 0.0f;
  int checked_rows = std::min(kTestM, kCheckRows);
  int checked_cols = std::min(kTestN, kCheckCols);
  for (int m = 0; m < checked_rows; ++m) {
    for (int n = 0; n < checked_cols; ++n) {
      float reference = 0.0f;
      for (int k = 0; k < kTestK; ++k) {
        float a = get_packed(h_a, m, k, a_stride_bytes) * kScaleA;
        float b = get_packed(h_b, n, k, b_stride_bytes) * kScaleB;
        reference += a * b;
      }
      float got =
          __half2float(h_c[static_cast<std::size_t>(m) * kTestN + n]);
      float error = std::fabs(got - reference);
      max_abs_error = std::max(max_abs_error, error);
      failures += error <= 0.5f ? 0 : 1;
    }
  }

  std::cout << std::fixed << std::setprecision(1)
            << "Checked " << (checked_rows * checked_cols)
            << " sampled outputs; max_abs_error="
            << max_abs_error << '\n'
            << "SFA/SFB uniform scales: " << kScaleA << " / "
            << kScaleB << " (scale application is part of the reference)\n"
            << "Workspace: " << workspace_bytes << " bytes\n";

  if (d_workspace != nullptr) {
    CUDA_CHECK(cudaFree(d_workspace));
  }
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_sfb));
  CUDA_CHECK(cudaFree(d_sfa));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaStreamDestroy(stream));

  if (failures != 0) {
    std::cerr << "Validation failed for " << failures << " outputs.\n";
    return 3;
  }
  std::cout << "Validation passed.\n";
  return 0;
}
