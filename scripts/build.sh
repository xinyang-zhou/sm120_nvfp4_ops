#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
PYTHON_BIN="${PYTHON:-python3}"
CMAKE_BIN="${CMAKE:-cmake}"
CUDA_ROOT="${CUDA_ROOT:-${CUDA_HOME:-/usr/local/cuda}}"
HOST_TOOLCHAIN_DIR="${HOST_TOOLCHAIN_DIR:-/usr/bin}"
export CUDA_HOME="${CUDA_ROOT}"

if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  echo "nvcc was not found below CUDA_ROOT=${CUDA_ROOT}" >&2
  exit 1
fi

if [[ -z "${CUTLASS_ROOT:-}" ]]; then
  CUTLASS_ROOT="$(${PYTHON_BIN} - <<'PY'
from pathlib import Path
try:
    import flashinfer
except ImportError:
    raise SystemExit(1)
root = Path(flashinfer.__file__).resolve().parent / "data" / "cutlass"
print(root)
PY
)" || {
    echo "Set CUTLASS_ROOT to a CUTLASS 4.2+ checkout." >&2
    exit 1
  }
fi

TORCH_ROOT="$(${PYTHON_BIN} -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')"
TORCH_CXX11_ABI="$(${PYTHON_BIN} -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"

"${CMAKE_BIN}" -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
  -DCMAKE_CXX_COMPILER="${CXX:-/usr/bin/g++}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX:-/usr/bin/g++}" \
  -DCMAKE_CUDA_FLAGS="-Xcompiler=-B${HOST_TOOLCHAIN_DIR}" \
  -DCUTLASS_ROOT="${CUTLASS_ROOT}" \
  -DTORCH_ROOT="${TORCH_ROOT}" \
  -DTORCH_CXX11_ABI="${TORCH_CXX11_ABI}" \
  -DSM120_NVFP4_BUILD_TORCH="${SM120_NVFP4_BUILD_TORCH:-ON}" \
  -DSM120_NVFP4_BUILD_TESTS="${SM120_NVFP4_BUILD_TESTS:-ON}" \
  -DSM120_NVFP4_BUILD_BENCHMARKS="${SM120_NVFP4_BUILD_BENCHMARKS:-ON}"

"${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-4}"

echo "Built SM120 NVFP4 operators in ${BUILD_DIR}"
