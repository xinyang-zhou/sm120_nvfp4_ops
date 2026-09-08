#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
PYTHON_BIN="${PYTHON:-python3}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${BUILD_DIR}/test_sm120_nvfp4_gemm"

PYTHONPATH="${BUILD_DIR}/python:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "${PYTHON_BIN}" -m unittest discover -v -s "${PROJECT_ROOT}/tests/python"
