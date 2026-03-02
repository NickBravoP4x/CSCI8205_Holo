#!/usr/bin/env bash
# Run C++/CUDA and OpenMP reconstruction benchmarks across all configurations.
#
# Usage:
#   bash Scripts/run_cpp_benchmarks.sh          # run from project root
#   bash Scripts/run_cpp_benchmarks.sh --quick  # fewer iterations for quick check

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/cpp_cuda/build"
RESULTS_DIR="$PROJECT_ROOT/Results/Reconstruction_Benchmark"

# Check binaries exist
if [[ ! -x "$BUILD_DIR/benchmark_cuda" ]] || [[ ! -x "$BUILD_DIR/benchmark_cpu" ]]; then
    echo "ERROR: Benchmarks not built. Run:"
    echo "  cd cpp_cuda/build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# Quick mode: fewer iterations
CUDA_ITERS=50
CUDA_WARMUP=10
CPU_ITERS=20
CPU_WARMUP=5
if [[ "${1:-}" == "--quick" ]]; then
    CUDA_ITERS=10
    CUDA_WARMUP=3
    CPU_ITERS=5
    CPU_WARMUP=2
    echo "=== QUICK MODE ==="
fi

SIZES=(128 256 512 1024)
DEPTHS=(1 3 5 10 20)

mkdir -p "$RESULTS_DIR"

# ── CUDA Benchmarks ──────────────────────────────────────────────────────────
CUDA_OUT="$RESULTS_DIR/cuda_benchmark_full.txt"
echo "Running CUDA benchmarks → $CUDA_OUT"
echo "CUDA Reconstruction Benchmarks — $(date)" > "$CUDA_OUT"
echo "Iterations: $CUDA_ITERS, Warmup: $CUDA_WARMUP" >> "$CUDA_OUT"
echo "" >> "$CUDA_OUT"

for size in "${SIZES[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        echo "  CUDA ${size}x${size} x ${depth} planes..."
        echo "=== CUDA ${size}x${size} x ${depth} planes ===" >> "$CUDA_OUT"
        "$BUILD_DIR/benchmark_cuda" \
            --size "$size" --depth "$depth" \
            --iterations "$CUDA_ITERS" --warmup "$CUDA_WARMUP" \
            >> "$CUDA_OUT" 2>&1
        echo "" >> "$CUDA_OUT"
    done
done
echo "CUDA benchmarks done."

# ── OpenMP CPU Benchmarks (thread scaling sweep) ─────────────────────────────
CPU_OUT="$RESULTS_DIR/cpu_benchmark_full.txt"
echo "Running OpenMP benchmarks → $CPU_OUT"
echo "OpenMP CPU Reconstruction Benchmarks — $(date)" > "$CPU_OUT"
echo "Iterations: $CPU_ITERS, Warmup: $CPU_WARMUP" >> "$CPU_OUT"
echo "" >> "$CPU_OUT"

for size in "${SIZES[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        echo "  CPU ${size}x${size} x ${depth} planes (thread sweep)..."
        echo "=== CPU ${size}x${size} x ${depth} planes (thread sweep) ===" >> "$CPU_OUT"
        "$BUILD_DIR/benchmark_cpu" \
            --size "$size" --depth "$depth" \
            --iterations "$CPU_ITERS" --warmup "$CPU_WARMUP" \
            --sweep \
            >> "$CPU_OUT" 2>&1
        echo "" >> "$CPU_OUT"
    done
done
echo "OpenMP benchmarks done."

echo ""
echo "Results saved to:"
echo "  $CUDA_OUT"
echo "  $CPU_OUT"
