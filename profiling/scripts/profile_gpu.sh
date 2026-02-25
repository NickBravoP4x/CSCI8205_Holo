#!/bin/bash
# GPU profiling with Nsight Compute (ncu)
# Run from the cpp_cuda/build directory
#
# Usage: bash profile_gpu.sh [size] [depth]
#   default: size=256, depth=3

set -e

SIZE=${1:-256}
DEPTH=${2:-3}
BUILD_DIR="$(cd "$(dirname "$0")/../../cpp_cuda/build" && pwd)"
REPORT_DIR="$(cd "$(dirname "$0")/.." && pwd)/ncu_reports"

mkdir -p "$REPORT_DIR"

BENCHMARK="$BUILD_DIR/benchmark_cuda"
if [ ! -f "$BENCHMARK" ]; then
    echo "ERROR: benchmark_cuda not found at $BENCHMARK"
    echo "Build with: cd cpp_cuda/build && cmake .. && make -j\$(nproc)"
    exit 1
fi

REPORT_FILE="$REPORT_DIR/ncu_${SIZE}x${SIZE}_d${DEPTH}"

echo "=== Nsight Compute GPU Profiling ==="
echo "Image size: ${SIZE}x${SIZE}, Depth: ${DEPTH}"
echo "Report: ${REPORT_FILE}.ncu-rep"
echo ""

# Full profiling: all metrics for roofline analysis
echo "Running full kernel profiling..."
ncu --set full \
    --target-processes all \
    --export "${REPORT_FILE}" \
    --force-overwrite \
    "$BENCHMARK" --size "$SIZE" --depth "$DEPTH" --iterations 3 --warmup 2

echo ""
echo "Exporting CSV for roofline plotting..."
ncu --import "${REPORT_FILE}.ncu-rep" \
    --csv \
    --page raw \
    > "${REPORT_FILE}.csv" 2>/dev/null || true

# Also export a summary
ncu --import "${REPORT_FILE}.ncu-rep" \
    --csv \
    --page details \
    > "${REPORT_FILE}_details.csv" 2>/dev/null || true

echo ""
echo "=== Profiling Complete ==="
echo "Reports saved to:"
echo "  ${REPORT_FILE}.ncu-rep  (interactive report)"
echo "  ${REPORT_FILE}.csv      (raw metrics CSV)"
echo "  ${REPORT_FILE}_details.csv (details CSV)"
echo ""
echo "View interactively: ncu-ui ${REPORT_FILE}.ncu-rep"
