#!/bin/bash
# CPU profiling with perf for cache hierarchy analysis and thread scaling
#
# Usage: bash profile_cpu.sh [size] [depth]
#   default: size=256, depth=3

set -e

SIZE=${1:-256}
DEPTH=${2:-3}
BUILD_DIR="$(cd "$(dirname "$0")/../../cpp_cuda/build" && pwd)"
REPORT_DIR="$(cd "$(dirname "$0")/.." && pwd)/perf_reports"

mkdir -p "$REPORT_DIR"

BENCHMARK="$BUILD_DIR/benchmark_cpu"
if [ ! -f "$BENCHMARK" ]; then
    echo "ERROR: benchmark_cpu not found at $BENCHMARK"
    exit 1
fi

echo "=== CPU Performance Profiling ==="
echo "Image size: ${SIZE}x${SIZE}, Depth: ${DEPTH}"
echo ""

# --- Single-thread cache analysis ---
echo "--- Cache Hierarchy Analysis (1 thread) ---"
PERF_OUT="$REPORT_DIR/perf_cache_${SIZE}x${SIZE}_d${DEPTH}.txt"
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,instructions,cycles,task-clock \
    "$BENCHMARK" --size "$SIZE" --depth "$DEPTH" --threads 1 --iterations 10 --warmup 3 \
    2>"$PERF_OUT" | tail -5

echo "Saved: $PERF_OUT"
echo ""

# --- Thread scaling sweep ---
echo "--- Thread Scaling Sweep ---"
SWEEP_OUT="$REPORT_DIR/perf_sweep_${SIZE}x${SIZE}_d${DEPTH}.txt"

echo "# Thread scaling: ${SIZE}x${SIZE}, depth=${DEPTH}" > "$SWEEP_OUT"
echo "# threads  mean_ms  throughput_ops  cache_misses  instructions  IPC" >> "$SWEEP_OUT"

for THREADS in 1 2 4 8 16 32 64; do
    echo -n "  Threads=$THREADS: "

    # Run perf stat and capture output
    PERF_TMP="$REPORT_DIR/perf_tmp_t${THREADS}.txt"
    perf stat -e cache-misses,instructions,cycles \
        "$BENCHMARK" --size "$SIZE" --depth "$DEPTH" --threads "$THREADS" --iterations 10 --warmup 3 \
        2>"$PERF_TMP" | grep "Threads:" | head -1

    # Extract metrics from perf output
    CACHE_MISSES=$(grep "cache-misses" "$PERF_TMP" | awk '{print $1}' | tr -d ',')
    INSTRUCTIONS=$(grep "instructions" "$PERF_TMP" | awk '{print $1}' | tr -d ',')
    IPC=$(grep "instructions" "$PERF_TMP" | grep -oP '\d+\.\d+ +insn' | awk '{print $1}')

    echo "$THREADS  $CACHE_MISSES  $INSTRUCTIONS  ${IPC:-N/A}" >> "$SWEEP_OUT"
    rm -f "$PERF_TMP"
done

echo "Saved: $SWEEP_OUT"
echo ""

echo "=== CPU Profiling Complete ==="
echo "Reports in: $REPORT_DIR/"
