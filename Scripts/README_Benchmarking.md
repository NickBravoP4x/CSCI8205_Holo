# Holographic Reconstruction Benchmarking Framework

Comprehensive performance analysis for holographic reconstruction across Python (PyTorch), CUDA C++ (cuFFT), and OpenMP C++ (FFTW) implementations.

Part of the CSCI 8205 Multi-GPU Pipeline Architecture project.

## Files

| File | Purpose |
|---|---|
| `holographic_reconstruction.py` | PyTorch reference implementation |
| `holographic_benchmark.py` | Multi-mode benchmark framework |
| `roofline_analysis.py` | Roofline model + enhanced visualizations |
| `validate_cuda.py` | Cross-validation: Python ↔ CUDA ↔ OpenMP |

## Benchmark Modes

| Mode | Backend | Notes |
|---|---|---|
| Single CPU | PyTorch CPU | One hologram at a time |
| Single GPU | PyTorch GPU | One hologram at a time, `cuda:0` |
| Batch CPU | PyTorch CPU | Batched via `generate_rgb_stack_images` |
| Batch GPU | PyTorch GPU | Batched on `cuda:0` |
| CUDA C++ | cuFFT via pybind11 | Requires `cuda_holographic` module |
| OpenMP N-threads | FFTW+OpenMP via pybind11 | Sweeps {1,2,4,8,16,32,64} threads |

## Quick Start

### Install dependencies

```bash
conda activate csci
pip install numpy torch matplotlib seaborn tqdm psutil GPUtil pybind11
```

### Run benchmarks

```bash
cd Scripts

# Quick benchmark (Python modes only)
python holographic_benchmark.py --quick

# Full benchmark (all 240 Python configs)
python holographic_benchmark.py

# Include C++/CUDA and OpenMP modes (requires build)
export PYTHONPATH=../cpp_cuda/build:$PYTHONPATH
python holographic_benchmark.py
```

### Run roofline analysis

```bash
# Generate all 7 plots from benchmark JSON
python roofline_analysis.py ../benchmark_results/benchmark_results.json

# With Nsight Compute empirical data overlay
python roofline_analysis.py ../benchmark_results/benchmark_results.json \
       --ncu-csv ../profiling/ncu_reports/ncu_512x512_d5.csv
```

### Cross-validate implementations

```bash
export PYTHONPATH=../cpp_cuda/build:$PYTHONPATH
python validate_cuda.py
```

Tests 20 configurations (4 image sizes × 5 depth counts), comparing Python (PyTorch CPU), CUDA C++, and OpenMP C++ outputs. Expected tolerances:
- CUDA ↔ OpenMP: < 1e-5 (same filter computation)
- C++ ↔ Python: < 1e-2 (different FFT libraries: cuFFT/FFTW vs PyTorch)

## Benchmark Configuration

### Full (default)

- **Image sizes**: 128×128, 256×256, 512×512, 1024×1024
- **Depth counts**: 1, 3, 5, 10, 20 planes
- **Batch sizes**: 1, 2, 4, 8, 16
- **Iterations**: 20 per configuration
- **Warmup**: 5 iterations

### Quick (`--quick`)

- **Image sizes**: 256×256, 512×512
- **Depth counts**: 3, 10
- **Batch sizes**: 1, 4
- **Iterations**: 5

## Performance Metrics

Each configuration records:

- **Latency**: mean, median, P95, P99, std_dev (milliseconds)
- **Throughput**: operations per second
- **Memory**: peak usage (MB)
- **System utilization**: CPU%, GPU%, GPU VRAM
- **Computational**: estimated FLOPS/sec, arithmetic intensity

## Output

### Benchmark results

`benchmark_results/benchmark_results.json` — Complete results with system info, configuration, and per-config metrics.

### Roofline analysis plots

| File | Description |
|---|---|
| `roofline_analysis.png` | CPU + GPU roofline with benchmark points |
| `multilevel_roofline.png` | L1/L2/L3/DRAM bandwidth ceilings, ncu overlay |
| `implementation_comparison.png` | Latency bars + speedup: Python vs CUDA vs OpenMP |
| `thread_scaling.png` | OpenMP speedup/efficiency vs threads, Amdahl's law |
| `latency_cdfs.png` | Per-mode CDF with 2.5 ms (400 Hz) budget line |
| `gpu_memory_utilization.png` | VRAM usage vs problem size |
| `scaling_analysis.png` | Image/depth/batch scaling + memory vs performance |
| `performance_report.txt` | Text summary of all modes |

## Hardware Specifications

Roofline analysis uses these hardware parameters (configured in `roofline_analysis.py`):

**CPU (AMD Threadripper PRO 7975WX)**:
- Peak FP32: 2.7 TFLOPS (32c × 2 FMA × 16 SP/AVX-512 × 5.3 GHz)
- DRAM bandwidth: 150 GB/s (8-channel DDR5-5200)
- L1d: 32 KB/core, L2: 1 MB/core, L3: 128 MB total

**GPU (NVIDIA RTX 5090)**:
- Peak FP32: 105 TFLOPS
- GDDR7 bandwidth: 1790 GB/s
- VRAM: 32 GB

## Building the C++/CUDA Module

The CUDA C++ and OpenMP benchmarks require building the `cuda_holographic` pybind11 module:

```bash
cd cpp_cuda
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
export PYTHONPATH=$(pwd):$PYTHONPATH
```

The benchmark framework gracefully skips C++ modes if the module is not available.

## Standalone Profiling

For hardware counter profiling without Python overhead:

```bash
# GPU: Nsight Compute kernel profiling
bash ../profiling/scripts/profile_gpu.sh 512 5

# CPU: perf cache hierarchy + thread scaling
bash ../profiling/scripts/profile_cpu.sh 512 5
```
