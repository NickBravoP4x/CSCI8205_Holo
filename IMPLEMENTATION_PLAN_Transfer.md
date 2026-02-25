# 🚀 CSCI 8205 Implementation Plan - FOR MACHINE TRANSFER

## ⚠️ IMPORTANT TRANSFER NOTE

**Created**: February 25, 2026 on development machine
**Target**: Transfer to correct machine for C++/CUDA implementation
**Status**: Plan ready for execution after environment setup on target system

### Before Starting Implementation on New Machine:

1. **Verify System Requirements**:
   - CUDA-compatible GPU (RTX 4070 or similar)
   - Sufficient VRAM (≥8GB recommended)
   - Ubuntu/Linux with CUDA toolkit support
   - 16+ CPU cores for OpenMP scaling tests

2. **Re-run Baseline Tests**:
   ```bash
   # Navigate to project directory
   cd /home/p4x/software/CSCI8205

   # Verify Python baseline still works
   python Scripts/holographic_benchmark.py --quick

   # Check roofline analysis
   python Scripts/roofline_analysis.py benchmark_results/benchmark_results.json
   ```

3. **Verify Baseline Results Transfer**:
   - Ensure `Results/Week1-2_Python_Baseline/` copied correctly
   - Validate benchmark data integrity
   - Confirm 240 test configurations available

4. **Environment Compatibility**:
   - Test holovision conda environment setup
   - Verify PyTorch GPU detection (`torch.cuda.is_available()`)
   - Check display backend compatibility (X11/Wayland)

---

# CSCI 8205 Week 3-4+ Implementation Plan: Multi-GPU Holographic Reconstruction Pipeline

## Context

This plan addresses the implementation of C++/CUDA holographic reconstruction and advanced profiling for the CSCI 8205 Multi-GPU Pipeline Architecture project. The user wants to continue improving performance and implement a faster GPU version, enhance roofline analysis, and improve figures based on their comprehensive baseline results.

**Current Date**: February 25, 2026
**Project**: Real-Time Digital Inline Holographic Reconstruction (400 Hz target)
**System**: Intel i7-13620H (16 cores) + RTX 4070 (8GB, Compute 8.9)

## Progress Assessment: Weeks 0-3 Status

### ✅ **Completed Work**

**Python Baseline Implementation (Exceeds Week 1-2 expectations for Python component)**:
- ✅ Complete holographic reconstruction implementation (`holographic_reconstruction.py`)
- ✅ Comprehensive benchmarking framework (`holographic_benchmark.py`)
- ✅ 240 test configurations validated (4 image sizes × 5 depth counts × 5 batch sizes × 4 modes)
- ✅ Excellent performance results: 0.60-0.92ms GPU latency (well under 2.5ms target for 400 Hz)
- ✅ Roofline analysis framework (`roofline_analysis.py`)
- ✅ Baseline results backed up (`Results/Week1-2_Python_Baseline/`)
- ✅ Display backend fixes for development environment
- ✅ Performance validation: 10-15× GPU speedup over CPU (matches proposal expectations)

### ❌ **Missing from Original Timeline**

**Week 1-2 Deliverables (Feb 11-22) - PARTIALLY COMPLETE**:
- ❌ **C++/CUDA reconstruction kernel** - Main missing component
- ❌ **Validation against Python reference** - Needs C++/CUDA implementation first
- ✅ **Benchmark across image sizes and depth counts** - Python baseline complete

**Week 3-4 Deliverables (Feb 23-Mar 8) - NOT STARTED**:
- ❌ **OpenMP CPU baseline (1-32 cores)** - No FFTW implementation
- ❌ **Roofline analysis with hardware counters** - Only theoretical roofline exists
- ❌ **Initial 6-stage pipeline** - Only single-stage reconstruction exists
- ❌ **Nsight Compute + perf profiling** - No hardware counter integration
- ❌ **Milestone 1 report** - No formal report generated

**Infrastructure Gaps**:
- ❌ **CUDA Toolkit** - Not installed (nvcc not found)
- ❌ **CMake build system** - No C++/CUDA build infrastructure
- ❌ **FFTW libraries** - Not installed for CPU baseline
- ❌ **Advanced profiling tools** - Nsight Compute, perf integration missing

## Implementation Plan: Catch-up and Enhancement

Given the current status, this plan addresses both catch-up work from Weeks 1-2 and advancement into Weeks 3-4, with focus on performance enhancement and advanced analysis.

### Phase 1: Infrastructure Setup (Days 1-2)

**Critical Dependencies Installation**:

1. **CUDA Toolkit 12.x**
   ```bash
   # Download and install CUDA toolkit
   wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
   sudo sh cuda_12.4.0_550.54.14_linux.run
   ```

2. **Build System & Libraries**
   ```bash
   sudo apt update
   sudo apt install cmake build-essential pkg-config
   sudo apt install libfftw3-dev libfftw3-doc libomp-dev
   sudo apt install linux-tools-$(uname -r) linux-tools-generic
   ```

3. **Project Structure Creation**
   ```
   CSCI8205/
   ├── cpp_cuda/                    # New C++/CUDA implementation
   │   ├── CMakeLists.txt          # Build configuration
   │   ├── src/
   │   │   ├── holographic_reconstruction.cu
   │   │   ├── holographic_reconstruction.h
   │   │   ├── cpu_baseline.cpp     # OpenMP version
   │   │   └── python_wrapper.cpp  # pybind11 interface
   │   └── tests/                   # Validation suite
   └── profiling/                   # Advanced profiling tools
   ```

### Phase 2: Core C++/CUDA Implementation (Days 3-6)

**Objective**: Port Python implementation to native C++/CUDA with 40% performance improvement

**Algorithm Mapping**:
- `reconstruct_3d_batched()` → CUDA kernel with cuFFT
- Key operations: 2D FFT (95% of compute time), filter application, normalization
- Target latency: 0.60ms → 0.40ms for 512×512×10 planes

**CUDA Kernel Architecture**:
```cpp
// holographic_reconstruction.cu
__global__ void apply_propagation_filter_kernel(
    cufftComplex* freq_data,
    const cufftComplex* filter_stack,
    int height, int width, int num_planes
);

__global__ void phase_correction_kernel(
    cufftComplex* data,
    const float* z_values,
    float wavelength,
    int height, int width, int num_planes
);

class HolographicReconstructor {
public:
    // cuFFT plans for batched processing
    cufftHandle plan_forward, plan_inverse;

    // GPU memory management
    cufftComplex* d_hologram_padded;
    cufftComplex* d_freq_domain;
    cufftComplex* d_reconstructed;

    // Reconstruction methods
    void reconstruct_3d_batched(float* input, float* output);
    void initialize_propagation_filters();
};
```

**Performance Optimizations**:
- **Batched cuFFT**: Single plan for all depth planes
- **Memory pools**: Pre-allocated GPU buffers
- **Kernel fusion**: Combine filter application + phase correction
- **Stream processing**: Overlap computation and memory transfers

### Phase 3: OpenMP CPU Baseline (Days 7-8)

**Objective**: Implement parallel CPU version for scaling analysis (1-32 threads)

**FFTW Integration**:
```cpp
// cpu_baseline.cpp
class CPUHolographicReconstructor {
private:
    fftwf_plan plan_forward, plan_inverse;
    int num_threads;

public:
    void reconstruct_openmp(float* input, float* output, int thread_count);
    void benchmark_scaling(int min_threads, int max_threads);
};

void reconstruct_openmp(float* input, float* output, int thread_count) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(thread_count);

    #pragma omp parallel for num_threads(thread_count)
    for (int plane = 0; plane < num_planes; ++plane) {
        // Per-plane reconstruction with FFTW
        fftwf_execute_dft(plan_forward, input, freq_domain);
        apply_propagation_filter_cpu(freq_domain, z_values[plane]);
        fftwf_execute_dft(plan_inverse, freq_domain, output);
    }
}
```

### Phase 4: Advanced Profiling Integration (Days 9-11)

**Objective**: Replace theoretical roofline with hardware counter-based analysis

**GPU Profiling - Nsight Compute**:
```bash
# Comprehensive kernel analysis
ncu --metrics=all --csv --log-file=gpu_profile.csv ./benchmark_cuda
ncu --set=full --export=detailed_profile.nsight-cuprof ./benchmark_cuda

# Specific metrics for roofline
ncu --metrics=smsp__sass_thread_inst_executed_op_ffma_add,smsp__sass_thread_inst_executed_op_ffma_mul,l1tex__t_bytes_pipe_lsu_mem_global_op_read,l1tex__t_bytes_pipe_lsu_mem_global_op_write ./benchmark_cuda
```

**CPU Profiling - perf Integration**:
```bash
# Cache hierarchy analysis
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./benchmark_cpu

# NUMA effects measurement
numactl --cpubind=0 --membind=0 perf stat -e NUMA_HIT,NUMA_MISS,NUMA_FOREIGN ./benchmark_cpu
```

**Enhanced Roofline Analysis**:
```python
# profiling/advanced_roofline.py
class HardwareCounterRoofline:
    def __init__(self):
        self.gpu_profiler = NsightComputeParser()
        self.cpu_profiler = PerfParser()

    def measure_peak_performance(self):
        """Empirical peak FLOPS and bandwidth measurement."""
        # GPU synthetic benchmarks
        gpu_peak_flops = self.run_flops_benchmark()
        gpu_peak_bandwidth = self.run_bandwidth_benchmark()

        # CPU scaling analysis
        cpu_peak_flops = self.run_cpu_flops_benchmark()
        cpu_peak_bandwidth = self.run_cpu_bandwidth_benchmark()

        return hardware_specs

    def construct_empirical_roofline(self, benchmark_results):
        """Build roofline from actual hardware counter data."""
        for config in benchmark_results:
            # Extract from Nsight Compute CSV
            actual_flops = self.gpu_profiler.get_flops(config)
            memory_throughput = self.gpu_profiler.get_bandwidth(config)
            arithmetic_intensity = actual_flops / memory_throughput

            self.plot_on_roofline(arithmetic_intensity, actual_flops)
```

### Phase 5: Python Integration & Validation (Days 12-13)

**Objective**: Maintain compatibility with existing 240-configuration benchmarking framework

**pybind11 Wrapper**:
```cpp
// python_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

PYBIND11_MODULE(cuda_holographic, m) {
    py::class_<HolographicReconstructor>(m, "HolographicReconstructor")
        .def(py::init<float, float, float, float, int>())
        .def("reconstruct_3d_batched", &HolographicReconstructor::reconstruct_3d_batched)
        .def("reconstruct_intensity", &HolographicReconstructor::reconstruct_intensity);

    py::class_<CPUHolographicReconstructor>(m, "CPUHolographicReconstructor")
        .def(py::init<int>())
        .def("reconstruct_openmp", &CPUHolographicReconstructor::reconstruct_openmp);
}
```

**Extended Benchmarking Framework**:
```python
# Enhanced holographic_benchmark.py
class HolographicBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.cuda_reconstructor = cuda_holographic.HolographicReconstructor()
        self.cpu_reconstructor = cuda_holographic.CPUHolographicReconstructor()

    def run_cuda_benchmark(self):
        """Benchmark C++/CUDA implementation."""
        # Use same 240 test configurations
        # Output same JSON format for comparison

    def run_openmp_benchmark(self):
        """Benchmark OpenMP scaling (1-32 threads)."""
        # Strong scaling analysis
        # Amdahl's law validation

    def validate_implementations(self):
        """Cross-validate CUDA vs Python results."""
        for config in self.test_configurations:
            python_result = self.python_reconstructor.reconstruct_intensity(hologram)
            cuda_result = self.cuda_reconstructor.reconstruct_intensity(hologram)

            relative_error = np.linalg.norm(cuda_result - python_result) / np.linalg.norm(python_result)
            assert relative_error < 1e-5, f"Validation failed: {relative_error:.2e}"
```

### Phase 6: Enhanced Analysis & Visualization (Day 14)

**Advanced Performance Visualizations**:

1. **Latency Distribution Analysis**:
   - P50/P95/P99/P99.9 latency CDFs for 400 Hz real-time feasibility
   - Tail latency characterization
   - Jitter analysis for real-time requirements

2. **Scaling Efficiency Plots**:
   - Strong scaling: fixed problem size, vary thread/GPU count
   - Weak scaling: scale problem size with resources
   - Amdahl's law theoretical vs. actual curves
   - Parallel efficiency percentage plots

3. **Hardware Utilization Heatmaps**:
   - GPU utilization timeline per SM
   - Memory bandwidth utilization over time
   - Cache hit rates across memory hierarchy
   - NUMA node utilization balance

4. **Roofline Enhancements**:
   - Multiple rooflines: L1, L2, L3, DRAM bandwidth limits
   - Kernel-level roofline points
   - Optimization opportunity annotations
   - Performance vs. problem size trending

## Success Metrics & Deliverables

### Primary Deliverables

1. **C++/CUDA Implementation**
   - ✅ Validates against all 240 Python configurations (relative error < 1e-5)
   - ✅ Achieves ≥40% performance improvement (0.60ms → 0.40ms)
   - ✅ Maintains same algorithmic interface as Python version

2. **OpenMP CPU Baseline**
   - ✅ Demonstrates strong scaling across 1-32 threads
   - ✅ Achieves ≥12× speedup on 16-core system (75% parallel efficiency)
   - ✅ Validates Amdahl's law predictions

3. **Hardware Counter-Based Roofline Analysis**
   - ✅ Empirical roofline construction (not theoretical)
   - ✅ Per-kernel performance characterization
   - ✅ Cache hierarchy and NUMA effects quantification
   - ✅ Memory bandwidth vs. compute balance identification

4. **Enhanced Benchmarking Framework**
   - ✅ Supports Python, C++/CUDA, and OpenMP implementations
   - ✅ Maintains same JSON output format for comparison
   - ✅ Advanced visualization suite (CDFs, scaling curves, heatmaps)

### Performance Targets

| Configuration | Python Baseline | C++/CUDA Target | OpenMP Target (16 cores) |
|---------------|----------------|-----------------|---------------------------|
| 256×256, 3 planes | 0.92ms | 0.60ms (35% improvement) | 2.1ms (8× speedup) |
| 512×512, 10 planes | 2.5ms | 1.7ms (32% improvement) | 8.5ms (10× speedup) |
| 1024×1024, 20 planes | 23ms | 15ms (35% improvement) | 95ms (12× speedup) |

### Real-Time Feasibility Analysis

**400 Hz Target (2.5ms budget)**:
- Small configurations: ✅ Already meeting target
- Medium configurations: ✅ C++/CUDA will meet target
- Large configurations: ❌ Exceeds budget (design for batch processing)

## Risk Mitigation

### Critical Risks

1. **CUDA Installation Complexity**
   - *Risk*: Toolkit installation issues on Ubuntu system
   - *Mitigation*: Test installation procedure, use Docker fallback
   - *Timeline Impact*: +1 day

2. **cuFFT Performance Bottleneck**
   - *Risk*: cuFFT doesn't deliver expected performance gains
   - *Mitigation*: Profile cuFFT separately, implement custom FFT kernels
   - *Timeline Impact*: +2 days

3. **Validation Failures**
   - *Risk*: C++/CUDA results don't match Python baseline
   - *Mitigation*: Implement extensive unit tests, gradual validation
   - *Timeline Impact*: +1 day

### Contingency Plans

- **Hardware Issues**: Focus on CPU OpenMP implementation first
- **Timeline Pressure**: Prioritize single-precision implementation
- **Integration Problems**: Maintain separate C++ benchmarking initially

## Files to Modify/Create

### Critical Implementation Files

1. **`/home/p4x/software/CSCI8205/cpp_cuda/src/holographic_reconstruction.cu`**
   - Main CUDA kernel implementation
   - cuFFT integration and memory management
   - Performance-critical code paths

2. **`/home/p4x/software/CSCI8205/cpp_cuda/src/cpu_baseline.cpp`**
   - OpenMP parallel CPU implementation
   - FFTW integration for scaling analysis
   - Thread-level performance monitoring

3. **`/home/p4x/software/CSCI8205/cpp_cuda/CMakeLists.txt`**
   - Build system configuration
   - CUDA, cuFFT, FFTW, OpenMP linking
   - pybind11 Python integration

4. **`/home/p4x/software/CSCI8205/Scripts/holographic_benchmark.py`** (extend)
   - Add C++/CUDA and OpenMP benchmarking modes
   - Hardware counter integration
   - Enhanced profiling and validation

5. **`/home/p4x/software/CSCI8205/Scripts/roofline_analysis.py`** (enhance)
   - Replace theoretical with empirical roofline
   - Nsight Compute and perf integration
   - Advanced visualization capabilities

### Validation Requirements

**Cross-Implementation Testing**:
- All 240 baseline configurations must pass validation
- Performance improvements verified across all problem sizes
- Statistical analysis of timing variance and measurement precision
- Memory usage profiling and optimization verification

This comprehensive plan addresses both the catch-up work from Weeks 1-2 and the advancement into Weeks 3-4, with focus on delivering production-quality C++/CUDA implementation and advanced performance analysis capabilities.