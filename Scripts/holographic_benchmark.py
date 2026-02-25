#!/usr/bin/env python3
"""
Comprehensive benchmarking framework for holographic reconstruction.
Designed for CSCI 8205 Multi-GPU Pipeline Architecture project.

Benchmarks single CPU, single GPU, batch CPU, and batch GPU modes with
comprehensive performance metrics for image size, depth scaling, memory
usage, roofline analysis, and NUMA effects.
"""

import sys
import os
import gc
import time
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import multiprocessing as mp

# Handle display backend issues - fallback from Wayland to X11 if needed
def setup_display_backend():
    """Setup display backend with Wayland to X11 fallback."""
    if os.environ.get('WAYLAND_DISPLAY'):
        os.environ.setdefault('QT_QPA_PLATFORM', 'wayland')
    elif os.environ.get('DISPLAY'):
        os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

setup_display_backend()

import numpy as np
import torch
import torch.multiprocessing as tmp
import psutil
import GPUtil
from tqdm import tqdm

# Import the holographic reconstruction module
try:
    from holographic_reconstruction import HolographicReconstructor, generate_rgb_stack_images
except ImportError:
    print("Error: Cannot import holographic_reconstruction.py")
    print("Make sure the file is in the same directory or in your Python path")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    image_sizes: List[Tuple[int, int]]
    depth_counts: List[int]
    batch_sizes: List[int]
    num_warmup: int = 5
    num_iterations: int = 20
    measure_memory: bool = True
    measure_numa: bool = True
    save_results: bool = True
    output_dir: str = "benchmark_results"

@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""
    mode: str
    image_size: Tuple[int, int]
    depth_count: int
    batch_size: int

    # Timing metrics (milliseconds)
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    std_latency: float
    throughput: float  # operations per second

    # Memory metrics (MB)
    peak_memory_used: float
    memory_efficiency: float  # actual / theoretical minimum

    # System metrics
    cpu_utilization: float
    gpu_utilization: Optional[float]
    gpu_memory_used: Optional[float]

    # Computational intensity metrics
    flops_per_second: Optional[float]
    memory_bandwidth_used: Optional[float]
    arithmetic_intensity: Optional[float]

class SystemProfiler:
    """Profiles system performance during benchmarks."""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.gpu_available else 0
        self.cpu_count = mp.cpu_count()

        # Get system information
        self.cpu_info = self._get_cpu_info()
        self.memory_info = self._get_memory_info()
        self.gpu_info = self._get_gpu_info() if self.gpu_available else {}

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()

        info = {
            'model': 'Unknown',
            'cores': self.cpu_count,
            'threads': self.cpu_count
        }

        for line in cpuinfo.split('\n'):
            if 'model name' in line:
                info['model'] = line.split(':')[1].strip()
                break

        return info

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        mem = psutil.virtual_memory()
        return {
            'total_gb': round(mem.total / (1024**3), 2),
            'available_gb': round(mem.available / (1024**3), 2)
        }

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        if not self.gpu_available:
            return {}

        gpus = GPUtil.getGPUs()
        gpu_info = {}

        for i, gpu in enumerate(gpus):
            gpu_info[f'gpu_{i}'] = {
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'compute_capability': torch.cuda.get_device_capability(i)
            }

        return gpu_info

    def get_memory_usage(self, device: str = 'cpu') -> float:
        """Get current memory usage in MB."""
        if device == 'cpu':
            return psutil.Process().memory_info().rss / (1024**2)
        elif device.startswith('cuda'):
            if self.gpu_available:
                return torch.cuda.memory_allocated() / (1024**2)
        return 0.0

    def get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        if not self.gpu_available:
            return None
        try:
            gpus = GPUtil.getGPUs()
            return gpus[0].load * 100 if gpus else None
        except:
            return None

class HolographicBenchmark:
    """Main benchmarking class for holographic reconstruction."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.profiler = SystemProfiler()
        self.results: List[PerformanceMetrics] = []

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        print("=== System Information ===")
        print(f"CPU: {self.profiler.cpu_info['model']} ({self.profiler.cpu_count} cores)")
        print(f"Memory: {self.profiler.memory_info['total_gb']} GB")
        print(f"GPUs: {self.profiler.num_gpus}")
        for gpu_id, gpu_info in self.profiler.gpu_info.items():
            print(f"  {gpu_id}: {gpu_info['name']} ({gpu_info['memory_total']} MB)")
        print()

    def generate_test_hologram(self, height: int, width: int) -> torch.Tensor:
        """Generate synthetic hologram for testing."""
        # Create realistic hologram pattern with interference fringes
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Add multiple interference patterns at different scales
        pattern = torch.zeros_like(X)
        for scale in [2, 5, 10, 20]:
            pattern += torch.sin(scale * np.pi * (X**2 + Y**2))
            pattern += torch.cos(scale * np.pi * X * Y)

        # Add noise and normalize
        pattern += 0.1 * torch.randn_like(pattern)
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())

        return (255 * pattern).byte().float()

    def estimate_flops(self, image_size: Tuple[int, int], depth_count: int, batch_size: int = 1) -> float:
        """Estimate FLOPS for holographic reconstruction."""
        h, w = image_size

        # FFT operations dominate: N log N complex multiplies per plane
        # 2D FFT: 2 * N * log2(N) operations where N = h * w
        fft_ops_per_plane = 2 * h * w * (np.log2(h) + np.log2(w)) * 5  # 5 ops per complex multiply

        # Forward + inverse FFT per plane
        fft_ops = 2 * fft_ops_per_plane * depth_count * batch_size

        # Additional operations: padding, phase correction, magnitude
        misc_ops = h * w * depth_count * batch_size * 20

        return fft_ops + misc_ops

    def run_single_cpu_benchmark(self) -> None:
        """Benchmark single CPU mode."""
        print("=== Single CPU Benchmark ===")

        for image_size in tqdm(self.config.image_sizes, desc="Image sizes"):
            for depth_count in self.config.depth_counts:
                # Generate test data
                hologram = self.generate_test_hologram(*image_size)

                # Create reconstructor
                reconstructor = HolographicReconstructor(
                    num_planes=depth_count,
                    device='cpu'
                )

                # Warmup
                for _ in range(self.config.num_warmup):
                    _ = reconstructor.reconstruct_intensity(hologram)

                # Benchmark
                latencies = []
                memory_usage = []

                for _ in range(self.config.num_iterations):
                    # Measure memory before
                    mem_before = self.profiler.get_memory_usage('cpu')

                    # Time the operation
                    start_time = time.perf_counter()
                    result = reconstructor.reconstruct_intensity(hologram)
                    end_time = time.perf_counter()

                    # Measure memory after
                    mem_after = self.profiler.get_memory_usage('cpu')

                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    memory_usage.append(mem_after - mem_before)

                # Calculate metrics
                flops = self.estimate_flops(image_size, depth_count)
                mean_latency = np.mean(latencies)

                metrics = PerformanceMetrics(
                    mode="Single CPU",
                    image_size=image_size,
                    depth_count=depth_count,
                    batch_size=1,
                    mean_latency=mean_latency,
                    median_latency=np.median(latencies),
                    p95_latency=np.percentile(latencies, 95),
                    p99_latency=np.percentile(latencies, 99),
                    std_latency=np.std(latencies),
                    throughput=1000 / mean_latency,
                    peak_memory_used=np.max(memory_usage),
                    memory_efficiency=1.0,  # Baseline
                    cpu_utilization=psutil.cpu_percent(),
                    gpu_utilization=None,
                    gpu_memory_used=None,
                    flops_per_second=flops / (mean_latency / 1000),
                    memory_bandwidth_used=None,
                    arithmetic_intensity=None
                )

                self.results.append(metrics)

    def run_single_gpu_benchmark(self) -> None:
        """Benchmark single GPU mode."""
        if not self.profiler.gpu_available:
            print("Skipping GPU benchmarks - no CUDA available")
            return

        print("=== Single GPU Benchmark ===")

        for image_size in tqdm(self.config.image_sizes, desc="Image sizes"):
            for depth_count in self.config.depth_counts:
                # Generate test data
                hologram = self.generate_test_hologram(*image_size).cuda()

                # Create reconstructor
                reconstructor = HolographicReconstructor(
                    num_planes=depth_count,
                    device='cuda:0'
                )

                # Warmup
                for _ in range(self.config.num_warmup):
                    _ = reconstructor.reconstruct_intensity(hologram)
                torch.cuda.synchronize()

                # Benchmark
                latencies = []
                gpu_memory_usage = []

                for _ in range(self.config.num_iterations):
                    torch.cuda.reset_peak_memory_stats()

                    # Time the operation
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    result = reconstructor.reconstruct_intensity(hologram)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()

                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    gpu_memory_usage.append(torch.cuda.max_memory_allocated() / (1024**2))

                # Calculate metrics
                flops = self.estimate_flops(image_size, depth_count)
                mean_latency = np.mean(latencies)

                metrics = PerformanceMetrics(
                    mode="Single GPU",
                    image_size=image_size,
                    depth_count=depth_count,
                    batch_size=1,
                    mean_latency=mean_latency,
                    median_latency=np.median(latencies),
                    p95_latency=np.percentile(latencies, 95),
                    p99_latency=np.percentile(latencies, 99),
                    std_latency=np.std(latencies),
                    throughput=1000 / mean_latency,
                    peak_memory_used=np.max(gpu_memory_usage),
                    memory_efficiency=1.0,
                    cpu_utilization=psutil.cpu_percent(),
                    gpu_utilization=self.profiler.get_gpu_utilization(),
                    gpu_memory_used=np.max(gpu_memory_usage),
                    flops_per_second=flops / (mean_latency / 1000),
                    memory_bandwidth_used=None,
                    arithmetic_intensity=None
                )

                self.results.append(metrics)

    def run_batch_cpu_benchmark(self) -> None:
        """Benchmark batch CPU mode."""
        print("=== Batch CPU Benchmark ===")

        for image_size in tqdm(self.config.image_sizes, desc="Image sizes"):
            for depth_count in self.config.depth_counts:
                for batch_size in self.config.batch_sizes:
                    # Generate test data
                    batch_holograms = torch.stack([
                        self.generate_test_hologram(*image_size)
                        for _ in range(batch_size)
                    ]).unsqueeze(1)  # [N, 1, H, W]

                    # Warmup
                    for _ in range(self.config.num_warmup):
                        _ = generate_rgb_stack_images(
                            batch_holograms,
                            num_planes=depth_count
                        )

                    # Benchmark
                    latencies = []
                    memory_usage = []

                    for _ in range(self.config.num_iterations):
                        mem_before = self.profiler.get_memory_usage('cpu')

                        start_time = time.perf_counter()
                        result = generate_rgb_stack_images(
                            batch_holograms,
                            num_planes=depth_count
                        )
                        end_time = time.perf_counter()

                        mem_after = self.profiler.get_memory_usage('cpu')

                        latency_ms = (end_time - start_time) * 1000
                        latencies.append(latency_ms)
                        memory_usage.append(mem_after - mem_before)

                    # Calculate metrics
                    flops = self.estimate_flops(image_size, depth_count, batch_size)
                    mean_latency = np.mean(latencies)

                    metrics = PerformanceMetrics(
                        mode="Batch CPU",
                        image_size=image_size,
                        depth_count=depth_count,
                        batch_size=batch_size,
                        mean_latency=mean_latency,
                        median_latency=np.median(latencies),
                        p95_latency=np.percentile(latencies, 95),
                        p99_latency=np.percentile(latencies, 99),
                        std_latency=np.std(latencies),
                        throughput=1000 * batch_size / mean_latency,
                        peak_memory_used=np.max(memory_usage),
                        memory_efficiency=1.0,
                        cpu_utilization=psutil.cpu_percent(),
                        gpu_utilization=None,
                        gpu_memory_used=None,
                        flops_per_second=flops / (mean_latency / 1000),
                        memory_bandwidth_used=None,
                        arithmetic_intensity=None
                    )

                    self.results.append(metrics)

    def run_batch_gpu_benchmark(self) -> None:
        """Benchmark batch GPU mode."""
        if not self.profiler.gpu_available:
            return

        print("=== Batch GPU Benchmark ===")

        for image_size in tqdm(self.config.image_sizes, desc="Image sizes"):
            for depth_count in self.config.depth_counts:
                for batch_size in self.config.batch_sizes:
                    # Generate test data
                    batch_holograms = torch.stack([
                        self.generate_test_hologram(*image_size)
                        for _ in range(batch_size)
                    ]).unsqueeze(1).cuda()  # [N, 1, H, W]

                    # Warmup
                    for _ in range(self.config.num_warmup):
                        _ = generate_rgb_stack_images(
                            batch_holograms,
                            num_planes=depth_count
                        )
                    torch.cuda.synchronize()

                    # Benchmark
                    latencies = []
                    gpu_memory_usage = []

                    for _ in range(self.config.num_iterations):
                        torch.cuda.reset_peak_memory_stats()

                        torch.cuda.synchronize()
                        start_time = time.perf_counter()
                        result = generate_rgb_stack_images(
                            batch_holograms,
                            num_planes=depth_count
                        )
                        torch.cuda.synchronize()
                        end_time = time.perf_counter()

                        latency_ms = (end_time - start_time) * 1000
                        latencies.append(latency_ms)
                        gpu_memory_usage.append(torch.cuda.max_memory_allocated() / (1024**2))

                    # Calculate metrics
                    flops = self.estimate_flops(image_size, depth_count, batch_size)
                    mean_latency = np.mean(latencies)

                    metrics = PerformanceMetrics(
                        mode="Batch GPU",
                        image_size=image_size,
                        depth_count=depth_count,
                        batch_size=batch_size,
                        mean_latency=mean_latency,
                        median_latency=np.median(latencies),
                        p95_latency=np.percentile(latencies, 95),
                        p99_latency=np.percentile(latencies, 99),
                        std_latency=np.std(latencies),
                        throughput=1000 * batch_size / mean_latency,
                        peak_memory_used=np.max(gpu_memory_usage),
                        memory_efficiency=1.0,
                        cpu_utilization=psutil.cpu_percent(),
                        gpu_utilization=self.profiler.get_gpu_utilization(),
                        gpu_memory_used=np.max(gpu_memory_usage),
                        flops_per_second=flops / (mean_latency / 1000),
                        memory_bandwidth_used=None,
                        arithmetic_intensity=None
                    )

                    self.results.append(metrics)

    def run_cuda_cpp_benchmark(self) -> None:
        """Benchmark CUDA C++ (cuFFT) implementation."""
        try:
            import cuda_holographic
        except ImportError:
            print("Skipping CUDA C++ benchmarks - cuda_holographic module not found")
            print("Build with: cd cpp_cuda/build && cmake .. && make -j$(nproc)")
            return

        print("=== CUDA C++ Benchmark ===")

        for image_size in tqdm(self.config.image_sizes, desc="Image sizes"):
            for depth_count in self.config.depth_counts:
                hologram_np = self.generate_test_hologram(*image_size).numpy()

                recon = cuda_holographic.CUDAReconstructor(num_planes=depth_count)

                # Warmup
                for _ in range(self.config.num_warmup):
                    _ = recon.reconstruct_intensity(hologram_np)

                # Benchmark
                latencies = []
                for _ in range(self.config.num_iterations):
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    result = recon.reconstruct_intensity(hologram_np)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()

                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)

                flops = self.estimate_flops(image_size, depth_count)
                mean_latency = np.mean(latencies)

                metrics = PerformanceMetrics(
                    mode="CUDA C++",
                    image_size=image_size,
                    depth_count=depth_count,
                    batch_size=1,
                    mean_latency=mean_latency,
                    median_latency=np.median(latencies),
                    p95_latency=np.percentile(latencies, 95),
                    p99_latency=np.percentile(latencies, 99),
                    std_latency=np.std(latencies),
                    throughput=1000 / mean_latency,
                    peak_memory_used=0,
                    memory_efficiency=1.0,
                    cpu_utilization=psutil.cpu_percent(),
                    gpu_utilization=self.profiler.get_gpu_utilization(),
                    gpu_memory_used=None,
                    flops_per_second=flops / (mean_latency / 1000),
                    memory_bandwidth_used=None,
                    arithmetic_intensity=None
                )
                self.results.append(metrics)

    def run_openmp_benchmark(self) -> None:
        """Benchmark OpenMP C++ (FFTW) implementation with thread scaling."""
        try:
            import cuda_holographic
        except ImportError:
            print("Skipping OpenMP benchmarks - cuda_holographic module not found")
            return

        print("=== OpenMP C++ Benchmark ===")
        thread_counts = [1, 2, 4, 8, 16, 32, 64]

        for image_size in tqdm(self.config.image_sizes, desc="Image sizes"):
            for depth_count in self.config.depth_counts:
                hologram_np = self.generate_test_hologram(*image_size).numpy()

                for num_threads in thread_counts:
                    recon = cuda_holographic.CPUReconstructor(
                        num_planes=depth_count,
                        num_threads=num_threads
                    )

                    # Warmup
                    for _ in range(self.config.num_warmup):
                        _ = recon.reconstruct_intensity(hologram_np)

                    # Benchmark
                    latencies = []
                    for _ in range(self.config.num_iterations):
                        start_time = time.perf_counter()
                        result = recon.reconstruct_intensity(hologram_np)
                        end_time = time.perf_counter()

                        latency_ms = (end_time - start_time) * 1000
                        latencies.append(latency_ms)

                    flops = self.estimate_flops(image_size, depth_count)
                    mean_latency = np.mean(latencies)

                    metrics = PerformanceMetrics(
                        mode=f"OpenMP {num_threads}-threads",
                        image_size=image_size,
                        depth_count=depth_count,
                        batch_size=1,
                        mean_latency=mean_latency,
                        median_latency=np.median(latencies),
                        p95_latency=np.percentile(latencies, 95),
                        p99_latency=np.percentile(latencies, 99),
                        std_latency=np.std(latencies),
                        throughput=1000 / mean_latency,
                        peak_memory_used=0,
                        memory_efficiency=1.0,
                        cpu_utilization=psutil.cpu_percent(),
                        gpu_utilization=None,
                        gpu_memory_used=None,
                        flops_per_second=flops / (mean_latency / 1000),
                        memory_bandwidth_used=None,
                        arithmetic_intensity=None
                    )
                    self.results.append(metrics)

    def run_scaling_analysis(self) -> None:
        """Run scaling analysis across different dimensions."""
        print("=== Scaling Analysis ===")

        # Test with a medium-sized configuration
        base_size = (512, 512)
        base_depth = 10

        # Image size scaling (keep depth constant)
        print("Image Size Scaling:")
        sizes_to_test = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

        for mode_name, device in [("CPU", "cpu"), ("GPU", "cuda:0" if self.profiler.gpu_available else "cpu")]:
            if device.startswith("cuda") and not self.profiler.gpu_available:
                continue

            print(f"\n{mode_name} Scaling:")
            for size in sizes_to_test:
                try:
                    hologram = self.generate_test_hologram(*size)
                    if device.startswith("cuda"):
                        hologram = hologram.cuda()

                    reconstructor = HolographicReconstructor(
                        num_planes=base_depth,
                        device=device
                    )

                    # Quick timing
                    times = []
                    for _ in range(5):
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        _ = reconstructor.reconstruct_intensity(hologram)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        end = time.perf_counter()
                        times.append((end - start) * 1000)

                    mean_time = np.mean(times)
                    throughput = 1000 / mean_time
                    pixels = size[0] * size[1]

                    print(f"  {size}: {mean_time:.2f} ms, {throughput:.2f} ops/s, {pixels/mean_time*1000:.0f} pixels/s")

                except Exception as e:
                    print(f"  {size}: Failed - {e}")

        # Depth scaling (keep image size constant)
        print(f"\nDepth Scaling (size {base_size}):")
        depths_to_test = [1, 3, 5, 10, 20, 50]

        for mode_name, device in [("CPU", "cpu"), ("GPU", "cuda:0" if self.profiler.gpu_available else "cpu")]:
            if device.startswith("cuda") and not self.profiler.gpu_available:
                continue

            print(f"\n{mode_name} Depth Scaling:")
            hologram = self.generate_test_hologram(*base_size)
            if device.startswith("cuda"):
                hologram = hologram.cuda()

            for depth in depths_to_test:
                try:
                    reconstructor = HolographicReconstructor(
                        num_planes=depth,
                        device=device
                    )

                    # Quick timing
                    times = []
                    for _ in range(5):
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        _ = reconstructor.reconstruct_intensity(hologram)
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        end = time.perf_counter()
                        times.append((end - start) * 1000)

                    mean_time = np.mean(times)
                    throughput = 1000 / mean_time

                    print(f"  {depth} planes: {mean_time:.2f} ms, {throughput:.2f} ops/s, {mean_time/depth:.2f} ms/plane")

                except Exception as e:
                    print(f"  {depth} planes: Failed - {e}")

    def save_results(self) -> None:
        """Save benchmark results to files."""
        if not self.config.save_results:
            return

        # Convert results to JSON-serializable format
        results_dict = [asdict(result) for result in self.results]

        # Save detailed results
        results_file = Path(self.config.output_dir) / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'system_info': {
                    'cpu': self.profiler.cpu_info,
                    'memory': self.profiler.memory_info,
                    'gpu': self.profiler.gpu_info
                },
                'config': asdict(self.config),
                'results': results_dict
            }, f, indent=2)

        print(f"\nResults saved to {results_file}")

    def print_summary(self) -> None:
        """Print summary of benchmark results."""
        if not self.results:
            return

        print("\n=== Benchmark Summary ===")

        # Group results by mode
        by_mode = defaultdict(list)
        for result in self.results:
            by_mode[result.mode].append(result)

        for mode, results in by_mode.items():
            print(f"\n{mode} Results:")
            print(f"  Configurations tested: {len(results)}")

            # Find best and worst performance
            if results:
                best = min(results, key=lambda x: x.mean_latency)
                worst = max(results, key=lambda x: x.mean_latency)

                print(f"  Best latency: {best.mean_latency:.2f} ms ({best.image_size}, {best.depth_count} planes)")
                print(f"  Worst latency: {worst.mean_latency:.2f} ms ({worst.image_size}, {worst.depth_count} planes)")
                print(f"  Best throughput: {best.throughput:.2f} ops/s")

                # Average memory usage
                avg_memory = np.mean([r.peak_memory_used for r in results])
                print(f"  Average memory usage: {avg_memory:.1f} MB")

    def run_all_benchmarks(self) -> None:
        """Run all benchmark modes."""
        start_time = time.time()

        print("Starting comprehensive holographic reconstruction benchmark...")
        print(f"Test configurations: {len(self.config.image_sizes)} image sizes, "
              f"{len(self.config.depth_counts)} depth counts, "
              f"{len(self.config.batch_sizes)} batch sizes")
        print()

        # Run individual benchmarks
        self.run_single_cpu_benchmark()
        self.run_single_gpu_benchmark()
        self.run_batch_cpu_benchmark()
        self.run_batch_gpu_benchmark()

        # Run C++/CUDA and OpenMP benchmarks
        self.run_cuda_cpp_benchmark()
        self.run_openmp_benchmark()

        # Run scaling analysis
        self.run_scaling_analysis()

        # Save and summarize
        self.save_results()
        self.print_summary()

        total_time = time.time() - start_time
        print(f"\nTotal benchmark time: {total_time:.1f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Holographic Reconstruction Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer configurations")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    parser.add_argument("--iterations", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    if args.quick:
        config = BenchmarkConfig(
            image_sizes=[(256, 256), (512, 512)],
            depth_counts=[3, 10],
            batch_sizes=[1, 4],
            num_iterations=5,
            output_dir=args.output_dir,
            save_results=not args.no_save
        )
    else:
        config = BenchmarkConfig(
            image_sizes=[(128, 128), (256, 256), (512, 512), (1024, 1024)],
            depth_counts=[1, 3, 5, 10, 20],
            batch_sizes=[1, 2, 4, 8, 16],
            num_iterations=args.iterations,
            output_dir=args.output_dir,
            save_results=not args.no_save
        )

    benchmark = HolographicBenchmark(config)
    benchmark.run_all_benchmarks()

if __name__ == "__main__":
    main()