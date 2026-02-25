#!/usr/bin/env python3
"""
Cross-validation: Python (PyTorch CPU) vs CUDA C++ vs OpenMP C++.
Verifies that all three implementations produce matching results.

Part of CSCI 8205 Multi-GPU Pipeline Architecture project.
"""

import sys
import numpy as np
import torch

# Import Python baseline
try:
    from holographic_reconstruction import HolographicReconstructor
except ImportError:
    sys.path.insert(0, ".")
    from holographic_reconstruction import HolographicReconstructor

# Import C++/CUDA module
try:
    import cuda_holographic
except ImportError:
    print("ERROR: cuda_holographic module not found.")
    print("Build with: cd cpp_cuda/build && cmake .. && make -j$(nproc)")
    print("Then: export PYTHONPATH=.../cpp_cuda/build:$PYTHONPATH")
    sys.exit(1)


def generate_test_hologram(height: int, width: int) -> np.ndarray:
    """Generate synthetic hologram matching the benchmark's pattern."""
    x = np.linspace(-1, 1, width, dtype=np.float32)
    y = np.linspace(-1, 1, height, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')

    pattern = np.zeros_like(X)
    for scale in [2, 5, 10, 20]:
        pattern += np.sin(scale * np.pi * (X**2 + Y**2))
        pattern += np.cos(scale * np.pi * X * Y)

    pattern += 0.1 * np.random.RandomState(42).randn(*pattern.shape).astype(np.float32)
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    hologram = (255 * pattern).astype(np.uint8).astype(np.float32)
    return hologram


def compute_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute max relative error between two arrays."""
    max_abs = max(np.abs(a).max(), np.abs(b).max())
    if max_abs < 1e-10:
        return 0.0
    return float(np.abs(a - b).max() / max_abs)


def run_python_baseline(hologram_np: np.ndarray, depth: int) -> np.ndarray:
    """Run Python (PyTorch CPU) reconstruction."""
    recon = HolographicReconstructor(num_planes=depth, device='cpu')
    hologram_t = torch.from_numpy(hologram_np).float()
    result = recon.reconstruct_intensity(hologram_t)
    return result.numpy()


def run_cuda_cpp(hologram_np: np.ndarray, depth: int) -> np.ndarray:
    """Run CUDA C++ reconstruction."""
    recon = cuda_holographic.CUDAReconstructor(num_planes=depth)
    return recon.reconstruct_intensity(hologram_np)


def run_openmp_cpp(hologram_np: np.ndarray, depth: int, threads: int = 4) -> np.ndarray:
    """Run OpenMP C++ reconstruction."""
    recon = cuda_holographic.CPUReconstructor(num_planes=depth, num_threads=threads)
    return recon.reconstruct_intensity(hologram_np)


def main():
    # Cross-implementation tolerance: float32 FFT libraries (cuFFT, FFTW, PyTorch)
    # use different radix algorithms and accumulation orders, causing rounding
    # differences that grow with problem size. After min-max normalization
    # (intensity range ~0-100k mapped to [0,1]), these small absolute errors
    # become relative errors of ~1e-4 to 1e-2. The intra-C++ tolerance (CUDA
    # vs OpenMP) is much tighter (~1e-5) since both use the same filter values.
    CROSS_IMPL_TOLERANCE = 1e-2   # C++ vs Python (different FFT backends)
    INTRA_CPP_TOLERANCE  = 1e-3   # CUDA vs OpenMP (same algorithm)

    image_sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    depth_counts = [1, 3, 5, 10, 20]

    print("=" * 70)
    print("Cross-Validation: Python (PyTorch CPU) vs CUDA C++ vs OpenMP C++")
    print("=" * 70)
    print(f"Cross-implementation tolerance (C++ vs Python): {CROSS_IMPL_TOLERANCE}")
    print(f"Intra-C++ tolerance (CUDA vs OpenMP):           {INTRA_CPP_TOLERANCE}")
    print()

    total_configs = 0
    passed_configs = 0
    failed_configs = []

    for h, w in image_sizes:
        for depth in depth_counts:
            total_configs += 1
            config_str = f"{h}x{w} x {depth} planes"
            print(f"Testing {config_str}...", end=" ", flush=True)

            hologram = generate_test_hologram(h, w)

            # Run all three implementations
            try:
                py_result = run_python_baseline(hologram, depth)
            except Exception as e:
                print(f"FAIL (Python error: {e})")
                failed_configs.append((config_str, f"Python error: {e}"))
                continue

            try:
                cuda_result = run_cuda_cpp(hologram, depth)
            except Exception as e:
                print(f"FAIL (CUDA error: {e})")
                failed_configs.append((config_str, f"CUDA error: {e}"))
                continue

            try:
                omp_result = run_openmp_cpp(hologram, depth)
            except Exception as e:
                print(f"FAIL (OpenMP error: {e})")
                failed_configs.append((config_str, f"OpenMP error: {e}"))
                continue

            # Compare results
            err_cuda_py = compute_relative_error(cuda_result, py_result)
            err_omp_py  = compute_relative_error(omp_result, py_result)
            err_cuda_omp = compute_relative_error(cuda_result, omp_result)

            cross_pass = max(err_cuda_py, err_omp_py) < CROSS_IMPL_TOLERANCE
            intra_pass = err_cuda_omp < INTRA_CPP_TOLERANCE
            passed = cross_pass and intra_pass

            if passed:
                passed_configs += 1
                print(f"PASS (CUDA-Py: {err_cuda_py:.2e}, OMP-Py: {err_omp_py:.2e}, CUDA-OMP: {err_cuda_omp:.2e})")
            else:
                reasons = []
                if not cross_pass:
                    reasons.append(f"cross-impl err={max(err_cuda_py, err_omp_py):.2e}")
                if not intra_pass:
                    reasons.append(f"intra-C++ err={err_cuda_omp:.2e}")
                failed_configs.append((config_str, "; ".join(reasons)))
                print(f"FAIL (CUDA-Py: {err_cuda_py:.2e}, OMP-Py: {err_omp_py:.2e}, CUDA-OMP: {err_cuda_omp:.2e})")

    print()
    print("=" * 70)
    print(f"Results: {passed_configs}/{total_configs} passed")

    if failed_configs:
        print("\nFailed configurations:")
        for cfg, reason in failed_configs:
            print(f"  {cfg}: {reason}")
        return 1
    else:
        print("All configurations passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
