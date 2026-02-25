# Week 1-2 Python Baseline Results
## CSCI 8205: Multi-GPU Pipeline Architecture for Real-Time Digital Inline Holographic Reconstruction

**Date**: February 25, 2026
**Phase**: Initial baseline measurements (Week 1-2 of project timeline)
**Implementation**: Python/PyTorch baseline

## System Configuration
- **CPU**: Intel i7-13620H (16 cores)
- **Memory**: 46.67 GB
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8GB, Compute 8.9)
- **Environment**: Python with PyTorch, holovision conda environment

## Benchmark Results Summary

### Key Performance Metrics
- **Total configurations tested**: 240 (comprehensive parameter sweep)
- **Image sizes**: 128×128 to 1024×1024
- **Depth counts**: 1-20 reconstruction planes
- **Batch sizes**: 1-16 images per batch

### Critical Findings for Project
1. **Real-time feasibility**: ✅ GPU achieves 0.60-0.92ms (well under 2.5ms target for 400 Hz)
2. **GPU speedup**: ~10-15× average, up to 20× for large problems (matches proposal expectations)
3. **Memory scaling**: Up to 2.2GB GPU memory for largest configurations
4. **Throughput range**: 1.98 to 1,691 ops/second

### Project Timeline Validation
These results validate the approach outlined in the CSCI 8205 proposal:
- Single GPU performance exceeds real-time requirements
- Multi-GPU architecture will be critical for autofocus workload (10-100× more expensive)
- Memory bandwidth and compute-bound characterization ready for roofline analysis
- Baseline established for C++/CUDA comparison (Weeks 3-4)

## Files Included
- `benchmark_results/` - Complete benchmark data (JSON format)
- `full_analysis/` - Roofline plots and performance report
- `holographic_reconstruction.py` - Baseline Python implementation
- `holographic_benchmark.py` - Benchmarking framework
- `roofline_analysis.py` - Performance analysis tools

## Next Steps (Week 3-4)
1. Port to C++/CUDA implementation
2. Validate against these Python baselines
3. Measure performance improvements
4. Begin multi-GPU pipeline architecture design

## Usage Notes
- All benchmarks run with automatic Wayland→X11 display fallback
- Results compatible with roofline analysis framework
- Benchmarking framework ready for comparison with future implementations

---
*This baseline establishes the foundation for the multi-GPU holographic processing pipeline architecture research.*