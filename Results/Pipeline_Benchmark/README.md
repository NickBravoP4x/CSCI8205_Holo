# Pipeline Benchmark Results
## CSCI 8205: Multi-GPU Pipeline Architecture for Real-Time Digital Inline Holographic Reconstruction

**Date**: February 25, 2026
**Phase**: End-to-end pipeline profiling (GPU vs CPU comparison)
**Implementation**: Python/PyTorch pipeline (`Scripts/pipeline.py`)

## System Configuration
- **CPU**: x86_64 (used for CPU-only baseline)
- **GPU**: NVIDIA GeForce RTX 5090 (32 GB GDDR7, Compute 12.0)
- **PyTorch**: 2.10.0+cu130
- **CUDA**: 13.0
- **Python**: 3.11.14

## Pipeline Stages

| # | Stage | Description |
|---|-------|-------------|
| 1 | Image Load | Read frame from disk + transfer to device |
| 2 | EMA Background | Exponential moving average background subtraction |
| 3 | Holographic Recon | FFT-based holographic reconstruction |
| 4 | Peak Detection | Heatmap inference + peak finding (top-k) |
| 5 | Classification | Per-detection crop + classifier inference |
| 6 | Post-processing | Threshold filtering + result aggregation |

## Results Summary

### GPU vs CPU Per-Stage Latency

| Stage | GPU mean (ms) | CPU mean (ms) | Speedup |
|-------|--------------|--------------|---------|
| Image Load | 5.629 | 5.745 | 1.0x |
| EMA Background | 0.232 | 0.330 | 1.4x |
| Holographic Recon | 0.439 | 8.237 | 18.8x |
| Peak Detection | 2.484 | 12.173 | 4.9x |
| Classification | 9.003 | 22.524 | 2.5x |
| Post-processing | 0.239 | 0.302 | 1.3x |
| **End-to-End** | **18.026** | **49.311** | **2.7x** |

- **GPU throughput**: 55.5 FPS (10,000 images, 181.5 s wall-clock)
- **CPU throughput**: 20.3 FPS (100 images, 4.9 s wall-clock)
- **GPU peak memory**: 100.9 MB allocated / 130.0 MB reserved

### Key Observations
1. **Holographic reconstruction** shows the largest GPU speedup (18.8x) — dominated by FFT, ideal for GPU acceleration.
2. **Image load** is I/O-bound and shows no GPU benefit (1.0x), as expected.
3. **Classification** is the GPU bottleneck stage (9.0 ms mean) due to variable detection counts per frame; frames with no detections complete in microseconds while busy frames require full classifier inference.
4. End-to-end speedup (2.7x) is lower than per-stage compute speedups because I/O and classification dominate total time.

## Reproduction

### Prerequisites
```bash
conda activate holovision
# Requires: pytorch, matplotlib, seaborn, numpy
# Data directory: Data/MA/run1 (holographic image frames)
```

### Run benchmarks
```bash
# GPU full benchmark (10,000 images)
python Scripts/pipeline.py --device cuda --num-images 10000

# CPU benchmark (100 images — slower, reduced sample)
python Scripts/pipeline.py --device cpu --num-images 100

# GPU quick benchmark (100 images, for validation)
python Scripts/pipeline.py --device cuda --num-images 100
```

### Generate comparison figure
```bash
python Scripts/plot_pipeline_benchmark.py
# Produces: Results/Pipeline_Benchmark/figures/pipeline_gpu_vs_cpu.png
# Prints:   comparison table to stdout
```

## Files Included

| Path | Description |
|------|-------------|
| `gpu_full/pipeline_results.json` | GPU benchmark, 10,000 images (9,995 timed) |
| `gpu_quick/pipeline_results.json` | GPU benchmark, 100 images (95 timed) |
| `cpu_quick/pipeline_results.json` | CPU benchmark, 100 images (95 timed) |
| `figures/pipeline_gpu_vs_cpu.png` | GPU vs CPU comparison bar chart |
| `README.md` | This file |

## Methodology Notes
- **EMA warmup exclusion**: First 5 images are used for EMA warmup and excluded from timing (`num_timed = num_images - 5`).
- **Sample size difference**: GPU full uses 10,000 images for statistically robust measurements; CPU uses 100 images due to slower execution. Per-stage means are stable at both sample sizes (GPU quick vs GPU full show <5% variance).
- **Timing**: Per-stage latencies measured with `time.perf_counter()` (CPU) or `torch.cuda.Event` (GPU) with proper synchronization.
- **Detection variability**: Classification stage latency is bimodal — near-zero for frames with no detections, 15-25 ms for frames with detections. Mean reflects the mixture.

---
*Pipeline benchmark for the multi-GPU holographic processing pipeline architecture research.*
