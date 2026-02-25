# Multi-GPU Pipeline for Real-Time Digital Inline Holographic Reconstruction

CSCI 8205: Design and Implementation of Multiprocessor Systems — University of Minnesota

**Author:** Nicholas Bravo-Frank (bravo095@umn.edu)

## Overview

This project designs, implements, and evaluates a multi-GPU pipeline architecture for real-time digital inline holography (DIH) particle reconstruction on a 1–4 GPU workstation. The pipeline consists of six computational stages: image resize, EMA enhancement, angular spectrum propagation, CNN-based particle detection, ML classification, and post-processing.

## Key Components

- **C++/CUDA reconstruction** — Angular spectrum propagation using cuFFT with depth-parallel and spatial-parallel multi-GPU strategies
- **OpenMP CPU baseline** — FFTW-based reconstruction with strong scaling from 1–32 cores on AMD Threadripper
- **Pipeline stage grouping** — Evaluation of multiple stage-to-GPU mappings under compute-balanced and bottleneck-isolated strategies
- **Lock-free inter-stage communication** — Cache-line-aligned SPSC ring buffers with GPU-pinned memory, benchmarked against blocking queues and CUDA stream pipelining
- **Asynchronous autofocus subsystem** — Work-stealing queue with dedicated GPU workers for per-particle focal plane determination

## Repository Structure

```
cpp_cuda/          C++/CUDA source and CMake build system
Scripts/           Python scripts for analysis and visualization
Data/              Input hologram data
Results/           Benchmark and experiment outputs
Ref/               Project proposal and reference documents
benchmark_results/ Raw benchmark data
roofline_analysis/ Roofline model plots and data
full_analysis/     Comprehensive profiling results
```

## Building

```bash
cd cpp_cuda
mkdir build && cd build
cmake ..
make
```

Requires: CUDA Toolkit, cuFFT, CMake, a C++17 compiler. Optional: FFTW3, OpenMP.
