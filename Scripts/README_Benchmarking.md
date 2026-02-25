# Holographic Reconstruction Benchmarking Framework

This benchmarking framework provides comprehensive performance analysis for your holographic reconstruction implementation, supporting all the metrics and analysis needed for your CSCI 8205 project.

## Files Created

1. **`holographic_benchmark.py`** - Main benchmarking framework
2. **`roofline_analysis.py`** - Roofline performance analysis
3. **`CSCI8205_Proposal_BravoFrank.txt`** - Readable version of your proposal

## Features

### Benchmarking Modes
- **Single CPU**: Individual hologram processing on CPU
- **Single GPU**: Individual hologram processing on GPU
- **Batch CPU**: Batched hologram processing on CPU
- **Batch GPU**: Batched hologram processing on GPU

### Performance Metrics
- **Latency**: Mean, median, P95, P99 latencies
- **Throughput**: Operations per second
- **Memory Usage**: Peak memory consumption
- **System Utilization**: CPU and GPU utilization
- **Computational Intensity**: FLOPS/byte analysis
- **Scaling Analysis**: Performance across image sizes and depths

### Roofline Analysis
- Compute vs memory-bound characterization
- Operational intensity calculation
- Hardware utilization analysis
- Performance bottleneck identification

## Quick Start

### Install Dependencies
```bash
pip install numpy torch matplotlib seaborn tqdm psutil GPUtil
```

### Run Basic Benchmark
```bash
cd /home/p4x/software/CSCI8205/Scripts

# Quick benchmark (recommended for testing)
python holographic_benchmark.py --quick

# Full benchmark (comprehensive analysis)
python holographic_benchmark.py

# Custom configuration
python holographic_benchmark.py --iterations 50 --output-dir my_results
```

### Run Roofline Analysis
```bash
# After running benchmark, analyze results
python roofline_analysis.py benchmark_results/benchmark_results.json

# Analyze specific device
python roofline_analysis.py benchmark_results/benchmark_results.json --device gpu
```

## Benchmark Configuration

### Default Full Configuration
- **Image sizes**: 128×128, 256×256, 512×512, 1024×1024
- **Depth counts**: 1, 3, 5, 10, 20 planes
- **Batch sizes**: 1, 2, 4, 8, 16
- **Iterations**: 20 per configuration
- **Warmup**: 5 iterations

### Quick Configuration
- **Image sizes**: 256×256, 512×512
- **Depth counts**: 3, 10 planes
- **Batch sizes**: 1, 4
- **Iterations**: 5 per configuration

## Output Files

### Benchmark Results
- `benchmark_results/benchmark_results.json` - Complete benchmark data
- Contains system info, configuration, and detailed results

### Roofline Analysis
- `roofline_analysis/roofline_analysis.png` - Roofline plots
- `roofline_analysis/scaling_analysis.png` - Scaling efficiency plots
- `roofline_analysis/performance_report.txt` - Detailed text report

## Key Metrics for Your Project

Based on your proposal, focus on these metrics:

### Real-Time Requirements (400 Hz target)
- **Target latency**: <2.5ms per hologram (1000ms / 400Hz)
- **Pipeline latency**: 11-13ms total (your 6-stage pipeline)
- **Tail latency**: P99/P50 ratio for predictability

### Multi-GPU Scaling
- **Strong scaling**: Performance vs GPU count
- **Scaling efficiency**: Actual vs theoretical speedup
- **Memory bandwidth utilization**: Roofline analysis

### System Architecture
- **NUMA effects**: Memory locality impact
- **Cache coherence**: L1/L2/L3 miss rates
- **Pipeline utilization**: GPU active time percentage

## Expected Results (from your proposal)

1. **GPU vs CPU**: Expect ~10× GPU speedup
2. **Multi-GPU scaling**: 8-12× speedup with >80% efficiency up to 4 GPUs
3. **Memory bottleneck**: CPU saturates memory bandwidth earlier
4. **Autofocus impact**: 10-100× computational overhead vs core pipeline

## Integration with Your Project Timeline

### Week 1-2 (Current): Baseline Benchmarking
- Run comprehensive benchmarks on current Python implementation
- Establish baseline performance metrics
- Generate roofline analysis plots

### Week 3-4: C++/CUDA Validation
- Compare C++/CUDA implementation against these Python baselines
- Validate correctness using same test configurations
- Measure performance improvements

### Week 5+: Pipeline Architecture
- Use these metrics to inform stage-to-GPU grouping decisions
- Benchmark different pipeline configurations
- Compare against baseline single-stage performance

## System-Specific Tuning

You may need to adjust hardware specifications in `roofline_analysis.py`:

```python
self.cpu_specs = {
    'peak_compute': 2000e9,  # Adjust for your Threadripper
    'memory_bandwidth': 100e9,  # Adjust for your memory
}

self.gpu_specs = {
    'peak_compute': 10e12,  # Adjust for your GPU
    'memory_bandwidth': 900e9,  # Adjust for your GPU memory
}
```

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch sizes or image sizes
2. **Long benchmark time**: Use `--quick` flag for initial testing
3. **Missing dependencies**: Install with pip as shown above

### Performance Optimization
1. **GPU warmup**: Framework includes automatic warmup iterations
2. **Memory management**: Automatic cleanup between tests
3. **System interference**: Close other applications during benchmarking

## Next Steps

1. **Run baseline benchmarks**: Establish current performance
2. **Analyze bottlenecks**: Use roofline analysis to identify limitations
3. **Plan C++/CUDA port**: Use insights to optimize implementation
4. **Design pipeline**: Use metrics to inform multi-GPU architecture

This framework provides all the baseline measurements needed for your multi-GPU pipeline architecture project!