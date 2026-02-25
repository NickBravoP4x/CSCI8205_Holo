#!/usr/bin/env python3
"""
Roofline analysis for holographic reconstruction performance.
Analyzes compute vs memory-bound behavior across different configurations.

Part of the CSCI 8205 Multi-GPU Pipeline Architecture project.
"""

import os
import sys
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Handle display backend issues - fallback from Wayland to X11
def setup_display_backend():
    """Setup matplotlib backend with Wayland to X11 fallback."""
    # Force X11 if we have DISPLAY set and no WAYLAND_DISPLAY
    if os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    elif os.environ.get('WAYLAND_DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'wayland'
    else:
        # Headless environment - use non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        return

    # Try importing matplotlib with GUI backend
    try:
        import matplotlib.pyplot as plt
        # Test if backend works by creating a small plot
        fig, ax = plt.subplots(figsize=(1,1))
        plt.close(fig)
    except Exception as e:
        print(f"GUI backend failed, using non-interactive backend")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

# Setup backend before importing other matplotlib components
setup_display_backend()
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RooflineAnalyzer:
    """Roofline performance analysis for holographic reconstruction."""

    def __init__(self, results_file: str):
        """Load benchmark results for analysis."""
        with open(results_file, 'r') as f:
            self.data = json.load(f)

        self.system_info = self.data['system_info']
        self.results = self.data['results']

        # System specifications: AMD Threadripper PRO 7975WX + NVIDIA RTX 5090
        self.cpu_specs = {
            'peak_compute': 2700e9,  # ~2.7 TFLOPS FP32 (32c * 2 FMA * 16 SP/AVX-512 * 5.3GHz)
            'memory_bandwidth': 150e9,  # ~150 GB/s (8-channel DDR5-5200)
            'cache_l1': 32e3,  # 32 KB L1d per core
            'cache_l2': 1024e3,  # 1 MB L2 per core
            'cache_l3': 128e6   # 128 MB L3 total (4 x 32 MB CCDs)
        }

        self.gpu_specs = {
            'peak_compute': 105e12,  # ~105 TFLOPS FP32 (RTX 5090)
            'memory_bandwidth': 1790e9,  # ~1790 GB/s (GDDR7, RTX 5090)
            'memory_size': 32e9  # 32 GB VRAM per GPU
        }

    def calculate_operational_intensity(self, image_size: Tuple[int, int],
                                      depth_count: int, batch_size: int) -> float:
        """Calculate operational intensity (FLOPS/byte) for a configuration."""
        h, w = image_size
        total_pixels = h * w * depth_count * batch_size

        # FFT operations: ~5 * N * log2(N) per 2D FFT
        fft_ops = 2 * h * w * (np.log2(h) + np.log2(w)) * 5 * depth_count * batch_size

        # Memory transfers:
        # Input: h * w * batch_size * 4 bytes (float32)
        # Output: h * w * depth_count * batch_size * 4 bytes
        # Intermediate: padding, frequency domain data, etc.
        memory_bytes = (h * w * batch_size * 4 +  # input
                       h * w * depth_count * batch_size * 4 +  # output
                       2 * h * w * depth_count * batch_size * 8)  # intermediate complex data

        return fft_ops / memory_bytes

    def create_roofline_plot(self, device_type: str = 'both'):
        """Create roofline plot for the specified device(s)."""
        fig, axes = plt.subplots(1, 2 if device_type == 'both' else 1,
                                figsize=(15, 6) if device_type == 'both' else (8, 6))

        if device_type == 'both':
            cpu_ax, gpu_ax = axes
        else:
            if device_type == 'cpu':
                cpu_ax = axes
            else:
                gpu_ax = axes

        # Filter results by device
        cpu_results = [r for r in self.results if 'CPU' in r['mode']]
        gpu_results = [r for r in self.results if 'GPU' in r['mode']]

        if device_type in ['cpu', 'both'] and cpu_results:
            self._plot_roofline_single(cpu_ax, cpu_results, 'CPU', self.cpu_specs)

        if device_type in ['gpu', 'both'] and gpu_results:
            self._plot_roofline_single(gpu_ax, gpu_results, 'GPU', self.gpu_specs)

        plt.tight_layout()
        return fig

    def _plot_roofline_single(self, ax, results: List[Dict], device: str, specs: Dict):
        """Plot roofline for a single device type."""
        # Calculate operational intensity and achieved performance for each result
        intensities = []
        performances = []
        labels = []
        colors = []

        color_map = {'Single': 'blue', 'Batch': 'red'}

        for result in results:
            oi = self.calculate_operational_intensity(
                tuple(result['image_size']),
                result['depth_count'],
                result['batch_size']
            )

            # Convert throughput to FLOPS
            if result.get('flops_per_second'):
                perf = result['flops_per_second']
            else:
                # Estimate FLOPS from timing
                h, w = result['image_size']
                depth = result['depth_count']
                batch = result['batch_size']
                latency_s = result['mean_latency'] / 1000

                estimated_flops = self._estimate_flops(h, w, depth, batch)
                perf = estimated_flops / latency_s

            intensities.append(oi)
            performances.append(perf)

            # Create label
            mode = 'Single' if result['batch_size'] == 1 else 'Batch'
            label = f"{mode} {result['image_size'][0]}x{result['image_size'][1]}x{result['depth_count']}"
            labels.append(label)
            colors.append(color_map[mode])

        # Plot data points
        scatter = ax.scatter(intensities, performances, c=colors, alpha=0.7, s=60)

        # Draw roofline
        oi_range = np.logspace(-2, 2, 1000)

        # Memory-bound region
        memory_bound = specs['memory_bandwidth'] * oi_range

        # Compute-bound region (flat line)
        compute_bound = np.full_like(oi_range, specs['peak_compute'])

        # Actual roofline (minimum of the two)
        roofline = np.minimum(memory_bound, compute_bound)

        ax.loglog(oi_range, roofline, 'k-', linewidth=2, label='Roofline')
        ax.loglog(oi_range, memory_bound, 'g--', alpha=0.5, label='Memory Bound')
        ax.axhline(y=specs['peak_compute'], color='r', linestyle='--',
                  alpha=0.5, label='Compute Bound')

        ax.set_xlabel('Operational Intensity (FLOPS/byte)')
        ax.set_ylabel('Performance (FLOPS/s)')
        ax.set_title(f'{device} Roofline Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add some annotations for interesting points
        if len(intensities) > 0:
            max_perf_idx = np.argmax(performances)
            ax.annotate(f'Best: {labels[max_perf_idx]}',
                       xy=(intensities[max_perf_idx], performances[max_perf_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    def _estimate_flops(self, h: int, w: int, depth: int, batch: int) -> float:
        """Estimate FLOPS for holographic reconstruction."""
        # FFT operations dominate
        fft_ops_per_plane = 2 * h * w * (np.log2(h) + np.log2(w)) * 5
        total_ops = 2 * fft_ops_per_plane * depth * batch  # forward + inverse
        total_ops += h * w * depth * batch * 20  # misc operations
        return total_ops

    def analyze_scaling_efficiency(self):
        """Analyze scaling efficiency across different dimensions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Image size scaling
        self._plot_scaling_analysis(axes[0, 0], 'image_size', 'Image Size Scaling')

        # Depth scaling
        self._plot_scaling_analysis(axes[0, 1], 'depth_count', 'Depth Scaling')

        # Batch scaling
        self._plot_scaling_analysis(axes[1, 0], 'batch_size', 'Batch Size Scaling')

        # Memory usage vs performance
        self._plot_memory_vs_performance(axes[1, 1])

        plt.tight_layout()
        return fig

    def _plot_scaling_analysis(self, ax, dimension: str, title: str):
        """Plot scaling analysis for a specific dimension."""
        cpu_results = [r for r in self.results if 'CPU' in r['mode']]
        gpu_results = [r for r in self.results if 'GPU' in r['mode']]

        for results, label, color in [(cpu_results, 'CPU', 'blue'),
                                     (gpu_results, 'GPU', 'red')]:
            if not results:
                continue

            # Group by dimension value
            by_dim = {}
            for result in results:
                key = result[dimension]
                if dimension == 'image_size':
                    key = result['image_size'][0]  # Use width as key

                if key not in by_dim:
                    by_dim[key] = []
                by_dim[key].append(result)

            # Calculate average performance for each dimension value
            dim_values = sorted(by_dim.keys())
            avg_throughput = []

            for dim_val in dim_values:
                throughputs = [r['throughput'] for r in by_dim[dim_val]]
                avg_throughput.append(np.mean(throughputs))

            ax.plot(dim_values, avg_throughput, 'o-', label=label, color=color)

        ax.set_xlabel(dimension.replace('_', ' ').title())
        ax.set_ylabel('Throughput (ops/s)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if dimension in ['image_size', 'depth_count']:
            ax.set_yscale('log')

    def _plot_memory_vs_performance(self, ax):
        """Plot memory usage vs performance."""
        for results, label, color in [([r for r in self.results if 'CPU' in r['mode']], 'CPU', 'blue'),
                                     ([r for r in self.results if 'GPU' in r['mode']], 'GPU', 'red')]:
            if not results:
                continue

            memory_usage = [r['peak_memory_used'] for r in results]
            throughput = [r['throughput'] for r in results]

            ax.scatter(memory_usage, throughput, label=label, color=color, alpha=0.7)

        ax.set_xlabel('Peak Memory Usage (MB)')
        ax.set_ylabel('Throughput (ops/s)')
        ax.set_title('Memory Usage vs Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def generate_performance_report(self, output_dir: str):
        """Generate comprehensive performance report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create roofline plot
        roofline_fig = self.create_roofline_plot('both')
        roofline_fig.savefig(output_path / 'roofline_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(roofline_fig)

        # Create scaling analysis plot
        scaling_fig = self.analyze_scaling_efficiency()
        scaling_fig.savefig(output_path / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(scaling_fig)

        # Generate text report
        report_file = output_path / 'performance_report.txt'
        with open(report_file, 'w') as f:
            self._write_text_report(f)

        print(f"Performance report generated in {output_path}")
        print(f"  - roofline_analysis.png")
        print(f"  - scaling_analysis.png")
        print(f"  - performance_report.txt")

    def _write_text_report(self, f):
        """Write detailed text performance report."""
        f.write("HOLOGRAPHIC RECONSTRUCTION PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")

        # System information
        f.write("SYSTEM CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"CPU: {self.system_info['cpu']['model']}\n")
        f.write(f"CPU Cores: {self.system_info['cpu']['cores']}\n")
        f.write(f"Memory: {self.system_info['memory']['total_gb']} GB\n")

        if self.system_info['gpu']:
            for gpu_id, gpu_info in self.system_info['gpu'].items():
                f.write(f"{gpu_id}: {gpu_info['name']} ({gpu_info['memory_total']} MB)\n")
        f.write("\n")

        # Performance summary by mode
        f.write("PERFORMANCE SUMMARY BY MODE\n")
        f.write("-" * 30 + "\n")

        modes = set(r['mode'] for r in self.results)
        for mode in sorted(modes):
            mode_results = [r for r in self.results if r['mode'] == mode]

            f.write(f"\n{mode}:\n")
            f.write(f"  Configurations tested: {len(mode_results)}\n")

            if mode_results:
                latencies = [r['mean_latency'] for r in mode_results]
                throughputs = [r['throughput'] for r in mode_results]
                memories = [r['peak_memory_used'] for r in mode_results]

                f.write(f"  Latency: {np.min(latencies):.2f} - {np.max(latencies):.2f} ms\n")
                f.write(f"  Throughput: {np.min(throughputs):.2f} - {np.max(throughputs):.2f} ops/s\n")
                f.write(f"  Memory: {np.min(memories):.1f} - {np.max(memories):.1f} MB\n")

                # Best configuration
                best_idx = np.argmin(latencies)
                best = mode_results[best_idx]
                f.write(f"  Best config: {best['image_size']}x{best['depth_count']} "
                       f"(batch={best['batch_size']}) - {best['mean_latency']:.2f} ms\n")

def main():
    parser = argparse.ArgumentParser(description="Roofline Analysis for Holographic Reconstruction")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", default="roofline_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--device", choices=['cpu', 'gpu', 'both'], default='both',
                       help="Device type for roofline analysis")

    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"Error: Results file {args.results_file} not found")
        return

    analyzer = RooflineAnalyzer(args.results_file)
    analyzer.generate_performance_report(args.output_dir)

if __name__ == "__main__":
    main()