#!/usr/bin/env python3
"""
Roofline analysis for holographic reconstruction performance.
Analyzes compute vs memory-bound behavior across different configurations.

Part of the CSCI 8205 Multi-GPU Pipeline Architecture project.
"""

import os
import sys
import csv
import re
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

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

    # ─── Phase 6: Nsight Compute CSV parsing ───────────────────────────────
    def parse_ncu_csv(self, csv_path: str) -> List[Dict]:
        """Parse Nsight Compute CSV export for per-kernel empirical roofline data."""
        if not Path(csv_path).exists():
            print(f"Warning: ncu CSV not found at {csv_path}")
            return []

        kernels = []
        with open(csv_path, 'r') as f:
            # Skip lines until we find the header
            reader = csv.reader(f)
            header = None
            for row in reader:
                if row and 'Kernel Name' in row[0] if row else False:
                    header = row
                    break
                if row and len(row) > 5:
                    # Try to detect header
                    if any('Kernel' in str(c) for c in row):
                        header = row
                        break

            if header is None:
                # Fallback: try reading as standard CSV
                f.seek(0)
                reader = csv.DictReader(f)
                for row in reader:
                    kernels.append(dict(row))
                return kernels

            for row in reader:
                if len(row) >= len(header):
                    entry = dict(zip(header, row))
                    kernels.append(entry)

        return kernels

    def extract_ncu_roofline_points(self, ncu_data: List[Dict]) -> List[Dict]:
        """Extract (arithmetic_intensity, achieved_flops) from ncu metrics."""
        points = []
        for kernel in ncu_data:
            name = kernel.get('Kernel Name', kernel.get('kernel_name', 'unknown'))
            # Common ncu metric names
            flops = None
            bytes_moved = None
            duration_ns = None

            for key, val in kernel.items():
                key_l = key.lower().strip()
                try:
                    v = float(str(val).replace(',', ''))
                except (ValueError, TypeError):
                    continue

                if 'flop_count_sp' in key_l or 'sm__sass_thread_inst_executed_op_ffma_pred_on' in key_l:
                    flops = v
                elif 'dram__bytes' in key_l and 'sum' in key_l:
                    bytes_moved = v
                elif 'gpu__time_duration' in key_l or 'duration' in key_l:
                    duration_ns = v

            if flops and bytes_moved and bytes_moved > 0:
                ai = flops / bytes_moved
                achieved = flops / (duration_ns * 1e-9) if duration_ns else None
                points.append({
                    'kernel': name,
                    'arithmetic_intensity': ai,
                    'achieved_flops': achieved,
                    'flops': flops,
                    'bytes': bytes_moved,
                })

        return points

    def parse_perf_report(self, perf_path: str) -> Dict:
        """Parse perf stat output for CPU cache/cycle metrics."""
        if not Path(perf_path).exists():
            return {}

        metrics = {}
        with open(perf_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Parse lines like "123,456 cache-misses"
                m = re.match(r'([\d,]+)\s+(\S+)', line)
                if m:
                    val_str = m.group(1).replace(',', '')
                    try:
                        metrics[m.group(2)] = int(val_str)
                    except ValueError:
                        pass
                # Parse IPC from "insn per cycle" line
                m2 = re.match(r'.*#\s+([\d.]+)\s+insn per cycle', line)
                if m2:
                    metrics['ipc'] = float(m2.group(1))
        return metrics

    # ─── Multi-level roofline ────────────────────────────────────────────
    def create_multilevel_roofline(self, ncu_csv_path: Optional[str] = None):
        """Create multi-level roofline (L1/L2/L3/DRAM bandwidth ceilings)."""
        fig, (cpu_ax, gpu_ax) = plt.subplots(1, 2, figsize=(16, 7))

        oi_range = np.logspace(-2, 3, 1000)

        # ─ CPU multi-level roofline ─
        cpu_bw = {
            'L1 (~2 TB/s)':   2000e9,
            'L2 (~800 GB/s)':  800e9,
            'L3 (~400 GB/s)':  400e9,
            'DRAM (150 GB/s)': self.cpu_specs['memory_bandwidth'],
        }
        colors_bw = ['#ff7f0e', '#2ca02c', '#9467bd', '#1f77b4']
        for (label, bw), c in zip(cpu_bw.items(), colors_bw):
            cpu_ax.loglog(oi_range, bw * oi_range, '--', color=c, alpha=0.6, label=label)

        cpu_ax.axhline(y=self.cpu_specs['peak_compute'], color='r', linestyle='-',
                       linewidth=2, alpha=0.8, label=f"Peak FP32 ({self.cpu_specs['peak_compute']/1e12:.1f} TF)")
        roofline_cpu = np.minimum(self.cpu_specs['memory_bandwidth'] * oi_range,
                                  self.cpu_specs['peak_compute'])
        cpu_ax.loglog(oi_range, roofline_cpu, 'k-', linewidth=2.5, label='DRAM Roofline')

        # Plot benchmark data points
        cpu_results = [r for r in self.results if 'CPU' in r['mode'] or 'OpenMP' in r['mode']]
        self._scatter_on_roofline(cpu_ax, cpu_results)

        cpu_ax.set_xlabel('Arithmetic Intensity (FLOPS/byte)')
        cpu_ax.set_ylabel('Performance (FLOPS/s)')
        cpu_ax.set_title('CPU Multi-Level Roofline')
        cpu_ax.legend(fontsize=8, loc='lower right')
        cpu_ax.grid(True, alpha=0.3)
        cpu_ax.set_ylim(1e7, 1e13)

        # ─ GPU multi-level roofline ─
        gpu_bw = {
            'L1/Shared (~12 TB/s)': 12000e9,
            'L2 (~6 TB/s)':         6000e9,
            'GDDR7 (1790 GB/s)':    self.gpu_specs['memory_bandwidth'],
        }
        colors_gpu = ['#ff7f0e', '#2ca02c', '#1f77b4']
        for (label, bw), c in zip(gpu_bw.items(), colors_gpu):
            gpu_ax.loglog(oi_range, bw * oi_range, '--', color=c, alpha=0.6, label=label)

        gpu_ax.axhline(y=self.gpu_specs['peak_compute'], color='r', linestyle='-',
                       linewidth=2, alpha=0.8, label=f"Peak FP32 ({self.gpu_specs['peak_compute']/1e12:.0f} TF)")
        roofline_gpu = np.minimum(self.gpu_specs['memory_bandwidth'] * oi_range,
                                  self.gpu_specs['peak_compute'])
        gpu_ax.loglog(oi_range, roofline_gpu, 'k-', linewidth=2.5, label='GDDR7 Roofline')

        # Plot GPU data points
        gpu_results = [r for r in self.results if 'GPU' in r['mode'] or 'CUDA' in r['mode']]
        self._scatter_on_roofline(gpu_ax, gpu_results)

        # Overlay ncu empirical points if available
        if ncu_csv_path:
            ncu_data = self.parse_ncu_csv(ncu_csv_path)
            ncu_points = self.extract_ncu_roofline_points(ncu_data)
            if ncu_points:
                ais = [p['arithmetic_intensity'] for p in ncu_points if p['achieved_flops']]
                perfs = [p['achieved_flops'] for p in ncu_points if p['achieved_flops']]
                names = [p['kernel'] for p in ncu_points if p['achieved_flops']]
                gpu_ax.scatter(ais, perfs, marker='*', s=200, c='gold', edgecolors='black',
                             zorder=10, label='ncu empirical')
                for ai, pf, nm in zip(ais, perfs, names):
                    short = nm.split('(')[0][-25:]
                    gpu_ax.annotate(short, (ai, pf), fontsize=6,
                                   xytext=(5, 5), textcoords='offset points')

        gpu_ax.set_xlabel('Arithmetic Intensity (FLOPS/byte)')
        gpu_ax.set_ylabel('Performance (FLOPS/s)')
        gpu_ax.set_title('GPU Multi-Level Roofline')
        gpu_ax.legend(fontsize=8, loc='lower right')
        gpu_ax.grid(True, alpha=0.3)
        gpu_ax.set_ylim(1e8, 1e14)

        plt.tight_layout()
        return fig

    def _scatter_on_roofline(self, ax, results):
        """Add benchmark points to a roofline axis."""
        if not results:
            return

        # Color by mode
        mode_colors = {}
        cmap = plt.cm.tab10
        for i, mode in enumerate(sorted(set(r['mode'] for r in results))):
            mode_colors[mode] = cmap(i % 10)

        for result in results:
            oi = self.calculate_operational_intensity(
                tuple(result['image_size']), result['depth_count'], result['batch_size'])
            perf = result.get('flops_per_second')
            if not perf:
                h, w = result['image_size']
                perf = self._estimate_flops(h, w, result['depth_count'],
                                            result['batch_size']) / (result['mean_latency'] / 1000)
            ax.scatter(oi, perf, color=mode_colors[result['mode']], alpha=0.6, s=40,
                      edgecolors='none')

    # ─── Phase 7: Enhanced Visualizations ────────────────────────────────
    def plot_implementation_comparison(self):
        """Bar chart comparing Python vs CUDA C++ vs OpenMP latency."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Group results by mode, averaging across configs
        mode_data = {}
        for r in self.results:
            mode = r['mode']
            # Normalize mode names for grouping
            if 'OpenMP' in mode:
                mode = 'OpenMP (best)'
            if mode not in mode_data:
                mode_data[mode] = []
            mode_data[mode].append(r)

        # --- Left: Latency comparison for select configs ---
        ax = axes[0]
        target_configs = [(256, 3), (512, 5), (1024, 10)]
        impl_modes = ['Single CPU', 'Single GPU', 'CUDA C++']
        # Add best OpenMP if available
        openmp_results = [r for r in self.results if 'OpenMP' in r['mode']]
        if openmp_results:
            impl_modes.append('OpenMP (best)')

        x = np.arange(len(target_configs))
        width = 0.18
        for i, mode in enumerate(impl_modes):
            latencies = []
            for size_w, depth in target_configs:
                candidates = [r for r in self.results
                             if r['image_size'][0] == size_w and r['depth_count'] == depth]
                if mode == 'OpenMP (best)':
                    matches = [r for r in candidates if 'OpenMP' in r['mode']]
                else:
                    matches = [r for r in candidates if r['mode'] == mode]
                if matches:
                    latencies.append(min(r['mean_latency'] for r in matches))
                else:
                    latencies.append(0)
            ax.bar(x + i * width, latencies, width, label=mode)

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Implementation Comparison: Latency')
        ax.set_xticks(x + width * (len(impl_modes) - 1) / 2)
        ax.set_xticklabels([f'{s}x{s}x{d}' for s, d in target_configs])
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

        # --- Right: Speedup over Single CPU ---
        ax2 = axes[1]
        for i, mode in enumerate(impl_modes[1:], 1):  # Skip Single CPU (baseline)
            speedups = []
            for size_w, depth in target_configs:
                candidates = [r for r in self.results
                             if r['image_size'][0] == size_w and r['depth_count'] == depth]
                cpu_matches = [r for r in candidates if r['mode'] == 'Single CPU']
                if mode == 'OpenMP (best)':
                    other_matches = [r for r in candidates if 'OpenMP' in r['mode']]
                else:
                    other_matches = [r for r in candidates if r['mode'] == mode]
                if cpu_matches and other_matches:
                    cpu_lat = min(r['mean_latency'] for r in cpu_matches)
                    other_lat = min(r['mean_latency'] for r in other_matches)
                    speedups.append(cpu_lat / other_lat if other_lat > 0 else 0)
                else:
                    speedups.append(0)
            ax2.bar(x + (i - 1) * width, speedups, width, label=mode)

        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Speedup over Single CPU')
        ax2.set_title('Speedup Comparison')
        ax2.set_xticks(x + width * (len(impl_modes) - 2) / 2)
        ax2.set_xticklabels([f'{s}x{s}x{d}' for s, d in target_configs])
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_thread_scaling(self):
        """Thread scaling plot with Amdahl's law overlay."""
        openmp_results = [r for r in self.results if 'OpenMP' in r['mode']]
        if not openmp_results:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Extract thread counts and group
        thread_data = {}  # (image_size, depth) -> {threads: latency}
        for r in openmp_results:
            m = re.search(r'(\d+)-threads', r['mode'])
            if not m:
                continue
            threads = int(m.group(1))
            key = (r['image_size'][0], r['depth_count'])
            if key not in thread_data:
                thread_data[key] = {}
            thread_data[key][threads] = r['mean_latency']

        # --- Left: Speedup vs Thread Count ---
        ax = axes[0]
        for (sz, depth), td in sorted(thread_data.items()):
            threads_sorted = sorted(td.keys())
            base = td.get(1, td[min(td.keys())])
            speedups = [base / td[t] for t in threads_sorted]
            ax.plot(threads_sorted, speedups, 'o-', label=f'{sz}x{sz}x{depth}', markersize=4)

        # Amdahl's law overlay (estimate parallel fraction)
        t_range = np.array([1, 2, 4, 8, 16, 32, 64])
        for p_frac in [0.90, 0.95, 0.99]:
            amdahl = 1 / ((1 - p_frac) + p_frac / t_range)
            ax.plot(t_range, amdahl, '--', alpha=0.4, color='gray',
                   label=f"Amdahl p={p_frac}" if p_frac == 0.95 else None)

        ax.plot(t_range, t_range, ':', alpha=0.3, color='black', label='Linear')
        ax.set_xlabel('Thread Count')
        ax.set_ylabel('Speedup')
        ax.set_title('Thread Scaling (Speedup)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

        # --- Right: Efficiency vs Thread Count ---
        ax2 = axes[1]
        for (sz, depth), td in sorted(thread_data.items()):
            threads_sorted = sorted(td.keys())
            base = td.get(1, td[min(td.keys())])
            efficiency = [base / (td[t] * t) for t in threads_sorted]
            ax2.plot(threads_sorted, efficiency, 'o-', label=f'{sz}x{sz}x{depth}', markersize=4)

        ax2.axhline(y=1.0, color='black', linestyle=':', alpha=0.3, label='Ideal')
        ax2.set_xlabel('Thread Count')
        ax2.set_ylabel('Parallel Efficiency')
        ax2.set_title('Thread Scaling (Efficiency)')
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_ylim(0, 1.1)

        plt.tight_layout()
        return fig

    def plot_latency_cdfs(self):
        """Latency CDF per mode with 2.5ms (400 Hz) budget line."""
        fig, ax = plt.subplots(figsize=(10, 6))

        mode_colors = {}
        cmap = plt.cm.tab10
        modes_seen = sorted(set(r['mode'] for r in self.results
                               if 'OpenMP' not in r['mode']))  # Skip per-thread OpenMP
        # Add summary OpenMP modes
        openmp_threads = set()
        for r in self.results:
            m = re.search(r'(\d+)-threads', r['mode'])
            if m:
                openmp_threads.add(int(m.group(1)))
        if openmp_threads:
            modes_seen.append(f'OpenMP {max(openmp_threads)}-threads')

        for i, mode in enumerate(modes_seen):
            mode_colors[mode] = cmap(i % 10)

        for mode in modes_seen:
            mode_results = [r for r in self.results if r['mode'] == mode]
            if not mode_results:
                continue
            all_latencies = sorted([r['mean_latency'] for r in mode_results])
            cdf = np.arange(1, len(all_latencies) + 1) / len(all_latencies)
            ax.plot(all_latencies, cdf, label=mode, color=mode_colors[mode], linewidth=1.5)

        # 400 Hz budget line (2.5 ms)
        ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2,
                  label='2.5 ms (400 Hz budget)')

        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('CDF')
        ax.set_title('Latency CDF by Implementation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlim(0.01, 10000)

        plt.tight_layout()
        return fig

    def plot_gpu_memory_utilization(self):
        """GPU VRAM usage vs problem size."""
        gpu_results = [r for r in self.results
                      if ('GPU' in r['mode'] or 'CUDA' in r['mode'])
                      and r.get('gpu_memory_used') and r['gpu_memory_used'] > 0]
        if not gpu_results:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Problem size = H * W * depth
        sizes = []
        mem_usage = []
        labels = []
        for r in gpu_results:
            h, w = r['image_size']
            prob_size = h * w * r['depth_count'] * r['batch_size']
            sizes.append(prob_size)
            mem_usage.append(r['gpu_memory_used'])
            labels.append(r['mode'])

        mode_set = sorted(set(labels))
        for mode in mode_set:
            mask = [l == mode for l in labels]
            s = [sizes[i] for i in range(len(sizes)) if mask[i]]
            m = [mem_usage[i] for i in range(len(mem_usage)) if mask[i]]
            ax.scatter(s, m, label=mode, alpha=0.6, s=30)

        # VRAM budget line
        ax.axhline(y=32768, color='red', linestyle='--', alpha=0.5,
                  label='32 GB VRAM limit')

        ax.set_xlabel('Problem Size (total pixels)')
        ax.set_ylabel('GPU Memory (MB)')
        ax.set_title('GPU Memory Utilization vs Problem Size')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        plt.tight_layout()
        return fig

    def generate_performance_report(self, output_dir: str,
                                    ncu_csv: Optional[str] = None):
        """Generate comprehensive performance report with all plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Original plots
        roofline_fig = self.create_roofline_plot('both')
        roofline_fig.savefig(output_path / 'roofline_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(roofline_fig)

        scaling_fig = self.analyze_scaling_efficiency()
        scaling_fig.savefig(output_path / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(scaling_fig)

        # Multi-level roofline
        ml_fig = self.create_multilevel_roofline(ncu_csv)
        ml_fig.savefig(output_path / 'multilevel_roofline.png', dpi=300, bbox_inches='tight')
        plt.close(ml_fig)

        # Phase 7: Enhanced visualizations
        impl_fig = self.plot_implementation_comparison()
        impl_fig.savefig(output_path / 'implementation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(impl_fig)

        thread_fig = self.plot_thread_scaling()
        if thread_fig:
            thread_fig.savefig(output_path / 'thread_scaling.png', dpi=300, bbox_inches='tight')
            plt.close(thread_fig)

        cdf_fig = self.plot_latency_cdfs()
        cdf_fig.savefig(output_path / 'latency_cdfs.png', dpi=300, bbox_inches='tight')
        plt.close(cdf_fig)

        mem_fig = self.plot_gpu_memory_utilization()
        if mem_fig:
            mem_fig.savefig(output_path / 'gpu_memory_utilization.png', dpi=300, bbox_inches='tight')
            plt.close(mem_fig)

        # Text report
        report_file = output_path / 'performance_report.txt'
        with open(report_file, 'w') as f:
            self._write_text_report(f)

        print(f"Performance report generated in {output_path}")
        for name in ['roofline_analysis', 'scaling_analysis', 'multilevel_roofline',
                      'implementation_comparison', 'thread_scaling', 'latency_cdfs',
                      'gpu_memory_utilization', 'performance_report']:
            ext = '.txt' if name == 'performance_report' else '.png'
            p = output_path / (name + ext)
            if p.exists():
                print(f"  - {name}{ext}")


def main():
    parser = argparse.ArgumentParser(description="Roofline Analysis for Holographic Reconstruction")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", default="roofline_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--device", choices=['cpu', 'gpu', 'both'], default='both',
                       help="Device type for roofline analysis")
    parser.add_argument("--ncu-csv", default=None,
                       help="Path to Nsight Compute CSV export for empirical roofline")

    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"Error: Results file {args.results_file} not found")
        return

    analyzer = RooflineAnalyzer(args.results_file)
    analyzer.generate_performance_report(args.output_dir, ncu_csv=args.ncu_csv)

if __name__ == "__main__":
    main()