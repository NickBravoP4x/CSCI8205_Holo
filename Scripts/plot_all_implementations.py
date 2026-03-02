#!/usr/bin/env python3
"""
Compare all four holographic reconstruction implementations:
  1. Python CPU  (PyTorch, single-threaded)
  2. Python GPU  (PyTorch CUDA, RTX 5090)
  3. C++ CPU     (OpenMP + FFTW, best thread count)
  4. C++ GPU     (CUDA + cuFFT, RTX 5090)

Generates:
  1. Grouped bar chart: latency across all configs
  2. Speedup over Python CPU baseline
  3. Latency vs problem size (log-log scaling)
  4. Unified roofline (CPU + GPU panels, all 4 implementations)

Usage:
    python Scripts/plot_all_implementations.py
"""

import os
import re
import json
from pathlib import Path

import numpy as np

# Display backend
def setup_display_backend():
    if os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    elif os.environ.get('WAYLAND_DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'wayland'
    else:
        import matplotlib
        matplotlib.use('Agg')
        return
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1, 1))
        plt.close(fig)
    except Exception:
        import matplotlib
        matplotlib.use('Agg')

setup_display_backend()
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "Results"
OUTPUT_DIR = RESULTS_DIR / "Implementation_Comparison"

SIZES = [128, 256, 512, 1024]
DEPTHS = [1, 3, 5, 10, 20]
THREADS = [1, 2, 4, 8, 16, 32]

# Hardware specs for roofline (Threadripper PRO 7975WX + RTX 5090)
CPU_SPECS = {
    'peak_compute': 2700e9,      # ~2.7 TFLOPS FP32 (32c × 2 FMA × 16 SP/AVX-512 × 5.3 GHz)
    'memory_bandwidth': 150e9,   # ~150 GB/s (8-channel DDR5-5200)
}
GPU_SPECS = {
    'peak_compute': 105e12,      # ~105 TFLOPS FP32 (RTX 5090)
    'memory_bandwidth': 1790e9,  # ~1790 GB/s (GDDR7)
}

# Implementation display config
IMPL_NAMES = ['Python CPU', 'Python GPU', 'C++ OpenMP', 'C++ CUDA']
IMPL_COLORS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
IMPL_MARKERS = ['s', 'D', '^', 'o']


# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_python_results(path: Path) -> dict:
    """Parse Python JSON benchmark → {(size, depth): {'cpu': ms, 'gpu': ms}}"""
    with open(path) as f:
        data = json.load(f)

    out = {}
    for r in data['results']:
        if r['batch_size'] != 1:
            continue
        size = r['image_size'][0]
        depth = r['depth_count']
        key = (size, depth)
        if key not in out:
            out[key] = {}

        if r['mode'] == 'Single CPU':
            out[key]['cpu'] = r['mean_latency']
            out[key]['cpu_flops'] = r.get('flops_per_second')
        elif r['mode'] == 'Single GPU':
            out[key]['gpu'] = r['mean_latency']
            out[key]['gpu_flops'] = r.get('flops_per_second')
    return out


def parse_cuda_results(path: Path) -> dict:
    """Parse cuda_benchmark_full.txt → {(size, depth): {'mean': ms, ...}}"""
    data = {}
    text = path.read_text()
    blocks = re.split(r'=== CUDA (\d+)x\d+ x (\d+) planes ===', text)
    for i in range(1, len(blocks) - 2, 3):
        size = int(blocks[i])
        depth = int(blocks[i + 1])
        body = blocks[i + 2]
        entry = {}
        for line in body.splitlines():
            line = line.strip()
            if line.startswith('Mean:'):
                entry['mean'] = float(re.search(r'([\d.]+)', line).group(1))
            elif line.startswith('Median:'):
                entry['median'] = float(re.search(r'([\d.]+)', line).group(1))
            elif line.startswith('P95:'):
                entry['p95'] = float(re.search(r'([\d.]+)', line).group(1))
            elif line.startswith('Throughput:'):
                entry['throughput'] = float(re.search(r'([\d.]+)', line).group(1))
        if entry:
            data[(size, depth)] = entry
    return data


def parse_cpu_results(path: Path) -> dict:
    """Parse cpu_benchmark_full.txt → {(size, depth, threads): {'mean': ms, ...}}"""
    data = {}
    text = path.read_text()
    blocks = re.split(r'=== CPU (\d+)x\d+ x (\d+) planes \(thread sweep\) ===', text)
    for i in range(1, len(blocks) - 2, 3):
        size = int(blocks[i])
        depth = int(blocks[i + 1])
        body = blocks[i + 2]
        for line in body.splitlines():
            m = re.match(
                r'Threads:\s*(\d+)\s*\|\s*Mean:\s*([\d.]+)\s*ms\s*\|\s*Median:\s*([\d.]+)\s*ms\s*'
                r'\|\s*P95:\s*([\d.]+)\s*ms\s*\|\s*Throughput:\s*([\d.]+)',
                line.strip()
            )
            if m:
                threads = int(m.group(1))
                data[(size, depth, threads)] = {
                    'mean': float(m.group(2)),
                    'median': float(m.group(3)),
                    'p95': float(m.group(4)),
                    'throughput': float(m.group(5)),
                }
    return data


def best_cpu_latency(cpu_data: dict, size: int, depth: int) -> float:
    """Return lowest mean latency across all thread counts for a given config."""
    best = float('inf')
    for t in THREADS:
        val = cpu_data.get((size, depth, t), {}).get('mean', float('inf'))
        if val < best:
            best = val
    return best if best < float('inf') else 0.0


def best_cpu_threads(cpu_data: dict, size: int, depth: int) -> int:
    """Return thread count that achieved lowest latency."""
    best_t, best_v = 1, float('inf')
    for t in THREADS:
        val = cpu_data.get((size, depth, t), {}).get('mean', float('inf'))
        if val < best_v:
            best_v = val
            best_t = t
    return best_t


# ── Latency collector ────────────────────────────────────────────────────────

def collect_latencies(py_data, cuda_data, cpp_cpu_data):
    """Build {(size, depth): [py_cpu, py_gpu, cpp_cpu, cpp_cuda]} latency arrays."""
    results = {}
    for size in SIZES:
        for depth in DEPTHS:
            py = py_data.get((size, depth), {})
            py_cpu = py.get('cpu', 0)
            py_gpu = py.get('gpu', 0)
            cpp_cpu = best_cpu_latency(cpp_cpu_data, size, depth)
            cpp_cuda = cuda_data.get((size, depth), {}).get('mean', 0)
            results[(size, depth)] = [py_cpu, py_gpu, cpp_cpu, cpp_cuda]
    return results


# ── FLOPs estimator ──────────────────────────────────────────────────────────

def estimate_flops(h: int, w: int, depth: int) -> float:
    """Estimate FLOPs for holographic reconstruction (angular spectrum method).

    Dominant cost: 2D FFT forward + inverse per depth plane.
    FFT: ~5 N log2(N) real ops per complex 1D FFT of length N.
    2D FFT of HxW: 5*(H*log2(H)*W + W*log2(W)*H) per direction.
    """
    fft_2d_ops = 5 * (h * np.log2(h) * w + w * np.log2(w) * h)
    per_plane = 2 * fft_2d_ops  # forward + inverse
    per_plane += h * w * 20     # element-wise: pad, multiply Hz, phase correct, abs²
    return per_plane * depth


def calc_arithmetic_intensity(h: int, w: int, depth: int) -> float:
    """Operational intensity (FLOPS / byte transferred)."""
    flops = estimate_flops(h, w, depth)
    # Memory: input (float32) + output (float32 per plane) + intermediate complex
    mem_bytes = (h * w * 4 +                    # input hologram
                 h * w * depth * 4 +             # output intensity per plane
                 2 * h * w * depth * 8)          # intermediate complex FFT data
    return flops / mem_bytes


# ── Figure 1: Grouped bar chart — latency comparison ────────────────────────

def plot_latency_bars(lat: dict) -> plt.Figure:
    """Grouped bars: 4 implementations for each (size, depth) config."""
    # Select representative configs (not all 20 — too crowded)
    configs = [(s, d) for s in SIZES for d in [1, 5, 20]]
    labels = [f'{s}×{s}\n{d}p' for s, d in configs]

    n = len(configs)
    x = np.arange(n)
    width = 0.20

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (name, color) in enumerate(zip(IMPL_NAMES, IMPL_COLORS)):
        vals = [lat[(s, d)][i] for s, d in configs]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.88)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.05,
                        f'{v:.2f}' if v < 10 else f'{v:.1f}',
                        ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_yscale('log')
    ax.set_title('Holographic Reconstruction Latency — All Implementations',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # 2.5 ms budget line
    ax.axhline(y=2.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(n - 0.5, 2.7, '2.5 ms (400 Hz)', ha='right', va='bottom',
            color='red', fontsize=9, fontstyle='italic')

    fig.tight_layout()
    return fig


# ── Figure 2: Speedup over Python CPU ───────────────────────────────────────

def plot_speedup(lat: dict) -> plt.Figure:
    """Speedup of each implementation over Python CPU (single-threaded)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for si, (size, ax) in enumerate(zip(SIZES, axes)):
        x = np.arange(len(DEPTHS))
        width = 0.22
        for i in range(1, 4):  # skip Python CPU (baseline = 1)
            speedups = []
            for depth in DEPTHS:
                py_cpu = lat[(size, depth)][0]
                impl_val = lat[(size, depth)][i]
                speedups.append(py_cpu / impl_val if impl_val > 0 else 0)
            offset = (i - 2) * width
            bars = ax.bar(x + offset, speedups, width,
                          label=IMPL_NAMES[i], color=IMPL_COLORS[i], alpha=0.88)
            for bar, s in zip(bars, speedups):
                if s > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f'{s:.0f}×', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}p' for d in DEPTHS])
        ax.set_ylabel('Speedup')
        ax.set_title(f'{size}×{size}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

    fig.suptitle('Speedup over Python CPU (Single-Threaded)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ── Figure 3: Latency vs problem size (log-log) ─────────────────────────────

def plot_scaling(lat: dict) -> plt.Figure:
    """Log-log latency vs total pixels (H×W×depth) for each implementation."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (name, color, marker) in enumerate(zip(IMPL_NAMES, IMPL_COLORS, IMPL_MARKERS)):
        prob_sizes = []
        latencies = []
        for size in SIZES:
            for depth in DEPTHS:
                prob = size * size * depth
                val = lat[(size, depth)][i]
                if val > 0:
                    prob_sizes.append(prob)
                    latencies.append(val)
        ax.scatter(prob_sizes, latencies, color=color, marker=marker,
                   s=50, alpha=0.7, label=name, zorder=5)
        # Fit a log-log trend line
        if len(prob_sizes) > 2:
            log_x = np.log10(prob_sizes)
            log_y = np.log10(latencies)
            coeffs = np.polyfit(log_x, log_y, 1)
            x_fit = np.logspace(np.log10(min(prob_sizes)), np.log10(max(prob_sizes)), 100)
            y_fit = 10 ** np.polyval(coeffs, np.log10(x_fit))
            ax.plot(x_fit, y_fit, color=color, linewidth=1.5, alpha=0.5, linestyle='--')

    # 2.5 ms budget
    ax.axhline(y=2.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1.5e7, 2.7, '2.5 ms (400 Hz)', ha='right', color='red',
            fontsize=9, fontstyle='italic')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Problem Size (H × W × Depth)')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Latency Scaling with Problem Size', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Figure 4: Unified Roofline ──────────────────────────────────────────────

def plot_roofline(py_data, cuda_data, cpp_cpu_data, lat: dict) -> plt.Figure:
    """Two-panel roofline: CPU (left) and GPU (right) with all implementations."""
    fig, (cpu_ax, gpu_ax) = plt.subplots(1, 2, figsize=(16, 7))
    oi_range = np.logspace(-2, 3, 1000)

    # ── CPU panel ──
    # Multi-level bandwidth ceilings
    cpu_bw_levels = {
        'L1 (~2 TB/s)':   2000e9,
        'L2 (~800 GB/s)':  800e9,
        'L3 (~400 GB/s)':  400e9,
        'DRAM (150 GB/s)': CPU_SPECS['memory_bandwidth'],
    }
    bw_colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#1f77b4']
    for (label, bw), c in zip(cpu_bw_levels.items(), bw_colors):
        cpu_ax.loglog(oi_range, bw * oi_range, '--', color=c, alpha=0.5,
                      linewidth=1, label=label)

    cpu_ax.axhline(y=CPU_SPECS['peak_compute'], color='red', linestyle='-',
                   linewidth=2, alpha=0.7,
                   label=f'Peak FP32 ({CPU_SPECS["peak_compute"]/1e12:.1f} TF)')
    roofline_cpu = np.minimum(CPU_SPECS['memory_bandwidth'] * oi_range,
                              CPU_SPECS['peak_compute'])
    cpu_ax.loglog(oi_range, roofline_cpu, 'k-', linewidth=2.5, label='DRAM Roofline')

    # Plot Python CPU and C++ OpenMP points
    for impl_idx, name, color, marker in [
        (0, 'Python CPU', IMPL_COLORS[0], IMPL_MARKERS[0]),
        (2, 'C++ OpenMP', IMPL_COLORS[2], IMPL_MARKERS[2]),
    ]:
        ois, perfs = [], []
        for size in SIZES:
            for depth in DEPTHS:
                latency_ms = lat[(size, depth)][impl_idx]
                if latency_ms <= 0:
                    continue
                oi = calc_arithmetic_intensity(size, size, depth)
                flops = estimate_flops(size, size, depth)
                achieved = flops / (latency_ms / 1000)  # FLOPS/s
                ois.append(oi)
                perfs.append(achieved)
        cpu_ax.scatter(ois, perfs, color=color, marker=marker, s=60,
                       alpha=0.7, label=name, edgecolors='black', linewidths=0.5,
                       zorder=5)

    cpu_ax.set_xlabel('Arithmetic Intensity (FLOPS/byte)')
    cpu_ax.set_ylabel('Performance (FLOPS/s)')
    cpu_ax.set_title('CPU Roofline — Threadripper PRO 7975WX')
    cpu_ax.legend(fontsize=7, loc='lower right')
    cpu_ax.grid(True, alpha=0.3)
    cpu_ax.set_ylim(1e7, 1e13)
    cpu_ax.set_xlim(0.1, 100)

    # ── GPU panel ──
    gpu_bw_levels = {
        'L1/Shared (~12 TB/s)': 12000e9,
        'L2 (~6 TB/s)':         6000e9,
        'GDDR7 (1790 GB/s)':    GPU_SPECS['memory_bandwidth'],
    }
    gpu_bw_colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    for (label, bw), c in zip(gpu_bw_levels.items(), gpu_bw_colors):
        gpu_ax.loglog(oi_range, bw * oi_range, '--', color=c, alpha=0.5,
                      linewidth=1, label=label)

    gpu_ax.axhline(y=GPU_SPECS['peak_compute'], color='red', linestyle='-',
                   linewidth=2, alpha=0.7,
                   label=f'Peak FP32 ({GPU_SPECS["peak_compute"]/1e12:.0f} TF)')
    roofline_gpu = np.minimum(GPU_SPECS['memory_bandwidth'] * oi_range,
                              GPU_SPECS['peak_compute'])
    gpu_ax.loglog(oi_range, roofline_gpu, 'k-', linewidth=2.5, label='GDDR7 Roofline')

    # Plot Python GPU and C++ CUDA points
    for impl_idx, name, color, marker in [
        (1, 'Python GPU', IMPL_COLORS[1], IMPL_MARKERS[1]),
        (3, 'C++ CUDA',   IMPL_COLORS[3], IMPL_MARKERS[3]),
    ]:
        ois, perfs = [], []
        for size in SIZES:
            for depth in DEPTHS:
                latency_ms = lat[(size, depth)][impl_idx]
                if latency_ms <= 0:
                    continue
                oi = calc_arithmetic_intensity(size, size, depth)
                flops = estimate_flops(size, size, depth)
                achieved = flops / (latency_ms / 1000)
                ois.append(oi)
                perfs.append(achieved)
        gpu_ax.scatter(ois, perfs, color=color, marker=marker, s=60,
                       alpha=0.7, label=name, edgecolors='black', linewidths=0.5,
                       zorder=5)

    gpu_ax.set_xlabel('Arithmetic Intensity (FLOPS/byte)')
    gpu_ax.set_ylabel('Performance (FLOPS/s)')
    gpu_ax.set_title('GPU Roofline — RTX 5090')
    gpu_ax.legend(fontsize=7, loc='lower right')
    gpu_ax.grid(True, alpha=0.3)
    gpu_ax.set_ylim(1e8, 1e14)
    gpu_ax.set_xlim(0.1, 100)

    fig.suptitle('Multi-Level Roofline Analysis — All Implementations',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ── Figure 5: Summary table as figure ───────────────────────────────────────

def plot_summary_table(lat: dict, cpp_cpu_data: dict) -> plt.Figure:
    """Render a publication-ready summary table as a figure."""
    configs = [(s, d) for s in [256, 512, 1024] for d in [1, 5, 20]]

    col_labels = ['Config', 'Py CPU\n(ms)', 'Py GPU\n(ms)', 'C++ OMP\n(ms)',
                  'C++ CUDA\n(ms)', 'Best\nOMP Thr', 'GPU vs\nPy CPU', 'CUDA vs\nOMP']
    cell_text = []
    for s, d in configs:
        vals = lat[(s, d)]
        best_t = best_cpu_threads(cpp_cpu_data, s, d)
        gpu_speedup = vals[0] / vals[1] if vals[1] > 0 else 0
        cuda_vs_omp = vals[2] / vals[3] if vals[3] > 0 else 0
        cell_text.append([
            f'{s}×{s} × {d}p',
            f'{vals[0]:.2f}',
            f'{vals[1]:.3f}',
            f'{vals[2]:.2f}',
            f'{vals[3]:.3f}',
            f'{best_t}',
            f'{gpu_speedup:.0f}×',
            f'{cuda_vs_omp:.1f}×',
        ])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4c72b0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(len(cell_text)):
        bg = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)

    ax.set_title('Performance Summary — Holographic Reconstruction',
                 fontsize=13, fontweight='bold', pad=20)
    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Locate all data files
    py_json = RESULTS_DIR / 'Week1-2_Python_Baseline_5090' / 'benchmark_results' / 'benchmark_results.json'
    cuda_txt = RESULTS_DIR / 'Reconstruction_Benchmark' / 'cuda_benchmark_full.txt'
    cpp_cpu_txt = RESULTS_DIR / 'Reconstruction_Benchmark' / 'cpu_benchmark_full.txt'

    missing = []
    for p in [py_json, cuda_txt, cpp_cpu_txt]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: Missing benchmark files:")
        for m in missing:
            print(f"  {m}")
        return

    print("Parsing results...")
    py_data = parse_python_results(py_json)
    cuda_data = parse_cuda_results(cuda_txt)
    cpp_cpu_data = parse_cpu_results(cpp_cpu_txt)
    print(f"  Python:   {len(py_data)} configs")
    print(f"  C++ CUDA: {len(cuda_data)} configs")
    print(f"  C++ CPU:  {len(cpp_cpu_data)} configs")

    lat = collect_latencies(py_data, cuda_data, cpp_cpu_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Latency bars
    fig = plot_latency_bars(lat)
    p = OUTPUT_DIR / "latency_comparison_all.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # Figure 2: Speedup
    fig = plot_speedup(lat)
    p = OUTPUT_DIR / "speedup_over_python_cpu.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # Figure 3: Scaling
    fig = plot_scaling(lat)
    p = OUTPUT_DIR / "latency_scaling.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # Figure 4: Roofline
    fig = plot_roofline(py_data, cuda_data, cpp_cpu_data, lat)
    p = OUTPUT_DIR / "roofline_all_implementations.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # Figure 5: Summary table
    fig = plot_summary_table(lat, cpp_cpu_data)
    p = OUTPUT_DIR / "summary_table.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # Print summary to terminal
    print("\n" + "=" * 100)
    print("ALL IMPLEMENTATIONS — SUMMARY")
    print("=" * 100)
    header = (f"{'Config':<20s} {'Py CPU (ms)':>12s} {'Py GPU (ms)':>12s} "
              f"{'C++ OMP (ms)':>13s} {'C++ CUDA (ms)':>14s} {'GPU/PyCPU':>10s} {'CUDA/OMP':>10s}")
    print(header)
    print("-" * len(header))
    for size in SIZES:
        for depth in DEPTHS:
            v = lat[(size, depth)]
            gpu_sp = v[0] / v[1] if v[1] > 0 else 0
            cuda_sp = v[2] / v[3] if v[3] > 0 else 0
            print(f"{size}x{size} x {depth:2d}p     "
                  f"{v[0]:12.3f} {v[1]:12.3f} {v[2]:13.3f} {v[3]:14.3f} "
                  f"{gpu_sp:9.1f}× {cuda_sp:9.1f}×")


if __name__ == '__main__':
    main()
