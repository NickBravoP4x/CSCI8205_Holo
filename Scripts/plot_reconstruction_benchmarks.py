#!/usr/bin/env python3
"""
Plot CUDA and OpenMP reconstruction benchmark results.

Generates:
  1. CUDA latency heatmap (size x depth)
  2. OpenMP thread scaling curves (speedup vs threads)
  3. GPU vs CPU speedup comparison
  4. Absolute latency comparison (GPU vs best CPU)

Reads from Results/Reconstruction_Benchmark/{cuda,cpu}_benchmark_full.txt

Usage:
    python Scripts/plot_reconstruction_benchmarks.py
"""

import os
import re
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results" / "Reconstruction_Benchmark"
OUTPUT_DIR = RESULTS_DIR / "figures"

SIZES = [128, 256, 512, 1024]
DEPTHS = [1, 3, 5, 10, 20]
THREADS = [1, 2, 4, 8, 16, 32]  # skip 64 (oversubscribed, always worse)


# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_cuda_results(path: Path) -> dict:
    """Parse cuda_benchmark_full.txt into {(size, depth): {'mean': ..., 'median': ..., ...}}"""
    data = {}
    text = path.read_text()
    blocks = re.split(r'=== CUDA (\d+)x\d+ x (\d+) planes ===', text)
    # blocks: ['header', '128', '1', 'result_text', '128', '3', 'result_text', ...]
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
            elif line.startswith('P99:'):
                entry['p99'] = float(re.search(r'([\d.]+)', line).group(1))
            elif line.startswith('Throughput:'):
                entry['throughput'] = float(re.search(r'([\d.]+)', line).group(1))
        if entry:
            data[(size, depth)] = entry
    return data


def parse_cpu_results(path: Path) -> dict:
    """Parse cpu_benchmark_full.txt into {(size, depth, threads): {'mean': ..., ...}}"""
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


# ── Figure 1: CUDA Latency Heatmap ──────────────────────────────────────────

def plot_cuda_heatmap(cuda_data: dict) -> plt.Figure:
    matrix = np.zeros((len(SIZES), len(DEPTHS)))
    for si, size in enumerate(SIZES):
        for di, depth in enumerate(DEPTHS):
            matrix[si, di] = cuda_data.get((size, depth), {}).get('mean', 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(DEPTHS)))
    ax.set_xticklabels(DEPTHS)
    ax.set_yticks(range(len(SIZES)))
    ax.set_yticklabels([f'{s}x{s}' for s in SIZES])
    ax.set_xlabel('Depth Planes')
    ax.set_ylabel('Image Size')
    ax.set_title('CUDA Reconstruction Latency (ms) — RTX 5090')

    # Annotate cells
    for si in range(len(SIZES)):
        for di in range(len(DEPTHS)):
            val = matrix[si, di]
            color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(di, si, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, label='Mean Latency (ms)')
    fig.tight_layout()
    return fig


# ── Figure 2: OpenMP Thread Scaling ──────────────────────────────────────────

def plot_thread_scaling(cpu_data: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = sns.color_palette("husl", len(DEPTHS))

    for si, (size, ax) in enumerate(zip(SIZES, axes)):
        for di, depth in enumerate(DEPTHS):
            speedups = []
            base = cpu_data.get((size, depth, 1), {}).get('mean', None)
            if base is None:
                continue
            for t in THREADS:
                val = cpu_data.get((size, depth, t), {}).get('mean', None)
                if val is not None:
                    speedups.append(base / val)
                else:
                    speedups.append(None)

            valid_t = [t for t, s in zip(THREADS, speedups) if s is not None]
            valid_s = [s for s in speedups if s is not None]
            ax.plot(valid_t, valid_s, 'o-', label=f'{depth} planes',
                    color=colors[di], linewidth=2, markersize=5)

        # Ideal scaling reference
        ax.plot(THREADS, THREADS, '--', color='gray', alpha=0.4, label='Ideal')

        ax.set_title(f'{size}x{size}')
        ax.set_xlabel('Threads')
        ax.set_ylabel('Speedup vs 1 Thread')
        ax.set_xscale('log', base=2)
        ax.set_xticks(THREADS)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('OpenMP Thread Scaling — AMD Threadripper PRO 7975WX (32 cores)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ── Figure 3: GPU vs CPU Speedup ─────────────────────────────────────────────

def plot_gpu_vs_cpu(cuda_data: dict, cpu_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(DEPTHS))
    width = 0.18
    colors = sns.color_palette("muted", len(SIZES))

    for si, size in enumerate(SIZES):
        speedups = []
        for depth in DEPTHS:
            gpu_ms = cuda_data.get((size, depth), {}).get('mean', None)
            # Best CPU = min mean across all thread counts
            best_cpu = None
            for t in THREADS:
                val = cpu_data.get((size, depth, t), {}).get('mean', None)
                if val is not None:
                    if best_cpu is None or val < best_cpu:
                        best_cpu = val
            if gpu_ms and best_cpu:
                speedups.append(best_cpu / gpu_ms)
            else:
                speedups.append(0)

        offset = (si - len(SIZES) / 2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width, label=f'{size}x{size}',
                      color=colors[si], alpha=0.85)
        for bar, s in zip(bars, speedups):
            if s > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{s:.0f}x', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{d} planes' for d in DEPTHS])
    ax.set_ylabel('GPU Speedup over Best CPU (32-thread)')
    ax.set_title('CUDA (RTX 5090) vs OpenMP+FFTW (Threadripper 7975WX, best thread count)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    return fig


# ── Figure 4: Absolute Latency Comparison ────────────────────────────────────

def plot_latency_comparison(cuda_data: dict, cpu_data: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: line chart by depth for 512x512
    ax = axes[0]
    target_size = 512
    gpu_latencies = [cuda_data.get((target_size, d), {}).get('mean', 0) for d in DEPTHS]
    cpu_1t = [cpu_data.get((target_size, d, 1), {}).get('mean', 0) for d in DEPTHS]
    cpu_32t = [cpu_data.get((target_size, d, 32), {}).get('mean', 0) for d in DEPTHS]

    ax.plot(DEPTHS, gpu_latencies, 'o-', linewidth=2.5, markersize=7, label='CUDA (RTX 5090)')
    ax.plot(DEPTHS, cpu_1t, 's--', linewidth=2, markersize=6, label='CPU 1 thread')
    ax.plot(DEPTHS, cpu_32t, '^--', linewidth=2, markersize=6, label='CPU 32 threads')
    ax.set_xlabel('Depth Planes')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title(f'Latency vs Depth — {target_size}x{target_size}')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: line chart by size for 3 planes
    ax = axes[1]
    target_depth = 3
    gpu_latencies = [cuda_data.get((s, target_depth), {}).get('mean', 0) for s in SIZES]
    cpu_1t = [cpu_data.get((s, target_depth, 1), {}).get('mean', 0) for s in SIZES]
    cpu_32t = [cpu_data.get((s, target_depth, 32), {}).get('mean', 0) for s in SIZES]
    size_labels = [f'{s}' for s in SIZES]

    ax.plot(size_labels, gpu_latencies, 'o-', linewidth=2.5, markersize=7, label='CUDA (RTX 5090)')
    ax.plot(size_labels, cpu_1t, 's--', linewidth=2, markersize=6, label='CPU 1 thread')
    ax.plot(size_labels, cpu_32t, '^--', linewidth=2, markersize=6, label='CPU 32 threads')
    ax.set_xlabel('Image Size (NxN)')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title(f'Latency vs Image Size — {target_depth} planes')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('GPU vs CPU Reconstruction Latency', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cuda_path = RESULTS_DIR / 'cuda_benchmark_full.txt'
    cpu_path = RESULTS_DIR / 'cpu_benchmark_full.txt'

    if not cuda_path.exists() or not cpu_path.exists():
        print(f"ERROR: Missing benchmark results in {RESULTS_DIR}")
        print(f"  Run: bash Scripts/run_cpp_benchmarks.sh")
        return

    print("Parsing results...")
    cuda_data = parse_cuda_results(cuda_path)
    cpu_data = parse_cpu_results(cpu_path)
    print(f"  CUDA: {len(cuda_data)} configs")
    print(f"  CPU:  {len(cpu_data)} configs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1
    fig = plot_cuda_heatmap(cuda_data)
    p = OUTPUT_DIR / "cuda_latency_heatmap.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # Figure 2
    fig = plot_thread_scaling(cpu_data)
    p = OUTPUT_DIR / "openmp_thread_scaling.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # Figure 3
    fig = plot_gpu_vs_cpu(cuda_data, cpu_data)
    p = OUTPUT_DIR / "gpu_vs_cpu_speedup.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # Figure 4
    fig = plot_latency_comparison(cuda_data, cpu_data)
    p = OUTPUT_DIR / "latency_comparison.png"
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # Print summary table
    print("\n" + "=" * 80)
    print("CUDA vs Best CPU Summary")
    print("=" * 80)
    header = f"{'Config':<20s} {'CUDA (ms)':>10s} {'CPU-1T (ms)':>12s} {'CPU-best (ms)':>14s} {'GPU Speedup':>12s}"
    print(header)
    print("-" * len(header))
    for size in SIZES:
        for depth in DEPTHS:
            gpu = cuda_data.get((size, depth), {}).get('mean', 0)
            cpu1 = cpu_data.get((size, depth, 1), {}).get('mean', 0)
            best_cpu = min(
                (cpu_data.get((size, depth, t), {}).get('mean', float('inf'))
                 for t in THREADS),
                default=0
            )
            speedup = best_cpu / gpu if gpu > 0 else 0
            print(f"{size}x{size} x {depth:2d}p    {gpu:10.3f} {cpu1:12.3f} {best_cpu:14.3f} {speedup:11.1f}x")


if __name__ == '__main__':
    main()
