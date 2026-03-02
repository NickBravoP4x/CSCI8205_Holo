#!/usr/bin/env python3
"""
Extended Roofline Analysis with Bandwidth Ceilings, In-Core Ceilings,
Locality Walls, and Full Pipeline Stage Overlay.

Extends the basic roofline from plot_all_implementations.py with:
  1. Bandwidth ceilings — sub-peak BW diagonals (cache hierarchy)
  2. In-core ceilings — horizontal lines below peak compute
     (no FMA, no SIMD, single-core, low occupancy)
  3. Locality walls — vertical lines at cache boundary arithmetic intensities
  4. Pipeline stage points — each pipeline stage plotted on the GPU roofline

References:
  [3] Williams, Waterman, Patterson. "Roofline: An Insightful Visual Performance
      Model for Multicore Architectures." CACM 2009.
  [4] Ilic, Sousa, Roma. "Cache-aware Roofline model: Upgrading the loft."
      IEEE Computer Architecture Letters, 2014.

Usage:
    python Scripts/plot_extended_roofline.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np

# ── Display backend ──────────────────────────────────────────────────────────
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "Results"
OUTPUT_DIR = RESULTS_DIR / "Implementation_Comparison"

SIZES = [128, 256, 512, 1024]
DEPTHS = [1, 3, 5, 10, 20]
THREADS = [1, 2, 4, 8, 16, 32]

# ═══════════════════════════════════════════════════════════════════════════════
# Hardware Specifications
# ═══════════════════════════════════════════════════════════════════════════════

# Threadripper PRO 7975WX (Zen 4, 32 cores, 64 threads)
CPU_SPECS = {
    'name': 'Threadripper PRO 7975WX',
    'cores': 32,
    'boost_ghz': 5.3,
    'simd_width': 16,             # AVX-512 = 16 SP floats
    'fma_factor': 2,              # FMA = 2 FLOPs (mul + add)
    # Peak = 32 cores × 16 SP × 2 FMA × 5.3 GHz = 5427.2 GFLOPS
    # Conservative estimate accounting for not all cores boosting:
    'peak_compute': 2700e9,       # ~2.7 TFLOPS FP32 (realistic sustained)
    # Memory hierarchy bandwidth
    'l1_bw':    2000e9,           # ~2 TB/s (32 KB per core, ~64 B/cycle)
    'l2_bw':     800e9,           # ~800 GB/s (1 MB per core)
    'l3_bw':     400e9,           # ~400 GB/s (128 MB shared, 4 CCDs)
    'dram_bw':   150e9,           # 8-ch DDR5-5200 = ~150 GB/s
    # Cache sizes
    'l1_bytes':  32 * 1024,       # 32 KB per core
    'l2_bytes':  1 * 1024 * 1024, # 1 MB per core
    'l3_bytes': 128 * 1024 * 1024,# 128 MB shared
}

# RTX 5090 (Blackwell GB202, 170 SMs)
GPU_SPECS = {
    'name': 'RTX 5090',
    'sms': 170,
    'cuda_cores': 21760,
    'boost_ghz': 2.407,
    # Peak = 21760 × 2 FMA × 2.407 GHz = 104.8 TFLOPS
    'peak_compute': 105e12,       # ~105 TFLOPS FP32
    # Memory hierarchy bandwidth
    'l1_shared_bw': 12000e9,      # ~12 TB/s (128 KB per SM × 170 SMs)
    'l2_bw':         6000e9,      # ~6 TB/s (96 MB L2)
    'gddr7_bw':      1790e9,      # 1790 GB/s GDDR7
    # Cache sizes
    'l1_shared_bytes': 128 * 1024,    # 128 KB per SM
    'l2_bytes':         96 * 1024 * 1024,  # 96 MB total
}

# ═══════════════════════════════════════════════════════════════════════════════
# In-Core Computation Ceilings
# ═══════════════════════════════════════════════════════════════════════════════

CPU_INCORE_CEILINGS = {
    # Peak FP32 (AVX-512 + FMA, all cores)
    'Peak FP32 (AVX-512 + FMA)': CPU_SPECS['peak_compute'],
    # No FMA — only MUL or ADD, halves peak
    'No FMA (MUL or ADD only)': CPU_SPECS['peak_compute'] / 2,
    # No vectorization — scalar only, /16 for AVX-512
    'Scalar (no SIMD)': CPU_SPECS['peak_compute'] / CPU_SPECS['simd_width'],
    # Single-core scalar (Python/GIL limited)
    'Single Core (scalar)': CPU_SPECS['peak_compute'] / CPU_SPECS['simd_width'] / CPU_SPECS['cores'],
}

GPU_INCORE_CEILINGS = {
    # Peak FP32 (all SMs, FMA)
    'Peak FP32': GPU_SPECS['peak_compute'],
    # No FMA — halves peak
    'No FMA': GPU_SPECS['peak_compute'] / 2,
    # 50% occupancy (common for register-heavy kernels like FFT)
    '50% Occupancy': GPU_SPECS['peak_compute'] / 2,
    # Typical cuFFT efficiency (~30% of peak for large 2D FFTs)
    'Typical cuFFT (~30%)': GPU_SPECS['peak_compute'] * 0.3,
    # Single SM
    f'Single SM (1/{GPU_SPECS["sms"]})': GPU_SPECS['peak_compute'] / GPU_SPECS['sms'],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Locality Walls — vertical lines at cache boundary AIs
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fft_locality_wall(cache_bytes, padded_n):
    """
    For a 2D complex FFT of padded_n × padded_n:
    Working set = padded_n² × 8 bytes (complex float32).
    When WS > cache, data spills to next level.

    The AI at the transition = FLOPs_in_cache / cache_bytes
    For FFT: reuse factor ~ log2(N), so:
      AI_wall ≈ (5 × log2(padded_n) × elements_fitting) / cache_bytes

    Simplified: the wall is at the AI where the working set equals cache size.
    """
    ws = padded_n * padded_n * 8  # complex float32
    if ws <= cache_bytes:
        return None  # Working set fits, no spill
    # Effective AI when spilling: FLOPs / bytes_that_must_transfer
    # For FFT: ~5*N*log2(N) FLOPs per 1D pass, 2 passes for 2D
    flops_per_element = 5 * 2 * np.log2(padded_n)
    # Bytes per element transferred through this cache level
    bytes_per_element = 8  # complex float32
    return flops_per_element / bytes_per_element


def compute_locality_walls_cpu():
    """Compute AI walls for CPU cache hierarchy using FFT working sets."""
    walls = {}
    # Key: the AI at which the working set starts spilling to the next level
    # For 2D FFT, padded sizes: 192, 320, 512, 576, 1088

    # L1 (32 KB per core): any 2D FFT > ~64x64 spills
    # → most of our workloads are L1-limited
    # AI at L1 boundary: for smallest padded size (192x192 = 288 KB >> 32 KB)
    walls['L1 → L2 spill'] = 5 * 2 * np.log2(64) / 8  # ~7.5

    # L2 (1 MB per core): 192x192 (288 KB) fits; 320x320 (800 KB) fits; 512x512 (2 MB) doesn't
    # Transition AI: for padded_n where WS = 1 MB → N ~ 362
    walls['L2 → L3 spill'] = 5 * 2 * np.log2(362) / 8  # ~10.6

    # L3 (128 MB shared / 4 CCDs = 32 MB per CCD):
    # All our sizes fit in L3 easily (9.4 MB max for 1024x1024)
    # But effective L3 BW depends on CCD-to-CCD traffic
    # Skip L3 wall since all working sets fit

    return walls


def compute_locality_walls_gpu():
    """Compute AI walls for GPU cache hierarchy using FFT working sets."""
    walls = {}

    # L1/Shared per SM (128 KB): 2D FFT tile must fit
    # For cuFFT, each SM processes a tile. Tile size depends on FFT length.
    # Rough boundary: when per-SM allocation > 128 KB
    # For N=128 padded to 192: whole 2D FFT = 288 KB, but 1D slices fit (192*8=1.5 KB)
    # cuFFT does row-by-row → L1 is per-row, L2 is per-column pass
    walls['L1/Shared → L2'] = 5 * 2 * np.log2(128) / 8  # ~8.75

    # L2 (96 MB total): all working sets fit easily
    # But effective L2 BW depends on access pattern (strided column access in 2D FFT)
    # Column pass stride = padded_N * 8 bytes → causes L2 cache line waste
    # Effective AI drops when stride > L2 line size
    walls['L2 → GDDR7 (stride)'] = 5 * 2 * np.log2(512) / 8  # ~11.25

    return walls


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Stage FLOP/Byte Estimates (for GPU roofline overlay)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_pipeline_stages():
    """
    Estimate FLOPs, bytes, and measured latency for each C++ pipeline stage.
    Returns list of dicts with keys: name, flops, bytes, latency_ms, color, marker.
    """
    S = 448
    HW = S * S  # 200704

    stages = []

    # Stage 1: Image Load (disk → CPU → GPU)
    # FLOPs: uint8→float32 conversion = HW muls
    # Bytes: read from disk (~200 KB compressed) + H2D (HW * 4)
    stages.append({
        'name': 'Load & H2D',
        'flops': HW * 1,           # trivial conversion
        'bytes': HW * 4 + HW * 1,  # H2D float32 + disk read uint8
        'latency_ms': 4.217,
        'color': '#e74c3c',
        'marker': 'v',
    })

    # Stage 2: EMA Background Subtraction
    # Per pixel: 2 mul (α*f, (1-α)*bg) + 1 add (new bg) + 1 sub (f-bg) +
    #            1 abs + 1 mul (*scale) + 1 add (+mean) + 1 clamp + 1 div (/255) = ~9 FLOPs
    # Bytes: read frame (4B) + read bg (4B) + write bg (4B) + write output (4B) = 16 B/pixel
    stages.append({
        'name': 'EMA',
        'flops': HW * 9,
        'bytes': HW * 16,
        'latency_ms': 0.007,
        'color': '#f39c12',
        'marker': '^',
    })

    # Stage 3: Holographic Reconstruction (3 planes, 448x448)
    # 2D FFT forward + 3× (element-wise Hz multiply + IFFT) + crop + normalize
    padded = S + 64  # 512
    fft_2d_ops = 5 * (padded * np.log2(padded) * padded +
                       padded * np.log2(padded) * padded)
    per_plane = 2 * fft_2d_ops + padded * padded * 20  # fwd/inv FFT + element-wise
    total_flops = per_plane * 3
    # Bytes: hologram (HW*4) + output (3*HW*4) + intermediate complex (2*padded²*8*3)
    total_bytes = (HW * 4 + 3 * HW * 4 + 2 * 3 * padded * padded * 8)
    stages.append({
        'name': 'Recon (3-plane)',
        'flops': total_flops,
        'bytes': total_bytes,
        'latency_ms': 0.145,
        'color': '#2ecc71',
        'marker': 'o',
    })

    # Stage 4: Heatmap Detection (TRT inference)
    # MobileNetV3-Small + FPN at 448x448
    # MobileNetV3-Small: ~57M MACs at 224x224 → ~228M MACs at 448x448 = 456M FLOPs
    # FPN overhead ~30%: ~593M FLOPs
    # ImageNet normalize: 3*HW*4 = ~2.4M FLOPs (negligible)
    # NMS: 112*112*121 = ~1.5M FLOPs (negligible)
    hm_flops = 600e6  # ~600M FLOPs total
    # Input: 3*448*448*4 = 2.4 MB, Output: 112*112*4 = 50 KB
    # Weights (read from global mem each inference): ~3.3 MB
    hm_bytes = 3 * S * S * 4 + 112 * 112 * 4 + 3.3e6
    stages.append({
        'name': 'Heatmap (TRT)',
        'flops': hm_flops,
        'bytes': hm_bytes,
        'latency_ms': 0.457,
        'color': '#3498db',
        'marker': 'D',
    })

    # Stage 5: Classification (TRT inference, batched crops)
    # YOLO11n-cls: ~1.5M params, ~60M FLOPs per 128x128 crop
    # Average ~45 detections per frame (456052 / 10000)
    avg_crops = 45
    crop_flops = 60e6 * avg_crops  # ~2.7G FLOPs
    # Bytes: crops H2D (avg_crops*3*128*128*4) + weights (~4.5 MB) + output D2H (tiny)
    crop_bytes = avg_crops * 3 * 128 * 128 * 4 + 4.5e6 + avg_crops * 2 * 4
    stages.append({
        'name': 'Classify (TRT)',
        'flops': crop_flops,
        'bytes': crop_bytes,
        'latency_ms': 1.447,
        'color': '#9b59b6',
        'marker': 's',
    })

    # Stage 6: Post-processing (CPU)
    # Negligible FLOPs — pairwise distance for ~45 detections × 50 history
    # ~45*50*3 = 6750 FLOPs, ~45*50*4 = 9000 bytes
    stages.append({
        'name': 'Post-proc',
        'flops': 45 * 50 * 3,
        'bytes': 45 * 50 * 4 * 4,  # float coords
        'latency_ms': 0.016,
        'color': '#95a5a6',
        'marker': 'P',
    })

    return stages


# ═══════════════════════════════════════════════════════════════════════════════
# Reconstruction benchmark parsers (reused from plot_all_implementations.py)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_python_results(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for r in data['results']:
        if r['batch_size'] != 1:
            continue
        key = (r['image_size'][0], r['depth_count'])
        if key not in out:
            out[key] = {}
        if r['mode'] == 'Single CPU':
            out[key]['cpu'] = r['mean_latency']
        elif r['mode'] == 'Single GPU':
            out[key]['gpu'] = r['mean_latency']
    return out


def parse_cuda_results(path):
    data = {}
    text = path.read_text()
    blocks = re.split(r'=== CUDA (\d+)x\d+ x (\d+) planes ===', text)
    for i in range(1, len(blocks) - 2, 3):
        size, depth = int(blocks[i]), int(blocks[i + 1])
        body = blocks[i + 2]
        for line in body.splitlines():
            if line.strip().startswith('Mean:'):
                data[(size, depth)] = float(re.search(r'([\d.]+)', line).group(1))
    return data


def parse_cpu_results(path):
    data = {}
    text = path.read_text()
    blocks = re.split(r'=== CPU (\d+)x\d+ x (\d+) planes \(thread sweep\) ===', text)
    for i in range(1, len(blocks) - 2, 3):
        size, depth = int(blocks[i]), int(blocks[i + 1])
        body = blocks[i + 2]
        best = float('inf')
        for line in body.splitlines():
            m = re.match(r'Threads:\s*\d+\s*\|\s*Mean:\s*([\d.]+)', line.strip())
            if m:
                val = float(m.group(1))
                if val < best:
                    best = val
        if best < float('inf'):
            data[(size, depth)] = best
    return data


def estimate_flops(h, w, depth):
    fft_2d_ops = 5 * (h * np.log2(h) * w + w * np.log2(w) * h)
    per_plane = 2 * fft_2d_ops + h * w * 20
    return per_plane * depth


def calc_ai(h, w, depth):
    flops = estimate_flops(h, w, depth)
    mem = h * w * 4 + h * w * depth * 4 + 2 * h * w * depth * 8
    return flops / mem


# ═══════════════════════════════════════════════════════════════════════════════
# Plot: Extended CPU Roofline
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cpu_roofline(ax, py_data, cpp_cpu_data):
    oi_range = np.logspace(-2, 3, 1000)

    # ── Bandwidth ceilings (diagonals) ──
    bw_levels = [
        ('L1 (~2 TB/s)',       CPU_SPECS['l1_bw'],   '#ff7f0e', 0.3, 0.8),
        ('L2 (~800 GB/s)',     CPU_SPECS['l2_bw'],   '#2ca02c', 0.3, 0.8),
        ('L3 (~400 GB/s)',     CPU_SPECS['l3_bw'],   '#9467bd', 0.4, 1.0),
        ('DRAM (150 GB/s)',    CPU_SPECS['dram_bw'],  '#1f77b4', 0.6, 1.5),
    ]
    for label, bw, color, alpha, lw in bw_levels:
        ax.loglog(oi_range, bw * oi_range, '--', color=color, alpha=alpha,
                  linewidth=lw, label=label)

    # ── In-core ceilings (horizontal lines) ──
    ceiling_styles = [
        ('Peak FP32 (AVX-512 + FMA)', CPU_INCORE_CEILINGS['Peak FP32 (AVX-512 + FMA)'],
         'red', '-', 2.0, 0.8),
        ('No FMA', CPU_INCORE_CEILINGS['No FMA (MUL or ADD only)'],
         '#e74c3c', '-.', 1.2, 0.5),
        ('Scalar (no SIMD)', CPU_INCORE_CEILINGS['Scalar (no SIMD)'],
         '#d35400', ':', 1.2, 0.5),
        ('Single Core', CPU_INCORE_CEILINGS['Single Core (scalar)'],
         '#c0392b', ':', 1.0, 0.4),
    ]
    for label, val, color, ls, lw, alpha in ceiling_styles:
        ax.axhline(y=val, color=color, linestyle=ls, linewidth=lw, alpha=alpha)
        # Label on right edge
        ax.text(200, val * 1.15, label, fontsize=6, color=color, alpha=0.8,
                ha='right', va='bottom')

    # ── DRAM roofline (thick black line) ──
    roofline = np.minimum(CPU_SPECS['dram_bw'] * oi_range, CPU_SPECS['peak_compute'])
    ax.loglog(oi_range, roofline, 'k-', linewidth=2.5, label='DRAM Roofline')

    # ── Locality walls (vertical lines) ──
    cpu_walls = compute_locality_walls_cpu()
    wall_colors = {'L1 → L2 spill': '#ff7f0e', 'L2 → L3 spill': '#2ca02c'}
    for label, ai_val in cpu_walls.items():
        color = wall_colors.get(label, 'gray')
        ax.axvline(x=ai_val, color=color, linestyle=':', linewidth=1.5, alpha=0.5)
        ax.text(ai_val * 0.85, 3e7, label, fontsize=6, color=color, alpha=0.7,
                rotation=90, ha='right', va='bottom')

    # ── Data points: Python CPU ──
    ois, perfs = [], []
    for size in SIZES:
        for depth in DEPTHS:
            lat = py_data.get((size, depth), {}).get('cpu', 0)
            if lat <= 0:
                continue
            oi = calc_ai(size, size, depth)
            flops = estimate_flops(size, size, depth)
            ois.append(oi)
            perfs.append(flops / (lat / 1000))
    ax.scatter(ois, perfs, color='#4c72b0', marker='s', s=50, alpha=0.7,
               label='Python CPU', edgecolors='black', linewidths=0.5, zorder=5)

    # ── Data points: C++ OpenMP ──
    ois, perfs = [], []
    for size in SIZES:
        for depth in DEPTHS:
            lat = cpp_cpu_data.get((size, depth), 0)
            if lat <= 0:
                continue
            oi = calc_ai(size, size, depth)
            flops = estimate_flops(size, size, depth)
            ois.append(oi)
            perfs.append(flops / (lat / 1000))
    ax.scatter(ois, perfs, color='#55a868', marker='^', s=50, alpha=0.7,
               label='C++ OpenMP (best threads)', edgecolors='black', linewidths=0.5, zorder=5)

    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=11)
    ax.set_ylabel('Performance (FLOP/s)', fontsize=11)
    ax.set_title(f'Extended CPU Roofline — {CPU_SPECS["name"]}', fontsize=12, fontweight='bold')
    ax.set_xlim(0.05, 300)
    ax.set_ylim(1e7, 1e13)
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot: Extended GPU Roofline
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gpu_roofline(ax, py_data, cuda_data, pipeline_stages):
    oi_range = np.logspace(-2, 4, 1000)

    # ── Bandwidth ceilings (diagonals) ──
    bw_levels = [
        ('L1/Shared (~12 TB/s)', GPU_SPECS['l1_shared_bw'], '#ff7f0e', 0.3, 0.8),
        ('L2 (~6 TB/s)',          GPU_SPECS['l2_bw'],        '#2ca02c', 0.4, 1.0),
        ('GDDR7 (1790 GB/s)',     GPU_SPECS['gddr7_bw'],     '#1f77b4', 0.6, 1.5),
    ]
    for label, bw, color, alpha, lw in bw_levels:
        ax.loglog(oi_range, bw * oi_range, '--', color=color, alpha=alpha,
                  linewidth=lw, label=label)

    # ── In-core ceilings (horizontal lines) ──
    ceiling_items = [
        ('Peak FP32 (105 TF)', GPU_INCORE_CEILINGS['Peak FP32'],
         'red', '-', 2.0, 0.8),
        ('No FMA / 50% Occup.', GPU_INCORE_CEILINGS['No FMA'],
         '#e74c3c', '-.', 1.2, 0.5),
        ('Typical cuFFT (~30%)', GPU_INCORE_CEILINGS['Typical cuFFT (~30%)'],
         '#d35400', ':', 1.2, 0.5),
        (f'Single SM (1/{GPU_SPECS["sms"]})', GPU_INCORE_CEILINGS[f'Single SM (1/{GPU_SPECS["sms"]})'],
         '#c0392b', ':', 1.0, 0.4),
    ]
    for label, val, color, ls, lw, alpha in ceiling_items:
        ax.axhline(y=val, color=color, linestyle=ls, linewidth=lw, alpha=alpha)
        ax.text(3000, val * 1.15, label, fontsize=6, color=color, alpha=0.8,
                ha='right', va='bottom')

    # ── GDDR7 roofline (thick black) ──
    roofline = np.minimum(GPU_SPECS['gddr7_bw'] * oi_range, GPU_SPECS['peak_compute'])
    ax.loglog(oi_range, roofline, 'k-', linewidth=2.5, label='GDDR7 Roofline')

    # ── Locality walls (vertical lines) ──
    gpu_walls = compute_locality_walls_gpu()
    wall_colors = {'L1/Shared → L2': '#ff7f0e', 'L2 → GDDR7 (stride)': '#2ca02c'}
    for label, ai_val in gpu_walls.items():
        color = wall_colors.get(label, 'gray')
        ax.axvline(x=ai_val, color=color, linestyle=':', linewidth=1.5, alpha=0.5)
        ax.text(ai_val * 0.85, 5e8, label, fontsize=6, color=color, alpha=0.7,
                rotation=90, ha='right', va='bottom')

    # ── Data points: Python GPU (reconstruction) ──
    ois, perfs = [], []
    for size in SIZES:
        for depth in DEPTHS:
            lat = py_data.get((size, depth), {}).get('gpu', 0)
            if lat <= 0:
                continue
            oi = calc_ai(size, size, depth)
            flops = estimate_flops(size, size, depth)
            ois.append(oi)
            perfs.append(flops / (lat / 1000))
    ax.scatter(ois, perfs, color='#dd8452', marker='D', s=50, alpha=0.7,
               label='Python GPU (recon)', edgecolors='black', linewidths=0.5, zorder=5)

    # ── Data points: C++ CUDA (reconstruction) ──
    ois, perfs = [], []
    for size in SIZES:
        for depth in DEPTHS:
            lat = cuda_data.get((size, depth), 0)
            if lat <= 0:
                continue
            oi = calc_ai(size, size, depth)
            flops = estimate_flops(size, size, depth)
            ois.append(oi)
            perfs.append(flops / (lat / 1000))
    ax.scatter(ois, perfs, color='#c44e52', marker='o', s=50, alpha=0.7,
               label='C++ CUDA (recon)', edgecolors='black', linewidths=0.5, zorder=5)

    # ── Pipeline stage overlay points ──
    for stage in pipeline_stages:
        ai = stage['flops'] / stage['bytes'] if stage['bytes'] > 0 else 0.01
        perf = stage['flops'] / (stage['latency_ms'] / 1000) if stage['latency_ms'] > 0 else 0
        if perf > 0:
            ax.scatter([ai], [perf], color=stage['color'], marker=stage['marker'],
                       s=150, alpha=0.9, edgecolors='black', linewidths=1.0, zorder=10)
            # Label with offset to avoid overlap
            offset_y = 1.5 if stage['name'] not in ('Post-proc', 'Load & H2D') else 0.5
            ax.annotate(stage['name'],
                        xy=(ai, perf), xytext=(5, 8),
                        textcoords='offset points',
                        fontsize=7, fontweight='bold', color=stage['color'],
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor=stage['color'], alpha=0.8))

    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=11)
    ax.set_ylabel('Performance (FLOP/s)', fontsize=11)
    ax.set_title(f'Extended GPU Roofline — {GPU_SPECS["name"]}', fontsize=12, fontweight='bold')
    ax.set_xlim(0.01, 5000)
    ax.set_ylim(1e6, 5e14)
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot: Pipeline-Only Roofline (focused view)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pipeline_roofline(pipeline_stages):
    """Focused GPU roofline showing only pipeline stages with annotations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    oi_range = np.logspace(-2, 4, 1000)

    # Bandwidth ceilings
    for label, bw, color in [
        ('L1/Shared (~12 TB/s)', GPU_SPECS['l1_shared_bw'], '#ff7f0e'),
        ('L2 (~6 TB/s)', GPU_SPECS['l2_bw'], '#2ca02c'),
        ('GDDR7 (1790 GB/s)', GPU_SPECS['gddr7_bw'], '#1f77b4'),
    ]:
        ax.loglog(oi_range, bw * oi_range, '--', color=color, alpha=0.4,
                  linewidth=1.0, label=label)

    # In-core ceilings
    for label, val, ls in [
        ('Peak FP32 (105 TF)', GPU_SPECS['peak_compute'], '-'),
        ('No FMA / 50% Occup.', GPU_SPECS['peak_compute'] / 2, '-.'),
        ('Typical cuFFT (~30%)', GPU_SPECS['peak_compute'] * 0.3, ':'),
    ]:
        ax.axhline(y=val, color='red', linestyle=ls, linewidth=1.0, alpha=0.4)
        ax.text(2000, val * 1.1, label, fontsize=7, color='red', alpha=0.6, ha='right')

    # GDDR7 roofline
    roofline = np.minimum(GPU_SPECS['gddr7_bw'] * oi_range, GPU_SPECS['peak_compute'])
    ax.loglog(oi_range, roofline, 'k-', linewidth=2.5, alpha=0.8, label='GDDR7 Roofline')

    # Locality walls
    for label, ai_val, color in [
        ('L1/Shared → L2', 5 * 2 * np.log2(128) / 8, '#ff7f0e'),
        ('L2 → GDDR7', 5 * 2 * np.log2(512) / 8, '#2ca02c'),
    ]:
        ax.axvline(x=ai_val, color=color, linestyle=':', linewidth=1.5, alpha=0.4)
        ax.text(ai_val * 0.8, 2e6, label, fontsize=7, color=color, alpha=0.6,
                rotation=90, ha='right')

    # Pipeline stages (large markers with detailed annotations)
    for stage in pipeline_stages:
        ai = stage['flops'] / stage['bytes'] if stage['bytes'] > 0 else 0.01
        perf = stage['flops'] / (stage['latency_ms'] / 1000) if stage['latency_ms'] > 0 else 1

        ax.scatter([ai], [perf], color=stage['color'], marker=stage['marker'],
                   s=200, alpha=0.9, edgecolors='black', linewidths=1.5, zorder=10)

        # Determine what limits this stage
        roofline_at_ai = min(GPU_SPECS['gddr7_bw'] * ai, GPU_SPECS['peak_compute'])
        efficiency = perf / roofline_at_ai * 100

        # Detailed annotation
        label_text = (f"{stage['name']}\n"
                      f"AI={ai:.1f}, {stage['latency_ms']:.3f} ms\n"
                      f"{efficiency:.0f}% of roofline")
        ax.annotate(label_text,
                    xy=(ai, perf), xytext=(15, -10),
                    textcoords='offset points',
                    fontsize=8, color=stage['color'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=stage['color'], alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color=stage['color'], lw=1.5))

    # Ridge point annotation
    ridge = GPU_SPECS['peak_compute'] / GPU_SPECS['gddr7_bw']
    ax.axvline(x=ridge, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.text(ridge, 1e7, f'Ridge Point\nAI={ridge:.1f}', fontsize=7,
            ha='center', color='gray', alpha=0.6)

    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    ax.set_ylabel('Performance (FLOP/s)', fontsize=12)
    ax.set_title('C++ Pipeline Stages on GPU Roofline — RTX 5090',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0.01, 5000)
    ax.set_ylim(1e6, 5e14)
    ax.grid(True, alpha=0.2)

    # Custom legend
    handles = [Line2D([0], [0], color='k', linewidth=2.5, label='GDDR7 Roofline')]
    handles += [Line2D([0], [0], marker=s['marker'], color=s['color'], markersize=10,
                        linestyle='None', label=s['name']) for s in pipeline_stages]
    ax.legend(handles=handles, loc='lower right', fontsize=8)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Locate data files
    py_json = RESULTS_DIR / 'Week1-2_Python_Baseline_5090' / 'benchmark_results' / 'benchmark_results.json'
    cuda_txt = RESULTS_DIR / 'Reconstruction_Benchmark' / 'cuda_benchmark_full.txt'
    cpp_cpu_txt = RESULTS_DIR / 'Reconstruction_Benchmark' / 'cpu_benchmark_full.txt'

    for p in [py_json, cuda_txt, cpp_cpu_txt]:
        if not p.exists():
            print(f"ERROR: Missing {p}")
            return

    print("Parsing reconstruction benchmarks...")
    py_data = parse_python_results(py_json)
    cuda_data = parse_cuda_results(cuda_txt)
    cpp_cpu_data = parse_cpu_results(cpp_cpu_txt)
    print(f"  Python: {len(py_data)} configs, C++ CUDA: {len(cuda_data)}, C++ CPU: {len(cpp_cpu_data)}")

    # Pipeline stage estimates
    pipeline_stages = estimate_pipeline_stages()
    print(f"\nPipeline stage estimates (C++ TRT, 448x448, 3 planes):")
    print(f"{'Stage':<20s} {'AI':>8s} {'GFLOP/s':>10s} {'Latency':>10s} {'Roofline %':>10s}")
    print("-" * 60)
    for s in pipeline_stages:
        ai = s['flops'] / s['bytes'] if s['bytes'] > 0 else 0
        perf = s['flops'] / (s['latency_ms'] / 1000) if s['latency_ms'] > 0 else 0
        roof = min(GPU_SPECS['gddr7_bw'] * ai, GPU_SPECS['peak_compute'])
        eff = perf / roof * 100 if roof > 0 else 0
        print(f"{s['name']:<20s} {ai:>8.1f} {perf/1e9:>10.1f} {s['latency_ms']:>8.3f} ms {eff:>9.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    # ── Figure 1: Extended two-panel roofline ──
    fig, (cpu_ax, gpu_ax) = plt.subplots(1, 2, figsize=(18, 8))
    plot_cpu_roofline(cpu_ax, py_data, cpp_cpu_data)
    plot_gpu_roofline(gpu_ax, py_data, cuda_data, pipeline_stages)

    # Shared legend for bandwidth ceilings
    fig.suptitle('Extended Roofline Analysis — Ceilings, Walls, and Pipeline Stages',
                 fontsize=14, fontweight='bold')

    # Build compact legends
    for ax_item in [cpu_ax, gpu_ax]:
        ax_item.legend(fontsize=6, loc='lower right', ncol=1)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUTPUT_DIR / "extended_roofline.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)
    print(f"\nSaved: {path}")

    # ── Figure 2: Pipeline-focused roofline ──
    fig = plot_pipeline_roofline(pipeline_stages)
    path = OUTPUT_DIR / "pipeline_roofline.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)
    print(f"Saved: {path}")

    # ── Figure 3: CPU-only extended roofline (larger) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_cpu_roofline(ax, py_data, cpp_cpu_data)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    path = OUTPUT_DIR / "extended_roofline_cpu.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)
    print(f"Saved: {path}")

    # ── Figure 4: GPU-only extended roofline (larger) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_gpu_roofline(ax, py_data, cuda_data, pipeline_stages)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    path = OUTPUT_DIR / "extended_roofline_gpu.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)
    print(f"Saved: {path}")

    print(f"\nTotal: {len(saved)} figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
