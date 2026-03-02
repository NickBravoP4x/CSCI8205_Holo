#!/usr/bin/env python3
"""
Comprehensive C++ vs Python pipeline comparison.

Compares per-stage latency, end-to-end throughput, and speedup across
all available pipeline implementations: Python CPU, Python PyTorch GPU,
Python TensorRT, and C++ TensorRT.

Outputs to Results/Pipeline_Benchmark/figures/

Usage:
    python Scripts/plot_pipeline_comparison.py
"""

import json
import os
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
import matplotlib.ticker as ticker
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results" / "Pipeline_Benchmark"
OUTPUT_DIR = RESULTS_DIR / "figures"

# ── Canonical stage ordering ─────────────────────────────────────────────────
# Map from various JSON key names to canonical stage names
STAGE_KEY_MAP = {
    # Python keys
    'stage1_load': 'load',
    'stage2_ema': 'ema',
    'stage3_holographic': 'recon',
    'stage4_detect': 'detect',
    'stage5_classify': 'classify',
    'stage6_postprocess': 'postproc',
    # C++ keys (slightly different names)
    'stage4_heatmap': 'detect',
    'stage5_classification': 'classify',
    'stage6_postprocessing': 'postproc',
}

CANONICAL_STAGES = ['load', 'ema', 'recon', 'detect', 'classify', 'postproc']

STAGE_DISPLAY = {
    'load': 'Image Load & Transfer',
    'ema': 'EMA Background Sub.',
    'recon': 'Holographic Recon (3-plane)',
    'detect': 'Heatmap Detection',
    'classify': 'Classification',
    'postproc': 'Post-processing',
}

# ── Configuration definitions ────────────────────────────────────────────────
# (subdir, json_filename, display_label, color)
CONFIGS = [
    ('cpu_quick',   'pipeline_results.json',     'Python CPU',         '#e74c3c'),
    ('pytorch',     'pipeline_results.json',     'Python PyTorch GPU', '#3498db'),
    ('trt',         'pipeline_results.json',     'Python TensorRT',    '#2ecc71'),
    ('cpp_trt',     'pipeline_results_10k.json', 'C++ TensorRT',       '#9b59b6'),
]


def load_config(subdir, filename):
    """Load a benchmark JSON, return None if not found."""
    path = RESULTS_DIR / subdir / filename
    if not path.exists():
        # Fallback: try default name
        path = RESULTS_DIR / subdir / 'pipeline_results.json'
        if not path.exists():
            return None
    with open(path) as f:
        return json.load(f)


def normalize_stages(raw_stats):
    """Map raw stage keys to canonical names, return dict of canonical -> stats."""
    out = {}
    for raw_key, stats in raw_stats.items():
        canon = STAGE_KEY_MAP.get(raw_key)
        if canon:
            out[canon] = stats
    return out


def get_num_images(data):
    """Get number of timed images from either Python or C++ format."""
    cfg = data.get('config', {})
    return cfg.get('num_timed', cfg.get('num_images', '?'))


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Per-stage grouped horizontal bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_stage(configs, labels, colors, stages_data):
    n_configs = len(configs)
    n_stages = len(CANONICAL_STAGES)

    fig, ax = plt.subplots(figsize=(13, 6))
    y = np.arange(n_stages)
    bar_h = 0.8 / n_configs

    for ci, (cfg, label, color) in enumerate(zip(configs, labels, colors)):
        means = []
        errs_lo = []
        errs_hi = []
        for stage in CANONICAL_STAGES:
            s = stages_data[cfg].get(stage)
            if s:
                means.append(s['mean_ms'])
                errs_lo.append(max(0, s['mean_ms'] - s['min_ms']))
                errs_hi.append(max(0, s['p95_ms'] - s['mean_ms']))
            else:
                means.append(0)
                errs_lo.append(0)
                errs_hi.append(0)

        offset = (ci - n_configs / 2 + 0.5) * bar_h
        ax.barh(y + offset, means, bar_h,
                xerr=[errs_lo, errs_hi],
                label=label, color=color, capsize=2, alpha=0.85)

        for i, m in enumerate(means):
            if m > 0.001:
                ax.text(m * 1.08, y[i] + offset, f"{m:.3f}" if m < 1 else f"{m:.2f}",
                        va='center', ha='left', fontsize=7, color=color, fontweight='bold')

    ax.axvline(x=2.5, color='red', linestyle='--', lw=1.5, alpha=0.6,
               label='2.5 ms target (400 Hz)')
    ax.set_yticks(y)
    ax.set_yticklabels([STAGE_DISPLAY[s] for s in CANONICAL_STAGES], fontsize=10)
    ax.set_xlabel("Mean Latency (ms)", fontsize=11)
    ax.set_xscale("log")
    ax.set_title("Per-Stage Latency: Python vs C++ Pipeline", fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: End-to-end bar chart with FPS annotations
# ═══════════════════════════════════════════════════════════════════════════════
def plot_e2e(configs, labels, colors, raw_data):
    fig, ax = plt.subplots(figsize=(10, 5))

    means = [raw_data[c]['end_to_end']['mean_ms'] for c in configs]
    medians = [raw_data[c]['end_to_end']['median_ms'] for c in configs]
    p95s = [raw_data[c]['end_to_end']['p95_ms'] for c in configs]

    x = np.arange(len(configs))
    w = 0.25

    bars_mean = ax.bar(x - w, means, w, color=colors, alpha=0.9, label='Mean')
    bars_med = ax.bar(x, medians, w, color=colors, alpha=0.6, label='Median')
    bars_p95 = ax.bar(x + w, p95s, w, color=colors, alpha=0.35, label='P95')

    for bar, m in zip(bars_mean, means):
        fps = 1000.0 / m
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{m:.1f} ms\n{fps:.0f} FPS",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=2.5, color='red', linestyle='--', lw=1.5, alpha=0.6,
               label='2.5 ms target (400 Hz)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("End-to-End Pipeline Latency: Mean / Median / P95", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Speedup over Python CPU baseline (per-stage + E2E)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_speedup(configs, labels, colors, stages_data, raw_data, baseline_cfg):
    baseline_stages = stages_data[baseline_cfg]
    baseline_e2e = raw_data[baseline_cfg]['end_to_end']['mean_ms']

    # Skip baseline itself
    compare_cfgs = [(c, l, col) for c, l, col in zip(configs, labels, colors) if c != baseline_cfg]
    if not compare_cfgs:
        return None

    stages_plus_e2e = CANONICAL_STAGES + ['e2e']
    display_plus_e2e = [STAGE_DISPLAY[s] for s in CANONICAL_STAGES] + ['End-to-End']

    n_compare = len(compare_cfgs)
    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(stages_plus_e2e))
    bar_h = 0.8 / n_compare

    for ci, (cfg, label, color) in enumerate(compare_cfgs):
        speedups = []
        for stage in CANONICAL_STAGES:
            base_s = baseline_stages.get(stage)
            comp_s = stages_data[cfg].get(stage)
            if base_s and comp_s and comp_s['mean_ms'] > 0:
                speedups.append(base_s['mean_ms'] / comp_s['mean_ms'])
            else:
                speedups.append(0)
        # E2E speedup
        comp_e2e = raw_data[cfg]['end_to_end']['mean_ms']
        speedups.append(baseline_e2e / comp_e2e)

        offset = (ci - n_compare / 2 + 0.5) * bar_h
        ax.barh(y + offset, speedups, bar_h,
                label=label, color=color, alpha=0.85)

        for i, sp in enumerate(speedups):
            if sp > 0:
                ax.text(sp + 0.1, y[i] + offset, f"{sp:.1f}x",
                        va='center', ha='left', fontsize=8, color=color, fontweight='bold')

    ax.axvline(x=1.0, color='gray', linestyle='-', lw=1, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(display_plus_e2e, fontsize=10)
    ax.set_xlabel("Speedup vs Python CPU", fontsize=11)
    ax.set_title("Per-Stage Speedup Over Python CPU Baseline", fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Stacked bar — E2E breakdown by stage
# ═══════════════════════════════════════════════════════════════════════════════
def plot_stacked_breakdown(configs, labels, colors, stages_data):
    stage_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#95a5a6']

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(configs))

    bottoms = np.zeros(len(configs))
    for si, stage in enumerate(CANONICAL_STAGES):
        vals = []
        for cfg in configs:
            s = stages_data[cfg].get(stage)
            vals.append(s['mean_ms'] if s else 0)
        ax.bar(x, vals, bottom=bottoms, color=stage_colors[si],
               label=STAGE_DISPLAY[stage], alpha=0.85, width=0.5)
        # Label significant stages
        for i, v in enumerate(vals):
            if v > 0.5:  # only label stages > 0.5 ms
                ax.text(x[i], bottoms[i] + v / 2, f"{v:.1f}",
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        bottoms += vals

    # Total label on top
    for i, total in enumerate(bottoms):
        ax.text(x[i], total + 0.3, f"{total:.1f} ms",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("End-to-End Latency Breakdown by Stage", fontsize=13, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5: C++ vs Python TRT head-to-head per-stage speedup
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cpp_vs_python_trt(stages_data, raw_data):
    if 'trt' not in stages_data or 'cpp_trt' not in stages_data:
        return None

    py_stages = stages_data['trt']
    cpp_stages = stages_data['cpp_trt']

    stages = CANONICAL_STAGES
    display = [STAGE_DISPLAY[s] for s in stages]

    py_means = [py_stages.get(s, {}).get('mean_ms', 0) for s in stages]
    cpp_means = [cpp_stages.get(s, {}).get('mean_ms', 0) for s in stages]
    speedups = []
    for p, c in zip(py_means, cpp_means):
        speedups.append(p / c if c > 0 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

    # Left: side-by-side bars
    y = np.arange(len(stages))
    bar_h = 0.35
    ax1.barh(y - bar_h/2, py_means, bar_h, label='Python TensorRT', color='#2ecc71', alpha=0.85)
    ax1.barh(y + bar_h/2, cpp_means, bar_h, label='C++ TensorRT', color='#9b59b6', alpha=0.85)

    for i in range(len(stages)):
        if py_means[i] > 0.001:
            ax1.text(py_means[i] * 1.05, y[i] - bar_h/2,
                     f"{py_means[i]:.3f}" if py_means[i] < 1 else f"{py_means[i]:.2f}",
                     va='center', fontsize=7, color='#2ecc71')
        if cpp_means[i] > 0.001:
            ax1.text(cpp_means[i] * 1.05, y[i] + bar_h/2,
                     f"{cpp_means[i]:.3f}" if cpp_means[i] < 1 else f"{cpp_means[i]:.2f}",
                     va='center', fontsize=7, color='#9b59b6')

    ax1.set_yticks(y)
    ax1.set_yticklabels(display, fontsize=10)
    ax1.set_xlabel("Latency (ms)", fontsize=11)
    ax1.set_xscale("log")
    ax1.set_title("Per-Stage: Python TRT vs C++ TRT", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Right: speedup bars
    colors = ['#27ae60' if s >= 1 else '#e74c3c' for s in speedups]
    ax2.barh(y, speedups, 0.5, color=colors, alpha=0.85)
    ax2.axvline(x=1.0, color='gray', linestyle='-', lw=1, alpha=0.5)
    for i, sp in enumerate(speedups):
        ax2.text(sp + 0.1, y[i], f"{sp:.1f}x", va='center', fontsize=9, fontweight='bold')
    ax2.set_yticks(y)
    ax2.set_yticklabels(display, fontsize=10)
    ax2.set_xlabel("Speedup (C++ vs Python)", fontsize=11)
    ax2.set_title("C++ Speedup per Stage", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # Add E2E speedup annotation
    py_e2e = raw_data['trt']['end_to_end']['mean_ms']
    cpp_e2e = raw_data['cpp_trt']['end_to_end']['mean_ms']
    e2e_speedup = py_e2e / cpp_e2e
    fig.text(0.5, 0.01,
             f"End-to-End: Python TRT {py_e2e:.1f} ms ({1000/py_e2e:.0f} FPS) "
             f"  vs  C++ TRT {cpp_e2e:.1f} ms ({1000/cpp_e2e:.0f} FPS) "
             f"  =  {e2e_speedup:.1f}x speedup",
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary(configs, labels, stages_data, raw_data):
    col_w = 18
    header = f"{'Stage':<28s}"
    for label in labels:
        header += f" {label:>{col_w}s}"
    sep = "=" * len(header)

    print()
    print("Pipeline Implementation Comparison")
    print(sep)
    print(header)
    print(sep)

    for stage in CANONICAL_STAGES:
        display = STAGE_DISPLAY[stage]
        row = f"{display:<28s}"
        for cfg in configs:
            s = stages_data[cfg].get(stage)
            if s:
                row += f" {s['mean_ms']:>{col_w}.3f}"
            else:
                row += f" {'N/A':>{col_w}s}"
        print(row)

    print(sep)
    row = f"{'End-to-End (mean)':.<28s}"
    for cfg in configs:
        row += f" {raw_data[cfg]['end_to_end']['mean_ms']:>{col_w}.3f}"
    print(row)

    row = f"{'Throughput (FPS)':.<28s}"
    for cfg in configs:
        row += f" {raw_data[cfg]['end_to_end']['throughput_fps']:>{col_w}.1f}"
    print(row)

    row = f"{'N (images)':.<28s}"
    for cfg in configs:
        row += f" {get_num_images(raw_data[cfg]):>{col_w}}"
    print(row)

    row = f"{'Total Detections':.<28s}"
    for cfg in configs:
        det = raw_data[cfg].get('detection_summary', {})
        total = det.get('total_filtered', det.get('total_detections', 'N/A'))
        row += f" {total:>{col_w}}"
    print(row)
    print(sep)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all available configs
    raw_data = {}
    available = []
    available_labels = []
    available_colors = []

    for subdir, filename, label, color in CONFIGS:
        data = load_config(subdir, filename)
        if data:
            raw_data[subdir] = data
            available.append(subdir)
            available_labels.append(label)
            available_colors.append(color)
            n = get_num_images(data)
            backend = data['config'].get('backend', 'unknown')
            print(f"  Loaded {subdir}: n={n}, backend={backend}")
        else:
            print(f"  SKIP {subdir}: results not found")

    if len(available) < 2:
        print("ERROR: Need at least 2 configs. Exiting.")
        return

    # Normalize stage keys
    stages_data = {}
    for cfg in available:
        stages_data[cfg] = normalize_stages(raw_data[cfg]['stage_statistics'])

    # Print summary table
    print_summary(available, available_labels, stages_data, raw_data)

    # Generate plots
    saved = []

    # 1. Per-stage comparison (all configs)
    fig = plot_per_stage(available, available_labels, available_colors, stages_data)
    path = OUTPUT_DIR / "cpp_vs_python_per_stage.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)

    # 2. E2E comparison
    fig = plot_e2e(available, available_labels, available_colors, raw_data)
    path = OUTPUT_DIR / "cpp_vs_python_e2e.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)

    # 3. Speedup over CPU baseline
    if 'cpu_quick' in available:
        fig = plot_speedup(available, available_labels, available_colors,
                           stages_data, raw_data, 'cpu_quick')
        if fig:
            path = OUTPUT_DIR / "cpp_vs_python_speedup.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

    # 4. Stacked breakdown
    fig = plot_stacked_breakdown(available, available_labels, available_colors, stages_data)
    path = OUTPUT_DIR / "cpp_vs_python_stacked.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)

    # 5. C++ vs Python TRT head-to-head
    fig = plot_cpp_vs_python_trt(stages_data, raw_data)
    if fig:
        path = OUTPUT_DIR / "cpp_vs_python_trt_headtohead.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)

    print(f"\nSaved {len(saved)} figures:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
