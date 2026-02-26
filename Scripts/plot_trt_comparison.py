#!/usr/bin/env python3
"""
N-way pipeline benchmark comparison: PyTorch vs TRT vs TRT+recon300.

Generates a grouped horizontal bar chart comparing per-stage latencies
across multiple pipeline configurations.

Reads JSON results from Results/Pipeline_Benchmark/<config>/pipeline_results.json

Usage:
    python Scripts/plot_trt_comparison.py
    python Scripts/plot_trt_comparison.py --configs pytorch trt trt_recon300
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Handle display backend issues
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
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results" / "Pipeline_Benchmark"
OUTPUT_DIR = RESULTS_DIR / "figures"

# Default configurations to compare (subdirectory names)
DEFAULT_CONFIGS = ['pytorch', 'trt', 'trt_recon300']

# Human-readable labels for configurations
CONFIG_LABELS = {
    'pytorch': 'PyTorch',
    'gpu_full': 'PyTorch (full)',
    'trt': 'TensorRT',
    'trt_recon300': 'TRT + 300-plane',
    'trt_preload_400hz': 'TRT + VCam 400 Hz',
    'trt_preload_unlimited': 'TRT + VCam Unlimited',
}

# Stage keys in display order
STAGE_KEYS = [
    'stage1_load',
    'stage2_ema',
    'stage3_holographic',
    'stage4_detect',
    'stage5_classify',
    'stage6_postprocess',
]

STAGE_LABELS = {
    'stage1_load': 'Image Load',
    'stage2_ema': 'EMA Background',
    'stage3_holographic': 'Holographic Recon (3-plane)',
    'stage4_detect': 'Heatmap Detection',
    'stage5_classify': 'Classification',
    'stage6_postprocess': 'Post-processing',
    'stage7_recon300': '300-Plane Recon',
}

# Colors for each configuration
CONFIG_COLORS = [
    '#1f77b4',  # blue
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#ff7f0e',  # orange
]


def load_results(config_name: str) -> dict:
    path = RESULTS_DIR / config_name / 'pipeline_results.json'
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def find_available_configs(requested: list[str]) -> list[str]:
    """Filter to only configs that have result files."""
    available = []
    for name in requested:
        path = RESULTS_DIR / name / 'pipeline_results.json'
        if path.exists():
            available.append(name)
        else:
            print(f"  WARNING: {path} not found, skipping '{name}'")
    return available


def plot_comparison(configs: list[str], data: dict[str, dict]) -> plt.Figure:
    """Create N-way grouped horizontal bar chart.

    Args:
        configs: List of config names to compare
        data: Mapping of config_name -> loaded JSON results
    """
    n_configs = len(configs)

    # Determine stages to show (include stage7 if any config has it)
    stage_keys = list(STAGE_KEYS)
    for cfg in configs:
        if 'stage7_recon300' in data[cfg].get('stage_statistics', {}):
            if 'stage7_recon300' not in stage_keys:
                stage_keys.append('stage7_recon300')

    n_stages = len(stage_keys)
    fig, ax = plt.subplots(figsize=(12, max(6, n_stages * 0.8)))

    y = np.arange(n_stages)
    bar_height = 0.8 / n_configs

    for ci, cfg in enumerate(configs):
        stats = data[cfg]['stage_statistics']
        means = []
        errs_low = []
        errs_high = []
        for key in stage_keys:
            if key in stats:
                s = stats[key]
                means.append(s['mean_ms'])
                errs_low.append(max(0, s['mean_ms'] - s['min_ms']))
                errs_high.append(max(0, s['p95_ms'] - s['mean_ms']))
            else:
                means.append(0)
                errs_low.append(0)
                errs_high.append(0)

        offset = (ci - n_configs / 2 + 0.5) * bar_height
        label = CONFIG_LABELS.get(cfg, cfg)
        n_timed = data[cfg]['config'].get('num_timed', '?')
        color = CONFIG_COLORS[ci % len(CONFIG_COLORS)]

        ax.barh(
            y + offset, means, bar_height,
            xerr=[errs_low, errs_high],
            label=f"{label} (n={n_timed})",
            color=color, capsize=2, alpha=0.85,
        )

        # Value labels
        for i, m in enumerate(means):
            if m > 0:
                ax.text(m * 1.05, y[i] + offset, f"{m:.2f}",
                        va='center', ha='left', fontsize=7, color=color)

    # 2.5 ms target line
    ax.axvline(x=2.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='2.5 ms target (400 Hz)')

    labels = [STAGE_LABELS.get(k, k) for k in stage_keys]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_xscale("log")
    ax.set_title("Pipeline Per-Stage Latency: Configuration Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_e2e_comparison(configs: list[str], data: dict[str, dict]) -> plt.Figure:
    """Create end-to-end latency comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 4))

    labels = []
    means = []
    colors = []
    for ci, cfg in enumerate(configs):
        e2e = data[cfg]['end_to_end']
        labels.append(CONFIG_LABELS.get(cfg, cfg))
        means.append(e2e['mean_ms'])
        colors.append(CONFIG_COLORS[ci % len(CONFIG_COLORS)])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, alpha=0.85, width=0.6)

    # Value labels
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{m:.2f} ms\n({1000/m:.0f} FPS)", ha='center', va='bottom', fontsize=9)

    # Target line
    ax.axhline(y=2.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='2.5 ms target (400 Hz)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_title("End-to-End Pipeline Latency Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    return fig


def print_summary(configs: list[str], data: dict[str, dict]):
    """Print a formatted comparison table."""
    # Collect all stage keys
    all_stages = list(STAGE_KEYS)
    for cfg in configs:
        if 'stage7_recon300' in data[cfg].get('stage_statistics', {}):
            if 'stage7_recon300' not in all_stages:
                all_stages.append('stage7_recon300')

    # Header
    col_w = 12
    header = f"{'Stage':<25s}"
    for cfg in configs:
        label = CONFIG_LABELS.get(cfg, cfg)[:col_w]
        header += f" {label:>{col_w}s}"
    sep = "-" * len(header)

    print()
    print("Pipeline Configuration Comparison")
    print(sep)
    print(header)
    print(sep)

    for key in all_stages:
        label = STAGE_LABELS.get(key, key)
        row = f"{label:<25s}"
        for cfg in configs:
            s = data[cfg]['stage_statistics'].get(key)
            if s:
                row += f" {s['mean_ms']:>{col_w}.3f}"
            else:
                row += f" {'N/A':>{col_w}s}"
        print(row)

    # End-to-end
    print(sep)
    row = f"{'End-to-End':<25s}"
    for cfg in configs:
        e2e = data[cfg]['end_to_end']
        row += f" {e2e['mean_ms']:>{col_w}.3f}"
    print(row)

    row = f"{'Throughput (FPS)':<25s}"
    for cfg in configs:
        e2e = data[cfg]['end_to_end']
        row += f" {e2e['throughput_fps']:>{col_w}.1f}"
    print(row)
    print(sep)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='N-way pipeline benchmark comparison plot')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS,
                        help='Configuration subdirectory names to compare')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory for figures')
    args = parser.parse_args()

    # Find available configs
    configs = find_available_configs(args.configs)
    if len(configs) < 2:
        # If we don't have enough of the requested configs, try gpu_full as pytorch
        if 'pytorch' not in configs:
            gpu_full = RESULTS_DIR / 'gpu_full' / 'pipeline_results.json'
            if gpu_full.exists():
                configs.insert(0, 'gpu_full')
                print("  Using gpu_full/ as PyTorch baseline")

    if len(configs) < 1:
        print("ERROR: Need at least 1 config with results. Available:")
        for d in sorted(RESULTS_DIR.iterdir()):
            if (d / 'pipeline_results.json').exists():
                print(f"  {d.name}/")
        return

    # Load data
    data = {}
    for cfg in configs:
        data[cfg] = load_results(cfg)
        print(f"Loaded {cfg}: {data[cfg]['config']['num_timed']} timed images, "
              f"backend={data[cfg]['config'].get('backend', 'pytorch')}")

    # Print text summary
    print_summary(configs, data)

    # Generate figures
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(configs) >= 2:
        fig = plot_comparison(configs, data)
        path = out_dir / "pipeline_trt_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Per-stage comparison saved to {path}")

        fig = plot_e2e_comparison(configs, data)
        path = out_dir / "pipeline_e2e_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"End-to-end comparison saved to {path}")
    else:
        print("Only 1 config available — skipping comparison plots (need >= 2)")


if __name__ == "__main__":
    main()
