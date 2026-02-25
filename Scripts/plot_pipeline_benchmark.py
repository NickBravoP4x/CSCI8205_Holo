#!/usr/bin/env python3
"""
Pipeline benchmark comparison: GPU vs CPU per-stage latency.

Generates a grouped horizontal bar chart comparing GPU and CPU pipeline
stage latencies with speedup annotations.

Part of the CSCI 8205 Multi-GPU Pipeline Architecture project.

Usage:
    python Scripts/plot_pipeline_benchmark.py
"""

import os
import json
import numpy as np
from pathlib import Path

# Handle display backend issues - fallback from Wayland to X11
def setup_display_backend():
    """Setup matplotlib backend with Wayland to X11 fallback."""
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

# --- Paths (relative to repo root) ---
RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results" / "Pipeline_Benchmark"
GPU_JSON = RESULTS_DIR / "gpu_full" / "pipeline_results.json"
CPU_JSON = RESULTS_DIR / "cpu_quick" / "pipeline_results.json"
OUTPUT_DIR = RESULTS_DIR / "figures"

# Stage key -> human-readable label
STAGE_LABELS = {
    "stage1_load": "Image Load",
    "stage2_ema": "EMA Background",
    "stage3_holographic": "Holographic Recon",
    "stage4_detect": "Peak Detection",
    "stage5_classify": "Classification",
    "stage6_postprocess": "Post-processing",
}


def load_results(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def build_comparison_table(gpu_data: dict, cpu_data: dict) -> list[dict]:
    """Build a list of dicts with GPU/CPU stats per stage + end-to-end."""
    rows = []
    gpu_stages = gpu_data["stage_statistics"]
    cpu_stages = cpu_data["stage_statistics"]

    for key, label in STAGE_LABELS.items():
        g = gpu_stages[key]
        c = cpu_stages[key]
        rows.append({
            "label": label,
            "gpu_mean": g["mean_ms"],
            "gpu_min": g["min_ms"],
            "gpu_p95": g["p95_ms"],
            "cpu_mean": c["mean_ms"],
            "cpu_min": c["min_ms"],
            "cpu_p95": c["p95_ms"],
            "speedup": c["mean_ms"] / g["mean_ms"] if g["mean_ms"] > 0 else float("inf"),
        })

    # End-to-end
    ge = gpu_data["end_to_end"]
    ce = cpu_data["end_to_end"]
    rows.append({
        "label": "End-to-End",
        "gpu_mean": ge["mean_ms"],
        "gpu_min": ge["min_ms"],
        "gpu_p95": ge["p95_ms"],
        "cpu_mean": ce["mean_ms"],
        "cpu_min": ce["min_ms"],
        "cpu_p95": ce["p95_ms"],
        "speedup": ce["mean_ms"] / ge["mean_ms"] if ge["mean_ms"] > 0 else float("inf"),
    })

    return rows


def plot_comparison(rows: list[dict], gpu_n: int, cpu_n: int) -> plt.Figure:
    """Create grouped horizontal bar chart with speedup annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [r["label"] for r in rows]
    y = np.arange(len(labels))
    bar_height = 0.35

    gpu_means = [r["gpu_mean"] for r in rows]
    cpu_means = [r["cpu_mean"] for r in rows]

    # Error bars: asymmetric [lower, upper] relative to mean
    gpu_err = np.array([
        [r["gpu_mean"] - r["gpu_min"] for r in rows],
        [r["gpu_p95"] - r["gpu_mean"] for r in rows],
    ])
    cpu_err = np.array([
        [r["cpu_mean"] - r["cpu_min"] for r in rows],
        [r["cpu_p95"] - r["cpu_mean"] for r in rows],
    ])
    # Clamp negative error bars to zero
    gpu_err = np.clip(gpu_err, 0, None)
    cpu_err = np.clip(cpu_err, 0, None)

    ax.barh(y + bar_height / 2, gpu_means, bar_height, xerr=gpu_err,
            label=f"GPU (n={gpu_n:,})", color="#1f77b4", capsize=3, alpha=0.9)
    ax.barh(y - bar_height / 2, cpu_means, bar_height, xerr=cpu_err,
            label=f"CPU (n={cpu_n:,})", color="#d62728", capsize=3, alpha=0.9)

    # Speedup annotations
    for i, r in enumerate(rows):
        x_pos = max(r["gpu_mean"], r["cpu_mean"])
        x_pos = max(x_pos, r.get("gpu_p95", x_pos), r.get("cpu_p95", x_pos))
        ax.text(x_pos * 1.15, y[i], f"{r['speedup']:.1f}x",
                va='center', ha='left', fontsize=9, fontweight='bold',
                color='#2ca02c')

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_xscale("log")
    ax.set_title("Pipeline Per-Stage Latency: GPU vs CPU")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def print_summary_table(rows: list[dict], gpu_n: int, cpu_n: int):
    """Print a formatted comparison table to stdout."""
    hdr = f"{'Stage':<20s} {'GPU (ms)':>10s} {'CPU (ms)':>10s} {'Speedup':>8s}"
    sep = "-" * len(hdr)
    print()
    print(f"Pipeline Benchmark: GPU (n={gpu_n:,}) vs CPU (n={cpu_n:,})")
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        print(f"{r['label']:<20s} {r['gpu_mean']:>10.3f} {r['cpu_mean']:>10.3f} {r['speedup']:>7.1f}x")
    print(sep)
    print()


def main():
    gpu_data = load_results(GPU_JSON)
    cpu_data = load_results(CPU_JSON)

    gpu_n = gpu_data["config"]["num_timed"]
    cpu_n = cpu_data["config"]["num_timed"]

    rows = build_comparison_table(gpu_data, cpu_data)

    # Print text summary
    print_summary_table(rows, gpu_n, cpu_n)

    # Generate figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_comparison(rows, gpu_n, cpu_n)
    out_path = OUTPUT_DIR / "pipeline_gpu_vs_cpu.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
