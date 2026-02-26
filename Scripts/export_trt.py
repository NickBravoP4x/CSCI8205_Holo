#!/usr/bin/env python3
"""
Export heatmap and classifier models to TensorRT engines.

Heatmap model  -> torch_tensorrt.compile() -> TorchScript .ts file
YOLO classifier -> ultralytics model.export(format='engine') -> .engine file

Usage:
    python Scripts/export_trt.py                          # export both
    python Scripts/export_trt.py --heatmap-only           # heatmap only
    python Scripts/export_trt.py --classifier-only        # classifier only
    python Scripts/export_trt.py --device cuda:1          # specific GPU
"""

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

REF_DIR = PROJECT_ROOT / 'Ref' / 'CurrentPipelineCodes' / 'inference'


def export_heatmap(model_path: str, output_path: str, device: str = 'cuda'):
    """Export heatmap detector to TorchScript TRT engine.

    Uses torch_tensorrt.compile() with torchscript IR for reliable serialization.
    The model's forward() is already unrolled for tracing compatibility.
    """
    import torch_tensorrt

    from heatmap_model import load_heatmap_detector

    print(f"Loading heatmap model from {model_path}...")
    model = load_heatmap_detector(model_path, device)
    model.eval()

    # Trace the model first (unrolled forward makes this safe)
    print("Tracing model with torch.jit.trace...")
    example_input = torch.randn(1, 3, 448, 448, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Define dynamic batch input spec
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 3, 448, 448],
            opt_shape=[4, 3, 448, 448],
            max_shape=[32, 3, 448, 448],
            dtype=torch.float32,
        )
    ]

    print("Compiling with torch_tensorrt (this may take a few minutes)...")
    trt_model = torch_tensorrt.compile(
        traced,
        ir='torchscript',
        inputs=inputs,
        enabled_precisions={torch.float32},
        truncate_long_and_double=True,
        workspace_size=1 << 30,  # 1 GB workspace
    )

    # Save as TorchScript
    torch.jit.save(trt_model, output_path)
    print(f"Heatmap TRT engine saved to {output_path}")

    # Verify
    print("Verifying exported model...")
    loaded = torch.jit.load(output_path, map_location=device)
    dummy = torch.randn(1, 3, 448, 448, device=device)
    with torch.no_grad():
        out = loaded(dummy)
    print(f"  Output shape: {list(out.shape)}, range: [{out.min():.4f}, {out.max():.4f}]")
    assert out.shape == (1, 1, 112, 112), f"Shape mismatch: {out.shape}"
    print("  PASS")


def export_classifier(model_path: str, device: str = 'cuda'):
    """Export YOLO classifier to TensorRT engine via ultralytics.

    Uses the officially supported ultralytics export path which handles
    ONNX intermediate conversion automatically.
    """
    from ultralytics import YOLO

    print(f"Loading YOLO classifier from {model_path}...")
    model = YOLO(model_path, task='classify')

    print("Exporting to TensorRT engine (this may take a few minutes)...")
    engine_path = model.export(
        format='engine',
        dynamic=True,
        batch=32,
        half=True,
        imgsz=128,
        device=device,
    )
    print(f"Classifier TRT engine saved to {engine_path}")

    # Verify by loading and running inference
    print("Verifying exported engine...")
    import numpy as np
    engine_model = YOLO(engine_path, task='classify')
    dummy = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    results = engine_model.predict(dummy, device=device, imgsz=128, verbose=False)
    print(f"  Top-1 class: {results[0].probs.top1}, confidence: {results[0].probs.top1conf:.4f}")
    print("  PASS")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description='Export models to TensorRT engines')
    parser.add_argument('--device', type=str, default='cuda',
                        help='CUDA device (default: cuda)')
    parser.add_argument('--heatmap-model', type=str,
                        default=str(REF_DIR / 'fast_bacteria_hm_optimized.pt'),
                        help='Path to heatmap .pt model')
    parser.add_argument('--classifier-model', type=str,
                        default=str(REF_DIR / 'bacteria_class.pt'),
                        help='Path to YOLO classifier .pt model')
    parser.add_argument('--heatmap-output', type=str,
                        default=str(REF_DIR / 'heatmap_448_dynamic_32.ts'),
                        help='Output path for heatmap TRT engine')
    parser.add_argument('--heatmap-only', action='store_true',
                        help='Export only the heatmap model')
    parser.add_argument('--classifier-only', action='store_true',
                        help='Export only the classifier model')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for TensorRT export")
        sys.exit(1)

    export_both = not args.heatmap_only and not args.classifier_only

    if export_both or args.heatmap_only:
        export_heatmap(args.heatmap_model, args.heatmap_output, args.device)
        print()

    if export_both or args.classifier_only:
        export_classifier(args.classifier_model, args.device)

    print("\nAll exports complete.")


if __name__ == '__main__':
    main()
