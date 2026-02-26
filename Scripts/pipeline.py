"""
Standalone end-to-end holographic pipeline for benchmarking.

Processes real holograms through up to 7 stages:
    1. Load & Resize (CPU) — or GPU ring-buffer via --preload
    2. EMA Enhancement (GPU)
    3. Holographic Reconstruction, 3-plane (GPU)
    4. Heatmap Detection + Peak Finding (GPU) — PyTorch or TensorRT
    5. Classification (GPU) — PyTorch YOLO or TensorRT engine
    6. Post-processing (CPU)
    7. 300-plane Reconstruction (GPU, optional via --recon300)

Usage:
    python Scripts/pipeline.py --data-dir Data/MA/run1 --quick           # 100 images
    python Scripts/pipeline.py --data-dir Data/MA/run1                   # all images
    python Scripts/pipeline.py --data-dir Data/MA/run1 --device cuda:1   # different GPU
    python Scripts/pipeline.py --data-dir Data/MA/run1 --trt             # TensorRT models
    python Scripts/pipeline.py --data-dir Data/MA/run1 --recon300        # + 300-plane recon
    python Scripts/pipeline.py --data-dir Data/MA/run1 --preload --camera-fps 400  # virtual camera
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add Scripts directory to path for local imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from heatmap_model import load_heatmap_detector
from holographic_reconstruction import HolographicReconstructor


# ---------------------------------------------------------------------------
# Stage Timer
# ---------------------------------------------------------------------------

class StageTimer:
    """Per-stage timing with CUDA events for GPU stages and perf_counter for CPU."""

    GPU_STAGES = frozenset({
        'stage2_ema', 'stage3_holographic', 'stage4_detect', 'stage7_recon300',
    })

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = device.type == 'cuda'
        self.records: Dict[str, List[float]] = {}

    def start(self, stage: str) -> dict:
        """Return a timing context to pass to stop()."""
        ctx = {'stage': stage}
        if self.use_cuda and stage in self.GPU_STAGES:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            ctx['start_event'] = start_event
            ctx['end_event'] = end_event
            ctx['gpu'] = True
        else:
            ctx['t0'] = time.perf_counter()
            ctx['gpu'] = False
        return ctx

    def stop(self, ctx: dict):
        """Record elapsed time in milliseconds."""
        stage = ctx['stage']
        if ctx['gpu']:
            ctx['end_event'].record()
            torch.cuda.synchronize()
            elapsed_ms = ctx['start_event'].elapsed_time(ctx['end_event'])
        else:
            elapsed_ms = (time.perf_counter() - ctx['t0']) * 1000.0
        self.records.setdefault(stage, []).append(elapsed_ms)

    def stats(self, stage: str) -> dict:
        """Compute summary statistics for a stage."""
        times = np.array(self.records.get(stage, []))
        if len(times) == 0:
            return {}
        return {
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_fps': float(1000.0 / np.mean(times)) if np.mean(times) > 0 else 0.0,
            'num_samples': len(times),
        }


# ---------------------------------------------------------------------------
# Stage 2: EMA Enhancement
# ---------------------------------------------------------------------------

class EMAEnhancer:
    """GPU-accelerated exponential moving average background subtraction.

    Reimplemented from Ref moving_window_gpu.py (EMA mode only).
    """

    def __init__(self, alpha: float = 0.05, scale_factor: float = 2.0,
                 mean_intensity: float = 105.0, warmup: int = 5,
                 device: torch.device = torch.device('cuda')):
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.mean_intensity = mean_intensity
        self.warmup_target = warmup
        self.device = device
        self.background: Optional[torch.Tensor] = None
        self.frame_count = 0

    def reset(self):
        self.background = None
        self.frame_count = 0

    @property
    def is_warmed_up(self) -> bool:
        return self.frame_count >= self.warmup_target

    def enhance(self, frame: torch.Tensor) -> torch.Tensor:
        """Enhance a single frame.

        Args:
            frame: float32 tensor on GPU, values in [0, 255]

        Returns:
            float32 tensor in [0, 1], or zeros if still warming up
        """
        if self.background is None:
            self.background = frame.clone()
            self.frame_count = 1
            return torch.zeros_like(frame)

        # Update background
        self.background = self.alpha * frame + (1.0 - self.alpha) * self.background
        self.frame_count += 1

        if self.frame_count < self.warmup_target:
            return torch.zeros_like(frame)

        # Enhancement: |frame - background| * scale + mean_intensity
        enhanced = torch.abs(
            (frame - self.background) * self.scale_factor + self.mean_intensity
        )
        enhanced = torch.clamp(enhanced, 0, 255) / 255.0
        return enhanced


# ---------------------------------------------------------------------------
# Stage 4: Heatmap Preprocessing + Peak Detection
# ---------------------------------------------------------------------------

class HeatmapPreprocessor:
    """Fused grayscale-to-3ch + ImageNet normalization on GPU.

    Reimplemented from JetsonPreprocessor in reference pipeline.
    """

    def __init__(self, device: torch.device):
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        # Fused: output = input * scale + offset  (input already in [0,1])
        self.scale = (1.0 / std).to(device)
        self.offset = (-mean / std).to(device)

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        """Convert (H,W) float [0,1] to (1,3,H,W) ImageNet-normalized."""
        x = frame.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)  # (1,3,H,W)
        return x * self.scale + self.offset


class PeakDetector:
    """GPU-accelerated heatmap peak detection via max-pool NMS.

    Reimplemented from JetsonPeakDetector in reference pipeline.
    """

    def __init__(self, threshold: float = 0.3, min_distance: int = 5,
                 topk: int = 50, output_scale: float = 640.0,
                 device: torch.device = torch.device('cuda')):
        self.threshold = threshold
        self.kernel_size = min_distance * 2 + 1  # 11
        self.topk = topk
        self.output_scale = output_scale
        self.min_distance = min_distance
        self.device = device

    def detect(self, heatmap: torch.Tensor) -> List[Dict]:
        """Find peaks in heatmap.

        Args:
            heatmap: (1, 1, H, W) tensor

        Returns:
            List of {x, y, confidence} dicts in output_scale coordinates
        """
        hm = heatmap  # (1,1,H,W)
        hm_h, hm_w = hm.shape[-2:]

        # NMS via max pooling with replicate padding
        padded = F.pad(hm, [self.min_distance] * 4, mode='replicate')
        local_max = F.max_pool2d(padded, self.kernel_size, stride=1)
        peak_mask = (hm == local_max) & (hm > self.threshold)

        # Extract peak values
        peaks_flat = torch.where(peak_mask.view(-1), hm.view(-1), torch.zeros(1, device=self.device))

        # Top-k
        k = min(self.topk, int(peaks_flat.sum().item()) if peaks_flat.sum() > 0 else 1)
        if k == 0:
            return []
        values, indices = torch.topk(peaks_flat, k)

        # Filter zeros
        valid = values > 0
        values = values[valid]
        indices = indices[valid]

        if len(values) == 0:
            return []

        # Convert flat indices to (y, x) coordinates
        y_coords = indices // hm_w
        x_coords = indices % hm_w

        # Scale to output_scale
        scale_x = self.output_scale / hm_w
        scale_y = self.output_scale / hm_h

        detections = []
        for i in range(len(values)):
            detections.append({
                'x': float(x_coords[i].item() * scale_x),
                'y': float(y_coords[i].item() * scale_y),
                'confidence': float(values[i].item()),
            })
        return detections


# ---------------------------------------------------------------------------
# Stage 5: Classification
# ---------------------------------------------------------------------------

class BacteriaClassifier:
    """YOLO v11n-cls wrapper for bacteria classification.

    Loads an ultralytics classification model and classifies 128x128 crops.
    """

    CLASS_NAMES = {0: 'EC', 1: 'PD'}

    def __init__(self, model_path: str, device: str = 'cuda',
                 crop_size: int = 128, conf_threshold: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task='classify')
        self.device = device
        self.crop_size = crop_size
        self.conf_threshold = conf_threshold

    def extract_crops(self, image: np.ndarray, detections: List[Dict],
                      det_scale: float = 640.0, img_scale: float = 448.0) -> List[np.ndarray]:
        """Extract and letterbox 128x128 crops centered on each detection.

        Args:
            image: (H, W) or (H, W, C) uint8 array
            detections: List of {x, y, confidence} in det_scale coordinates
            det_scale: Coordinate space of detections (default 640)
            img_scale: Actual image size (default 448)
        """
        h, w = image.shape[:2]
        scale = img_scale / det_scale
        half = self.crop_size // 2
        crops = []

        for det in detections:
            cx = int(det['x'] * scale)
            cy = int(det['y'] * scale)

            # Compute crop bounds (may go out of image)
            x1 = max(cx - half, 0)
            y1 = max(cy - half, 0)
            x2 = min(cx + half, w)
            y2 = min(cy + half, h)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Letterbox: pad to crop_size x crop_size with gray=114
            ch, cw = crop.shape[:2]
            if len(image.shape) == 3:
                canvas = np.full((self.crop_size, self.crop_size, image.shape[2]), 114, dtype=np.uint8)
            else:
                canvas = np.full((self.crop_size, self.crop_size), 114, dtype=np.uint8)

            pad_y = (self.crop_size - ch) // 2
            pad_x = (self.crop_size - cw) // 2
            canvas[pad_y:pad_y + ch, pad_x:pad_x + cw] = crop
            crops.append(canvas)

        return crops

    def classify(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Classify detections in an image.

        Args:
            image: uint8 (H, W) or (H, W, 3)
            detections: List of {x, y, confidence}

        Returns:
            detections augmented with class_name, class_confidence
        """
        if not detections:
            return detections

        crops = self.extract_crops(image, detections)
        if not crops:
            return detections

        # Convert grayscale crops to RGB for YOLO
        rgb_crops = []
        for crop in crops:
            if len(crop.shape) == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            rgb_crops.append(crop)

        # Batch inference (chunk to max 32 for TRT engine compatibility)
        max_batch = 32
        results = []
        for start in range(0, len(rgb_crops), max_batch):
            chunk = rgb_crops[start:start + max_batch]
            results.extend(self.model.predict(chunk, device=self.device, imgsz=self.crop_size,
                                              verbose=False))

        classified = []
        for i, det in enumerate(detections):
            entry = dict(det)
            if i < len(results):
                probs = results[i].probs
                cls_id = int(probs.top1)
                cls_conf = float(probs.top1conf)
                entry['class_name'] = self.CLASS_NAMES.get(cls_id, f'class_{cls_id}')
                entry['class_confidence'] = cls_conf
            else:
                entry['class_name'] = 'unknown'
                entry['class_confidence'] = 0.0
            classified.append(entry)
        return classified


# ---------------------------------------------------------------------------
# Stage 6: Post-processing
# ---------------------------------------------------------------------------

class StaticCenterpointFilter:
    """Filters out persistent false positive detections.

    Reimplemented from Ref static_centerpoint_filter.py.
    """

    def __init__(self, max_history: int = 200, distance_threshold: float = 50.0,
                 min_hits: int = 5):
        self.history: deque = deque(maxlen=max_history)
        self.distance_threshold = distance_threshold
        self.min_hits = min_hits

    def update_and_filter(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter persistent points.

        Args:
            points: (N, 2) array of (x, y) centerpoints

        Returns:
            (filtered_points, boolean_mask)
        """
        if len(points) == 0:
            return points, np.array([], dtype=bool)

        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Build history array
        if len(self.history) > 0:
            history_array = np.array(list(self.history))
        else:
            history_array = np.empty((0, 2))

        # Append current points to history
        for pt in points:
            self.history.append(pt.copy())

        if len(history_array) == 0:
            return points, np.ones(len(points), dtype=bool)

        # Pairwise distances
        diff = points[:, np.newaxis, :] - history_array[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        hits = np.sum(distances < self.distance_threshold, axis=1)
        mask = hits < self.min_hits
        return points[mask], mask

    def reset(self):
        self.history.clear()


class PostProcessor:
    """Stage 6: Apply static filtering and confidence thresholds."""

    def __init__(self, detection_threshold: float = 0.3,
                 classification_threshold: float = 0.25):
        self.filter = StaticCenterpointFilter(max_history=200, distance_threshold=50.0,
                                              min_hits=5)
        self.detection_threshold = detection_threshold
        self.classification_threshold = classification_threshold
        self.pd_count = 0
        self.ec_count = 0

    def process(self, detections: List[Dict]) -> List[Dict]:
        """Filter and threshold detections.

        Args:
            detections: List of {x, y, confidence, class_name, class_confidence}

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Confidence threshold on detection
        detections = [d for d in detections if d['confidence'] >= self.detection_threshold]
        if not detections:
            return []

        # Static point filtering
        points = np.array([[d['x'], d['y']] for d in detections])
        _, mask = self.filter.update_and_filter(points)

        filtered = [d for d, keep in zip(detections, mask) if keep]

        # Classification confidence threshold
        result = []
        for d in filtered:
            if d.get('class_confidence', 0) >= self.classification_threshold:
                result.append(d)
                if d.get('class_name') == 'PD':
                    self.pd_count += 1
                elif d.get('class_name') == 'EC':
                    self.ec_count += 1

        return result

    def reset(self):
        self.filter.reset()
        self.pd_count = 0
        self.ec_count = 0


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

class HolographicPipeline:
    """End-to-end 6-stage holographic bacteria detection pipeline."""

    def __init__(self, data_dir: str, heatmap_model_path: str,
                 classifier_model_path: str, device: str = 'cuda',
                 num_images: Optional[int] = None, recon300: bool = False,
                 use_trt: bool = False, heatmap_engine: Optional[str] = None,
                 classifier_engine: Optional[str] = None,
                 frame_source=None):
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)
        self.num_images = num_images
        self.recon300 = recon300
        self.use_trt = use_trt
        self.frame_source = frame_source

        # Collect image paths (sorted for reproducibility)
        self.image_paths = sorted(self.data_dir.glob('*.jpg'))
        if not self.image_paths:
            self.image_paths = sorted(self.data_dir.glob('*.png'))
        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]

        print(f"Found {len(self.image_paths)} images in {self.data_dir}")

        # Stage 2: EMA Enhancer
        self.ema = EMAEnhancer(alpha=0.05, scale_factor=2.0, mean_intensity=105.0,
                               warmup=5, device=self.device)

        # Stage 3: Holographic Reconstruction (reuse existing)
        self.reconstructor = HolographicReconstructor(
            resolution=0.087,
            wavelength=405.0 / 1.33 / 1000.0,
            z_start=0, z_step=6, num_planes=3,
            device=str(self.device),
        )

        # Stage 4: Heatmap detection
        self._backend = 'pytorch'
        if use_trt and heatmap_engine:
            print(f"Loading TRT heatmap engine: {heatmap_engine}")
            import torch_tensorrt  # registers TRT custom classes for torch.jit.load
            self.heatmap_model = torch.jit.load(heatmap_engine, map_location=self.device)
            self.heatmap_model.eval()
            self._backend = 'tensorrt'
        else:
            print("Loading heatmap detector...")
            self.heatmap_model = load_heatmap_detector(heatmap_model_path, str(self.device))
        self.preprocessor = HeatmapPreprocessor(self.device)
        self.peak_detector = PeakDetector(threshold=0.3, min_distance=5, topk=50,
                                          output_scale=640.0, device=self.device)

        # Stage 5: Classification
        if use_trt and classifier_engine:
            print(f"Loading TRT classifier engine: {classifier_engine}")
            from ultralytics import YOLO
            self.classifier = BacteriaClassifier.__new__(BacteriaClassifier)
            self.classifier.model = YOLO(classifier_engine, task='classify')
            self.classifier.device = str(self.device)
            self.classifier.crop_size = 128
            self.classifier.conf_threshold = 0.25
            self._backend = 'tensorrt'
        else:
            print("Loading classifier...")
            self.classifier = BacteriaClassifier(classifier_model_path, device=str(self.device),
                                                 crop_size=128, conf_threshold=0.25)

        # Stage 6: Post-processing
        self.postprocessor = PostProcessor(detection_threshold=0.3,
                                           classification_threshold=0.25)

        # Stage 7 (optional): 300-plane holographic reconstruction
        self.reconstructor_300 = None
        if self.recon300:
            print("Initializing 300-plane reconstructor (Hz_stack will be cached)...")
            self.reconstructor_300 = HolographicReconstructor(
                resolution=0.087,
                wavelength=405.0 / 1.33 / 1000.0,
                z_start=0, z_step=0.2, num_planes=300,
                device=str(self.device),
            )

        # Timer
        self.timer = StageTimer(self.device)

    def stage1_load_resize(self, path: Path) -> np.ndarray:
        """Load grayscale image and resize to 448x448."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read {path}")
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
        return img

    def stage2_ema_enhance(self, frame_np: np.ndarray) -> torch.Tensor:
        """EMA background subtraction on GPU. Returns float32 [0,1] (448,448)."""
        frame_gpu = torch.from_numpy(frame_np.astype(np.float32)).to(self.device)
        return self.ema.enhance(frame_gpu)

    def stage3_holographic_reconstruct(self, enhanced: torch.Tensor) -> torch.Tensor:
        """Holographic reconstruction. Returns (3, 448, 448) RGB stack."""
        return self.reconstructor.generate_rgb_stack(enhanced)

    def stage4_detect(self, rgb_stack: torch.Tensor) -> List[Dict]:
        """Heatmap detection + peak finding. Returns detections in 640-space."""
        preprocessed = self.preprocessor(rgb_stack[0])  # Use first channel as grayscale proxy
        # Actually use the full 3-channel stack
        x = rgb_stack.unsqueeze(0)  # (1, 3, 448, 448)
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            heatmap = self.heatmap_model(x)
        return self.peak_detector.detect(heatmap)

    def stage5_classify(self, frame_np: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Classify detections using YOLO."""
        if not detections:
            return detections
        return self.classifier.classify(frame_np, detections)

    def stage6_postprocess(self, detections: List[Dict]) -> List[Dict]:
        """Apply static filtering and confidence thresholds."""
        return self.postprocessor.process(detections)

    def stage7_recon300(self, enhanced: torch.Tensor):
        """300-plane holographic reconstruction (optional, for autofocus benchmarking).

        Output is discarded — this stage exists purely for latency measurement.
        """
        volume = self.reconstructor_300.reconstruct_intensity(enhanced)
        del volume

    def warmup(self, n: int = 20):
        """Run warmup iterations to stabilize GPU clocks and JIT."""
        print(f"Warming up ({n} iterations)...")
        dummy_np = np.random.randint(0, 255, (448, 448), dtype=np.uint8)
        dummy_gpu = torch.from_numpy(dummy_np.astype(np.float32)).to(self.device)

        # Warmup classifier (triggers TRT engine load on first predict())
        dummy_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        self.classifier.model.predict(dummy_crop, device=str(self.device),
                                      imgsz=128, verbose=False)

        for _ in range(n):
            # Warmup EMA
            self.ema.enhance(dummy_gpu)
            # Warmup heatmap model
            with torch.no_grad():
                x = torch.randn(1, 3, 448, 448, device=self.device)
                self.heatmap_model(x)

        # Warmup 300-plane reconstructor if enabled
        if self.reconstructor_300 is not None:
            self.reconstructor_300.reconstruct_intensity(dummy_gpu)

        # Reset EMA state for actual run
        self.ema.reset()
        self.postprocessor.reset()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print("Warmup complete.")

    def run(self) -> Dict:
        """Run the full pipeline on all images. Returns JSON-serializable results."""
        self.warmup()

        total_detections = 0
        total_filtered = 0
        images_with_detections = 0
        ema_warmup_frames = 5

        # Determine iteration source
        use_frame_source = self.frame_source is not None
        n_images = len(self.image_paths)
        print(f"\nProcessing {n_images} images"
              f"{' (preloaded)' if use_frame_source else ''}...")
        t_start = time.perf_counter()

        for idx in range(n_images):
            skip_timing = idx < ema_warmup_frames

            # Stage 1: Load & Resize
            if use_frame_source:
                # FrameSource provides GPU tensor directly — minimal I/O
                if not skip_timing:
                    ctx = self.timer.start('stage1_load')
                frame_gpu = self.frame_source.next_frame()
                # Convert to numpy for classification stage (needs uint8 crops)
                frame_np = (frame_gpu * 255).byte().cpu().numpy()
                if not skip_timing:
                    self.timer.stop(ctx)
            else:
                if not skip_timing:
                    ctx = self.timer.start('stage1_load')
                frame_np = self.stage1_load_resize(self.image_paths[idx])
                if not skip_timing:
                    self.timer.stop(ctx)

            # Stage 2: EMA Enhancement
            if not skip_timing:
                ctx = self.timer.start('stage2_ema')
            if use_frame_source:
                # frame_gpu is already float32 [0,1] on GPU; EMA expects [0,255]
                enhanced = self.ema.enhance(frame_gpu * 255.0)
            else:
                enhanced = self.stage2_ema_enhance(frame_np)
            if not skip_timing:
                self.timer.stop(ctx)

            # Skip remaining stages during EMA warmup (output is zeros)
            if not self.ema.is_warmed_up:
                if (idx + 1) % 100 == 0 or idx == n_images - 1:
                    print(f"  [{idx+1}/{n_images}] (EMA warmup)")
                continue

            # Stage 3: Holographic Reconstruction
            if not skip_timing:
                ctx = self.timer.start('stage3_holographic')
            rgb_stack = self.stage3_holographic_reconstruct(enhanced)
            if not skip_timing:
                self.timer.stop(ctx)

            # Stage 4: Heatmap Detection
            if not skip_timing:
                ctx = self.timer.start('stage4_detect')
            detections = self.stage4_detect(rgb_stack)
            if not skip_timing:
                self.timer.stop(ctx)

            # Stage 5: Classification
            if not skip_timing:
                ctx = self.timer.start('stage5_classify')
            detections = self.stage5_classify(frame_np, detections)
            if not skip_timing:
                self.timer.stop(ctx)

            # Stage 6: Post-processing
            if not skip_timing:
                ctx = self.timer.start('stage6_postprocess')
            filtered = self.stage6_postprocess(detections)
            if not skip_timing:
                self.timer.stop(ctx)

            # Stage 7: 300-plane reconstruction (optional)
            if self.recon300 and self.reconstructor_300 is not None:
                if not skip_timing:
                    ctx = self.timer.start('stage7_recon300')
                self.stage7_recon300(enhanced)
                if not skip_timing:
                    self.timer.stop(ctx)

            total_detections += len(detections)
            total_filtered += len(filtered)
            if len(detections) > 0:
                images_with_detections += 1

            if (idx + 1) % 100 == 0 or idx == n_images - 1:
                print(f"  [{idx+1}/{n_images}] dets={len(detections)} "
                      f"filtered={len(filtered)}")

        t_total = time.perf_counter() - t_start

        # Build results
        stage_names = ['stage1_load', 'stage2_ema', 'stage3_holographic',
                        'stage4_detect', 'stage5_classify', 'stage6_postprocess']
        if self.recon300:
            stage_names.append('stage7_recon300')
        stage_statistics = {}
        for name in stage_names:
            stats = self.timer.stats(name)
            if stats:
                stage_statistics[name] = stats

        # End-to-end: sum per-image means
        e2e_mean = sum(s.get('mean_ms', 0) for s in stage_statistics.values())
        num_timed = len(self.image_paths) - ema_warmup_frames

        # Per-image end-to-end latencies
        e2e_latencies = []
        if num_timed > 0:
            for i in range(num_timed):
                total_ms = 0
                for name in stage_names:
                    times = self.timer.records.get(name, [])
                    if i < len(times):
                        total_ms += times[i]
                e2e_latencies.append(total_ms)

        e2e_arr = np.array(e2e_latencies) if e2e_latencies else np.array([0.0])

        # GPU memory
        gpu_mem = {}
        if self.device.type == 'cuda':
            gpu_mem = {
                'peak_allocated_mb': torch.cuda.max_memory_allocated(self.device) / 1e6,
                'peak_reserved_mb': torch.cuda.max_memory_reserved(self.device) / 1e6,
            }

        # System info
        gpu_name = ''
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)

        results = {
            'system_info': {
                'cpu': platform.processor() or platform.machine(),
                'gpu': gpu_name,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda or 'N/A',
                'python_version': platform.python_version(),
            },
            'config': {
                'num_images': len(self.image_paths),
                'num_timed': num_timed,
                'data_dir': str(self.data_dir),
                'heatmap_model': 'fast_bacteria_hm_optimized.pt',
                'classifier_model': 'bacteria_class.pt',
                'device': str(self.device),
                'backend': self._backend,
                'recon300': self.recon300,
                'ema_alpha': 0.05,
                'ema_warmup': ema_warmup_frames,
                'peak_threshold': 0.3,
                'topk': 50,
                'classification_threshold': 0.25,
                'frame_source': self.frame_source.__class__.__name__ if self.frame_source else 'cv2.imread',
            },
            'stage_statistics': stage_statistics,
            'end_to_end': {
                'mean_ms': float(np.mean(e2e_arr)),
                'median_ms': float(np.median(e2e_arr)),
                'std_ms': float(np.std(e2e_arr)),
                'min_ms': float(np.min(e2e_arr)),
                'max_ms': float(np.max(e2e_arr)),
                'p95_ms': float(np.percentile(e2e_arr, 95)),
                'p99_ms': float(np.percentile(e2e_arr, 99)),
                'throughput_fps': float(1000.0 / np.mean(e2e_arr)) if np.mean(e2e_arr) > 0 else 0.0,
                'wall_clock_seconds': t_total,
            },
            'gpu_memory': gpu_mem,
            'detection_summary': {
                'total_detections': total_detections,
                'total_filtered': total_filtered,
                'images_with_detections': images_with_detections,
                'pd_count': self.postprocessor.pd_count,
                'ec_count': self.postprocessor.ec_count,
            },
        }
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='End-to-end holographic pipeline benchmark')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing hologram images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device (default: cuda)')
    parser.add_argument('--quick', action='store_true',
                        help='Process only 100 images')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Process exactly N images')
    parser.add_argument('--output', type=str, default='Results/Pipeline_Benchmark/pipeline_results.json',
                        help='Output JSON path (default: Results/Pipeline_Benchmark/pipeline_results.json)')
    # Phase 1: 300-plane reconstruction
    parser.add_argument('--recon300', action='store_true',
                        help='Enable 300-plane holographic reconstruction (stage 7)')
    # Phase 2: TensorRT
    parser.add_argument('--trt', action='store_true',
                        help='Use TensorRT engines (auto-discovers .ts/.engine next to .pt weights)')
    parser.add_argument('--heatmap-engine', type=str, default=None,
                        help='Path to TRT heatmap engine (.ts file)')
    parser.add_argument('--classifier-engine', type=str, default=None,
                        help='Path to TRT classifier engine (.engine file)')
    # Phase 3: Virtual camera
    parser.add_argument('--preload', action='store_true',
                        help='Pre-load frames into GPU memory (virtual camera)')
    parser.add_argument('--no-preload', action='store_true',
                        help='Force original cv2.imread path (I/O-inclusive benchmark)')
    parser.add_argument('--camera-fps', type=float, default=None,
                        help='Virtual camera FPS (0=unlimited, 400=real-time). Implies --preload.')
    parser.add_argument('--preload-count', type=int, default=200,
                        help='Number of frames to pre-load into GPU (default: 200)')
    args = parser.parse_args()

    num_images = args.num_images
    if args.quick and num_images is None:
        num_images = 100

    # Default model paths
    ref_dir = PROJECT_ROOT / 'Ref' / 'CurrentPipelineCodes' / 'inference'
    heatmap_path = str(ref_dir / 'fast_bacteria_hm_optimized.pt')
    classifier_path = str(ref_dir / 'bacteria_class.pt')

    # TRT auto-discovery: find .ts/.engine files next to the .pt weights
    heatmap_engine = args.heatmap_engine
    classifier_engine = args.classifier_engine
    if args.trt:
        if heatmap_engine is None:
            candidate = ref_dir / 'heatmap_448_dynamic_32.ts'
            if candidate.exists():
                heatmap_engine = str(candidate)
            else:
                # Also check for .engine file from ultralytics export
                candidate2 = ref_dir / 'heatmap_448_dynamic_32.engine'
                if candidate2.exists():
                    heatmap_engine = str(candidate2)
        if classifier_engine is None:
            candidate = ref_dir / 'bacteria_class.engine'
            if candidate.exists():
                classifier_engine = str(candidate)

    # Virtual camera / frame source
    frame_source = None
    use_preload = args.preload or args.camera_fps is not None
    if use_preload and not args.no_preload:
        from frame_source import FrameSource
        # Determine mode
        if args.camera_fps is not None and args.camera_fps > 0:
            mode = 'fixed'
        else:
            mode = 'sync'
        frame_source = FrameSource(
            data_dir=args.data_dir,
            device=args.device,
            max_frames=args.preload_count,
            mode=mode,
            target_fps=args.camera_fps or 0,
            num_images=num_images,
        )

    pipeline = HolographicPipeline(
        data_dir=args.data_dir,
        heatmap_model_path=heatmap_path,
        classifier_model_path=classifier_path,
        device=args.device,
        num_images=num_images,
        recon300=args.recon300,
        use_trt=args.trt,
        heatmap_engine=heatmap_engine,
        classifier_engine=classifier_engine,
        frame_source=frame_source,
    )

    results = pipeline.run()

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Images processed: {results['config']['num_images']} "
          f"({results['config']['num_timed']} timed)")
    print(f"Device: {results['system_info']['gpu']}")
    print()

    print("Per-stage latency (ms):")
    for stage, stats in results['stage_statistics'].items():
        print(f"  {stage:25s}  mean={stats['mean_ms']:8.2f}  "
              f"median={stats['median_ms']:8.2f}  "
              f"p95={stats['p95_ms']:8.2f}  "
              f"p99={stats['p99_ms']:8.2f}")
    print()

    e2e = results['end_to_end']
    print(f"End-to-end:  mean={e2e['mean_ms']:.2f} ms  "
          f"median={e2e['median_ms']:.2f} ms  "
          f"throughput={e2e['throughput_fps']:.1f} FPS")
    print(f"Wall clock:  {e2e['wall_clock_seconds']:.1f} s")
    print()

    det = results['detection_summary']
    print(f"Detections: {det['total_detections']} total, "
          f"{det['total_filtered']} after filtering")
    print(f"  PD: {det['pd_count']}  EC: {det['ec_count']}  "
          f"Images with detections: {det['images_with_detections']}")

    if results['gpu_memory']:
        mem = results['gpu_memory']
        print(f"\nGPU memory: {mem['peak_allocated_mb']:.0f} MB allocated, "
              f"{mem['peak_reserved_mb']:.0f} MB reserved")

    # Save JSON
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
