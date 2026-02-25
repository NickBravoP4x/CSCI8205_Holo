#!/usr/bin/env python3
"""
Jetson-Optimized Heatmap Detection + YOLO Classification Pipeline

Combines:
- GPU-optimized heatmap detection (TensorRT)
- Batched YOLO classification of detected particles

Optimizations for Jetson:
- GPU-based preprocessing (no CPU/numpy in hot path)
- Fused normalization with pre-computed scale/offset
- max_pool2d for fast peak detection
- Batched operations to minimize kernel launches
- TensorRT-first design

Usage:
    python heatmap_classifier_pipeline_jetson.py /path/to/images/ --heatmap-model heatmap.engine --classifier-model classifier.pt
    python heatmap_classifier_pipeline_jetson.py /path/to/images/ --speedtest --detailed
"""

import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

# Enable cuDNN benchmark for faster convolutions
torch.backends.cudnn.benchmark = True

# TensorRT support
TRT_AVAILABLE = False
try:
    sys.path.insert(0, '/usr/lib/python3.10/dist-packages')
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    print("Warning: TensorRT not available")

# YOLO support
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not available, classification disabled")
    YOLO_AVAILABLE = False


class TensorRTModel:
    """Optimized TensorRT inference for Jetson."""

    def __init__(self, engine_path: str, device: str = 'cuda'):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)

        print(f"  Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get binding info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # Check for dynamic batch
        self.dynamic_batch = self.input_shape[0] == -1

        if self.dynamic_batch:
            print(f"  Dynamic batch engine detected")
            self.max_batch = 64
            self.input_buffer = None
            self.output_buffer = None
        else:
            self.max_batch = self.input_shape[0]
            self.input_buffer = torch.empty(self.input_shape, dtype=torch.float32, device=self.device)
            self.output_buffer = torch.empty(self.output_shape, dtype=torch.float32, device=self.device)
            self.context.set_tensor_address(self.input_name, self.input_buffer.data_ptr())
            self.context.set_tensor_address(self.output_name, self.output_buffer.data_ptr())

        # Create a dedicated non-default CUDA stream for TensorRT
        self.torch_stream = torch.cuda.Stream()
        self.stream = self.torch_stream.cuda_stream

        # Extract sizes
        self.input_size = self.input_shape[2] if len(self.input_shape) > 2 else 448
        self.heatmap_size = self.output_shape[2] if len(self.output_shape) > 2 else 112

        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name} {self.output_shape}")
        print(f"  Input size: {self.input_size}x{self.input_size}")
        print(f"  Heatmap size: {self.heatmap_size}x{self.heatmap_size}")

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference - input must already be on GPU."""
        batch_size = input_tensor.shape[0]

        with torch.cuda.stream(self.torch_stream):
            if self.dynamic_batch:
                input_shape = (batch_size,) + self.input_shape[1:]
                output_shape = (batch_size,) + self.output_shape[1:]

                if self.input_buffer is None or self.input_buffer.shape[0] != batch_size:
                    self.input_buffer = torch.empty(input_shape, dtype=torch.float32, device=self.device)
                    self.output_buffer = torch.empty(output_shape, dtype=torch.float32, device=self.device)

                self.input_buffer.copy_(input_tensor)
                self.context.set_input_shape(self.input_name, input_shape)
                self.context.set_tensor_address(self.input_name, self.input_buffer.data_ptr())
                self.context.set_tensor_address(self.output_name, self.output_buffer.data_ptr())
            else:
                self.input_buffer.copy_(input_tensor)

            self.context.execute_async_v3(self.stream)

        self.torch_stream.synchronize()
        return self.output_buffer

    def warmup(self, n: int = 20):
        """Warmup the engine."""
        if self.dynamic_batch:
            dummy = torch.randn(1, 3, self.input_size, self.input_size, dtype=torch.float32, device=self.device)
        else:
            dummy = torch.randn(self.input_shape, dtype=torch.float32, device=self.device)
        for _ in range(n):
            _ = self(dummy)
        torch.cuda.synchronize()
        print(f"  TensorRT engine warmed up ({n} iterations)")


class JetsonPreprocessor:
    """GPU-based preprocessing optimized for Jetson."""

    def __init__(self, device='cuda', input_size: int = 448, single_norm: bool = False):
        self.device = torch.device(device)
        self.input_size = (input_size, input_size)
        self.single_norm = single_norm

        # Fused normalization: (x/255 - mean) / std = x * scale + offset
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.norm_scale = (1.0 / (255.0 * std)).view(1, 3, 1, 1).to(self.device)
        self.norm_offset = (-mean / std).view(1, 3, 1, 1).to(self.device)

        # Single normalization mode for grayscale
        mean_gray = 0.449
        std_gray = 0.226
        self.norm_scale_single = torch.tensor(1.0 / (255.0 * std_gray), device=self.device)
        self.norm_offset_single = torch.tensor(-mean_gray / std_gray, device=self.device)

    def preprocess_batch_optimized(self, images: List[np.ndarray]) -> torch.Tensor:
        """Optimized batch preprocessing - handles grayscale on GPU."""
        batch_size = len(images)

        batch_tensor = torch.empty(
            (batch_size, 3, self.input_size[0], self.input_size[1]),
            dtype=torch.float32,
            device=self.device
        )

        for i, image in enumerate(images):
            is_grayscale = len(image.shape) == 2
            img_h, img_w = image.shape[:2]
            needs_resize = (img_h != self.input_size[0]) or (img_w != self.input_size[1])

            if is_grayscale:
                gpu_img = torch.from_numpy(image).to(self.device)
                float_1chw = gpu_img.unsqueeze(0).unsqueeze(0).float()

                # Skip resize if already at target size
                if needs_resize:
                    resized_1ch = F.interpolate(float_1chw, size=self.input_size, mode='bilinear', align_corners=False)
                else:
                    resized_1ch = float_1chw

                if self.single_norm:
                    normalized = resized_1ch * self.norm_scale_single + self.norm_offset_single
                    batch_tensor[i:i+1] = normalized.expand(-1, 3, -1, -1)
                else:
                    batch_tensor[i, 0:1] = resized_1ch * self.norm_scale[0, 0] + self.norm_offset[0, 0]
                    batch_tensor[i, 1:2] = resized_1ch * self.norm_scale[0, 1] + self.norm_offset[0, 1]
                    batch_tensor[i, 2:3] = resized_1ch * self.norm_scale[0, 2] + self.norm_offset[0, 2]
            else:
                gpu_img = torch.from_numpy(image).to(self.device)
                float_nchw = gpu_img.permute(2, 0, 1).unsqueeze(0).float()

                # Skip resize if already at target size
                if needs_resize:
                    resized = F.interpolate(float_nchw, size=self.input_size, mode='bilinear', align_corners=False)
                else:
                    resized = float_nchw

                batch_tensor[i:i+1] = resized.mul_(self.norm_scale).add_(self.norm_offset)

        return batch_tensor


class JetsonPeakDetector:
    """GPU-based peak detection using max_pool2d."""

    def __init__(self, device='cuda', peak_threshold: float = 0.3, min_distance: int = 5,
                 output_scale: float = 640.0, max_detections: int = 500, topk_k: int = 50):
        self.device = torch.device(device)
        self.peak_threshold = peak_threshold
        self.min_distance = min_distance
        self.output_scale = output_scale
        self.max_detections = max_detections
        self.topk_k = topk_k
        self.kernel_size = min_distance * 2 + 1

    def find_peaks_batch_fused(self, heatmaps: torch.Tensor, heatmap_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Batch-fused peak detection with batched counting."""
        batch_size = heatmaps.shape[0]
        padding = self.min_distance

        # Batch max pooling
        padded = F.pad(heatmaps, [padding] * 4, mode='replicate')
        local_max = F.max_pool2d(padded, self.kernel_size, stride=1)

        # Batch threshold + peak mask
        peak_mask = (heatmaps == local_max) & (heatmaps > self.peak_threshold)
        peaks_only = torch.where(peak_mask, heatmaps, torch.zeros_like(heatmaps))

        # Batch count peaks (single GPU->CPU sync)
        peak_counts = (peaks_only > 0).view(batch_size, -1).sum(dim=1)
        peak_counts_cpu = peak_counts.cpu().numpy()

        results = []
        scale = self.output_scale / heatmap_size
        h, w = heatmaps.shape[2], heatmaps.shape[3]

        for i in range(batch_size):
            num_peaks = int(peak_counts_cpu[i])

            if num_peaks > 0:
                flat = peaks_only[i, 0].flatten()
                k = min(num_peaks, self.topk_k)
                top_values, top_indices = torch.topk(flat, k, sorted=False)

                y_coords = top_indices // w
                x_coords = top_indices % w
                coords = torch.stack([x_coords.float() * scale, y_coords.float() * scale], dim=1)
            else:
                coords = torch.empty((0, 2), device=self.device)
                top_values = torch.empty((0,), device=self.device)

            results.append((coords, top_values))

        return results


class YOLOClassifier:
    """Batched YOLO classifier for detected particles."""

    def __init__(self, model_path: str, device: str = 'cuda', crop_size: int = 128,
                 max_batch_size: int = 32, conf_threshold: float = 0.25):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not available")

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.crop_size = crop_size
        self.max_batch_size = max_batch_size
        self.conf_threshold = conf_threshold
        self.model_path = model_path

        # Check if TensorRT engine
        self.is_tensorrt = model_path.endswith('.engine') or model_path.endswith('.trt')

        print(f"  Loading YOLO Classifier: {model_path}")
        print(f"  Crop size: {crop_size}x{crop_size}")
        print(f"  Format: {'TensorRT' if self.is_tensorrt else 'PyTorch'}")

        self._load_model()
        self._test_batch_classification()

        print(f"  YOLO Classifier initialized")

    def _load_model(self):
        """Load the YOLO classification model."""
        # TensorRT models need explicit task='classify' since it can't be auto-detected
        if self.is_tensorrt:
            self.model = YOLO(self.model_path, task='classify')
        else:
            self.model = YOLO(self.model_path)

        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
            print(f"  Classes: {list(self.class_names.values())}")
        else:
            self.class_names = None

        # TensorRT models don't support .to() - device is passed in predict()
        if not self.is_tensorrt:
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            else:
                self.model.to('cpu')
                self.device = 'cpu'

    def _test_batch_classification(self):
        """Test different batch sizes for classification."""
        test_batch_sizes = [1, 4, 8, 16, 32, 64]
        self.supported_batch_sizes = []

        print(f"  Testing classification batch sizes...")
        for batch_size in test_batch_sizes:
            if batch_size > self.max_batch_size:
                continue

            try:
                dummy_crops = [np.random.randint(0, 255, (self.crop_size, self.crop_size, 3), dtype=np.uint8)
                              for _ in range(batch_size)]
                results = self._classify_crops_batch(dummy_crops)

                if len(results) == batch_size:
                    self.supported_batch_sizes.append(batch_size)
                    print(f"    Batch size {batch_size}: OK")
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(f"    Batch size {batch_size}: OOM")
                    break
                else:
                    print(f"    Batch size {batch_size}: {str(e)[:50]}")

        if not self.supported_batch_sizes:
            self.supported_batch_sizes = [1]

        self.max_supported_batch_size = max(self.supported_batch_sizes)
        print(f"  Supported classification batch sizes: {self.supported_batch_sizes}")

    def extract_crops_from_detections(self, image: np.ndarray, detections: List[Dict]) -> List[np.ndarray]:
        """Extract crops from image based on detection coordinates."""
        crops = []

        for detection in detections:
            x, y = int(detection['x']), int(detection['y'])

            half_crop = self.crop_size // 2
            x1 = max(0, x - half_crop)
            y1 = max(0, y - half_crop)
            x2 = min(image.shape[1], x + half_crop)
            y2 = min(image.shape[0], y + half_crop)

            # Skip invalid crops (zero width or height)
            if x2 <= x1 or y2 <= y1:
                # Create a gray placeholder crop
                if len(image.shape) == 2:
                    placeholder = np.full((self.crop_size, self.crop_size), 114, dtype=np.uint8)
                    placeholder = cv2.cvtColor(placeholder, cv2.COLOR_GRAY2RGB)
                else:
                    placeholder = np.full((self.crop_size, self.crop_size, 3), 114, dtype=np.uint8)
                crops.append(placeholder)
                continue

            crop = image[y1:y2, x1:x2]
            crop_resized = self._letterbox_resize(crop, self.crop_size)
            crops.append(crop_resized)

        return crops

    def _letterbox_resize(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Letterbox resize image to target size."""
        h, w = image.shape[:2]

        # Handle edge case of zero-size image
        if w == 0 or h == 0:
            if len(image.shape) == 2:
                canvas = np.full((target_size, target_size), 114, dtype=np.uint8)
                return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
            else:
                return np.full((target_size, target_size, 3), 114, dtype=np.uint8)

        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Handle grayscale images
        if len(resized.shape) == 2:
            canvas = np.full((target_size, target_size), 114, dtype=np.uint8)
        else:
            canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Convert grayscale to RGB for YOLO
        if len(canvas.shape) == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

        return canvas

    def _classify_crops_batch(self, crops: List[np.ndarray]) -> List[Dict]:
        """Classify a batch of crops using YOLO."""
        if not crops:
            return []

        try:
            # TensorRT requires device and imgsz in predict call
            if self.is_tensorrt:
                results = self.model.predict(
                    crops,
                    conf=self.conf_threshold,
                    verbose=False,
                    device=self.device,
                    imgsz=self.crop_size
                )
            else:
                results = self.model(crops, conf=self.conf_threshold, verbose=False)

            classifications = []
            for result in results:
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs.data.cpu().numpy()
                    top_class = np.argmax(probs)
                    confidence = probs[top_class]

                    class_name = self.class_names[top_class] if self.class_names else str(top_class)

                    classifications.append({
                        'class_id': int(top_class),
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'probabilities': probs.tolist()
                    })
                else:
                    classifications.append({
                        'class_id': -1,
                        'class_name': 'unknown',
                        'confidence': 0.0,
                        'probabilities': []
                    })

            return classifications

        except Exception as e:
            if not hasattr(self, '_error_logged'):
                print(f"  Classification failed: {e}")
                # Debug info - only print once
                if crops:
                    print(f"    Crop shape: {crops[0].shape}, dtype: {crops[0].dtype}")
                    print(f"    Num crops: {len(crops)}, crop_size: {self.crop_size}")
                    print(f"    Model type: {type(self.model)}")
                    print(f"    Is TensorRT: {self.is_tensorrt}")
                    # Try to get raw output
                    try:
                        if self.is_tensorrt:
                            raw = self.model.predict(crops[:1], verbose=True, device=self.device, imgsz=self.crop_size)
                        else:
                            raw = self.model(crops[:1], verbose=True)
                        print(f"    Raw result type: {type(raw)}")
                        if raw:
                            print(f"    Raw result[0] type: {type(raw[0])}")
                            print(f"    Raw result[0] attrs: {[a for a in dir(raw[0]) if not a.startswith('_')]}")
                    except Exception as e2:
                        print(f"    Debug inference also failed: {e2}")
                self._error_logged = True
            return [{'class_id': -1, 'class_name': 'error', 'confidence': 0.0, 'probabilities': []}
                   for _ in crops]

    def classify_detections(self, image: np.ndarray, detections: List[Dict],
                          batch_size: int = None) -> List[Dict]:
        """Classify all detections in an image."""
        if not detections:
            return []

        if batch_size is None:
            batch_size = min(len(detections), self.max_supported_batch_size)

        crops = self.extract_crops_from_detections(image, detections)

        all_classifications = []
        num_batches = (len(crops) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(crops))
            batch_crops = crops[start_idx:end_idx]

            batch_classifications = self._classify_crops_batch(batch_crops)
            all_classifications.extend(batch_classifications)

        return all_classifications


class JetsonClassifierPipeline:
    """Jetson-optimized detection + classification pipeline."""

    def __init__(self, heatmap_model_path: str, classifier_model_path: str,
                 device: str = 'cuda', peak_threshold: float = 0.3,
                 detection_max_batch_size: int = 16, classification_max_batch_size: int = 32,
                 single_norm: bool = False, topk_k: int = 50, crop_size: int = 128):

        self.device = torch.device(device)

        print(f"Initializing Jetson Classifier Pipeline...")
        print(f"  Heatmap model: {heatmap_model_path}")
        print(f"  Classifier model: {classifier_model_path}")
        print(f"  Device: {device}")

        # Check model type
        ext = Path(heatmap_model_path).suffix.lower()
        if ext not in ['.engine', '.trt']:
            raise ValueError(f"Jetson pipeline requires TensorRT engine (.engine), got {ext}")

        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        # Load TensorRT heatmap model
        self.heatmap_model = TensorRTModel(heatmap_model_path, device)
        self.input_size = self.heatmap_model.input_size
        self.heatmap_size = self.heatmap_model.heatmap_size

        # Initialize preprocessor and peak detector
        self.preprocessor = JetsonPreprocessor(device, self.input_size, single_norm=single_norm)
        self.peak_detector = JetsonPeakDetector(
            device, peak_threshold, min_distance=5, output_scale=640.0, topk_k=topk_k
        )

        # Initialize YOLO classifier
        self.classifier = YOLOClassifier(
            model_path=classifier_model_path,
            device=device,
            crop_size=crop_size,
            max_batch_size=classification_max_batch_size
        )

        # Test detection batch sizes
        self.detection_max_batch_size = detection_max_batch_size
        self._test_detection_batch_sizes()

        # Warmup
        self.heatmap_model.warmup(20)

        print(f"Jetson Classifier Pipeline initialized")

    def _test_detection_batch_sizes(self):
        """Test which detection batch sizes work."""
        test_sizes = [1, 2, 4, 8, 16, 32]
        self.supported_detection_batch_sizes = []

        print(f"  Testing detection batch sizes...")
        for bs in test_sizes:
            if bs > self.detection_max_batch_size:
                continue
            if not self.heatmap_model.dynamic_batch and bs != self.heatmap_model.input_shape[0]:
                continue

            try:
                test_input = torch.randn(bs, 3, self.input_size, self.input_size, device=self.device)
                with torch.no_grad():
                    output = self.heatmap_model(test_input)
                torch.cuda.synchronize()

                expected = (bs, 1, self.heatmap_size, self.heatmap_size)
                if output.shape == expected:
                    self.supported_detection_batch_sizes.append(bs)
                del test_input, output
                torch.cuda.empty_cache()

            except Exception as e:
                if "out of memory" in str(e).lower():
                    break

        if not self.supported_detection_batch_sizes:
            self.supported_detection_batch_sizes = [1]

        self.max_supported_detection_batch_size = max(self.supported_detection_batch_sizes)
        print(f"  Supported detection batch sizes: {self.supported_detection_batch_sizes}")

    def process_batch(self, images: List[np.ndarray], original_images: List[np.ndarray],
                      detection_batch_size: int = None,
                      classification_batch_size: int = None) -> List[Dict[str, Any]]:
        """Process a batch through detection + classification pipeline."""

        if detection_batch_size is None:
            detection_batch_size = min(len(images), self.max_supported_detection_batch_size)
        if classification_batch_size is None:
            classification_batch_size = self.classifier.max_supported_batch_size

        # Stage 1: Heatmap detection
        all_detections = []
        num_det_batches = (len(images) + detection_batch_size - 1) // detection_batch_size

        for batch_idx in range(num_det_batches):
            start = batch_idx * detection_batch_size
            end = min(start + detection_batch_size, len(images))
            batch_images = images[start:end]

            # Preprocess
            batch_tensor = self.preprocessor.preprocess_batch_optimized(batch_images)

            # Inference
            with torch.no_grad():
                heatmaps = self.heatmap_model(batch_tensor)

            # Peak detection
            peak_results = self.peak_detector.find_peaks_batch_fused(heatmaps, self.heatmap_size)

            # Convert to detection format
            for coords, confidences in peak_results:
                coords_np = coords.cpu().numpy()
                conf_np = confidences.cpu().numpy()

                detections = [
                    {'x': float(coords_np[j, 0]), 'y': float(coords_np[j, 1]), 'confidence': float(conf_np[j])}
                    for j in range(len(coords_np))
                ]
                all_detections.append(detections)

        # Stage 2: Classification - batch across ALL images
        # Collect all crops with their image indices
        all_crops = []
        crop_to_image_map = []  # (image_idx, detection_idx) for each crop

        for img_idx, (orig_image, detections) in enumerate(zip(original_images, all_detections)):
            crops = self.classifier.extract_crops_from_detections(orig_image, detections)
            for det_idx, crop in enumerate(crops):
                all_crops.append(crop)
                crop_to_image_map.append((img_idx, det_idx))

        # Classify all crops in batches
        all_classifications = []
        if all_crops:
            num_cls_batches = (len(all_crops) + classification_batch_size - 1) // classification_batch_size
            for batch_idx in range(num_cls_batches):
                start = batch_idx * classification_batch_size
                end = min(start + classification_batch_size, len(all_crops))
                batch_crops = all_crops[start:end]
                batch_classifications = self.classifier._classify_crops_batch(batch_crops)
                all_classifications.extend(batch_classifications)

        # Map classifications back to images
        # Initialize results structure
        pipeline_results = []
        for i in range(len(original_images)):
            pipeline_results.append({
                'image_index': i,
                'detections': all_detections[i],
                'classifications': [None] * len(all_detections[i]),
                'combined_results': [],
                'num_particles': len(all_detections[i])
            })

        # Fill in classifications
        for crop_idx, (img_idx, det_idx) in enumerate(crop_to_image_map):
            pipeline_results[img_idx]['classifications'][det_idx] = all_classifications[crop_idx]

        # Build combined results
        for result in pipeline_results:
            for det, cls in zip(result['detections'], result['classifications']):
                if cls is not None:
                    result['combined_results'].append({
                        'x': det['x'],
                        'y': det['y'],
                        'detection_confidence': det['confidence'],
                        'class_id': cls['class_id'],
                        'class_name': cls['class_name'],
                        'classification_confidence': cls['confidence']
                    })

        return pipeline_results

    def process_batch_with_timing(self, images: List[np.ndarray], original_images: List[np.ndarray],
                                   detection_batch_size: int = None,
                                   classification_batch_size: int = None,
                                   detailed: bool = False) -> Tuple[List[Dict], Dict[str, float]]:
        """Process batch with timing breakdown."""

        if detection_batch_size is None:
            detection_batch_size = min(len(images), self.max_supported_detection_batch_size)
        if classification_batch_size is None:
            classification_batch_size = self.classifier.max_supported_batch_size

        torch.cuda.synchronize()
        total_start = time.perf_counter()

        prep_time = 0.0
        inf_time = 0.0
        post_time = 0.0
        crop_time = 0.0
        cls_time = 0.0

        # Stage 1: Heatmap detection
        all_detections = []
        num_det_batches = (len(images) + detection_batch_size - 1) // detection_batch_size

        for batch_idx in range(num_det_batches):
            start = batch_idx * detection_batch_size
            end = min(start + detection_batch_size, len(images))
            batch_images = images[start:end]

            # Preprocess
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch_tensor = self.preprocessor.preprocess_batch_optimized(batch_images)
            torch.cuda.synchronize()
            prep_time += time.perf_counter() - t0

            # Inference
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                heatmaps = self.heatmap_model(batch_tensor)
            torch.cuda.synchronize()
            inf_time += time.perf_counter() - t0

            # Peak detection
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            peak_results = self.peak_detector.find_peaks_batch_fused(heatmaps, self.heatmap_size)

            for coords, confidences in peak_results:
                coords_np = coords.cpu().numpy()
                conf_np = confidences.cpu().numpy()

                detections = [
                    {'x': float(coords_np[j, 0]), 'y': float(coords_np[j, 1]), 'confidence': float(conf_np[j])}
                    for j in range(len(coords_np))
                ]
                all_detections.append(detections)
            torch.cuda.synchronize()
            post_time += time.perf_counter() - t0

        # Stage 2: Classification - batch across ALL images
        # Crop extraction
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        all_crops = []
        crop_to_image_map = []

        for img_idx, (orig_image, detections) in enumerate(zip(original_images, all_detections)):
            crops = self.classifier.extract_crops_from_detections(orig_image, detections)
            for det_idx, crop in enumerate(crops):
                all_crops.append(crop)
                crop_to_image_map.append((img_idx, det_idx))

        torch.cuda.synchronize()
        crop_time = time.perf_counter() - t0

        # Batched classification across all images
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        all_classifications = []
        if all_crops:
            num_cls_batches = (len(all_crops) + classification_batch_size - 1) // classification_batch_size
            for batch_idx in range(num_cls_batches):
                start = batch_idx * classification_batch_size
                end = min(start + classification_batch_size, len(all_crops))
                batch_crops = all_crops[start:end]
                batch_classifications = self.classifier._classify_crops_batch(batch_crops)
                all_classifications.extend(batch_classifications)

        torch.cuda.synchronize()
        cls_time = time.perf_counter() - t0

        # Map classifications back to images
        pipeline_results = []
        for i in range(len(original_images)):
            pipeline_results.append({
                'image_index': i,
                'detections': all_detections[i],
                'classifications': [None] * len(all_detections[i]),
                'combined_results': [],
                'num_particles': len(all_detections[i])
            })

        for crop_idx, (img_idx, det_idx) in enumerate(crop_to_image_map):
            pipeline_results[img_idx]['classifications'][det_idx] = all_classifications[crop_idx]

        for result in pipeline_results:
            for det, cls in zip(result['detections'], result['classifications']):
                if cls is not None:
                    result['combined_results'].append({
                        'x': det['x'],
                        'y': det['y'],
                        'detection_confidence': det['confidence'],
                        'class_id': cls['class_id'],
                        'class_name': cls['class_name'],
                        'classification_confidence': cls['confidence']
                    })

        total_time = (time.perf_counter() - total_start) * 1000

        timing = {
            'preprocessing_ms': prep_time * 1000,
            'inference_ms': inf_time * 1000,
            'postprocessing_ms': post_time * 1000,
            'crop_extraction_ms': crop_time * 1000,
            'classification_ms': cls_time * 1000,
            'total_ms': total_time,
            'images_per_second': len(images) * 1000 / total_time,
            'total_crops': len(all_crops),
            'classification_batches': (len(all_crops) + classification_batch_size - 1) // classification_batch_size if all_crops else 0
        }

        return pipeline_results, timing


def process_folder(pipeline: JetsonClassifierPipeline, folder_path: Path,
                   detection_batch_size: int = None, classification_batch_size: int = None,
                   speedtest: bool = False, detailed: bool = False) -> Dict[str, Any]:
    """Process a folder of images through the pipeline."""

    # Find images
    image_files = sorted(list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")))
    if not image_files:
        raise ValueError(f"No images found in {folder_path}")

    print(f"Found {len(image_files)} images")

    # Load images
    print("Loading images...")
    grayscale_images = []  # For heatmap detection
    original_images = []   # For classification (need color/original)

    for f in image_files:
        img = cv2.imread(str(f))
        if img is not None:
            original_images.append(img)
            # Convert to grayscale for detection
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            grayscale_images.append(gray)

    print(f"Loaded {len(grayscale_images)} images")

    if speedtest:
        print(f"\nRunning pipeline speedtest (6 runs, first is warmup)...")
        times = []

        for run in range(6):
            results, timing = pipeline.process_batch_with_timing(
                grayscale_images, original_images,
                detection_batch_size, classification_batch_size, detailed
            )
            torch.cuda.synchronize()

            if run == 0:
                print(f"  Run {run+1}: {timing['total_ms']:.1f}ms (WARMUP)")
            else:
                times.append(timing)
                print(f"  Run {run+1}: {timing['total_ms']:.1f}ms "
                      f"(prep: {timing['preprocessing_ms']:.1f}, "
                      f"inf: {timing['inference_ms']:.1f}, "
                      f"post: {timing['postprocessing_ms']:.1f}, "
                      f"crop: {timing['crop_extraction_ms']:.1f}, "
                      f"cls: {timing['classification_ms']:.1f})")

        # Calculate stats
        avg_total = np.mean([t['total_ms'] for t in times])
        std_total = np.std([t['total_ms'] for t in times])
        avg_prep = np.mean([t['preprocessing_ms'] for t in times])
        avg_inf = np.mean([t['inference_ms'] for t in times])
        avg_post = np.mean([t['postprocessing_ms'] for t in times])
        avg_crop = np.mean([t['crop_extraction_ms'] for t in times])
        avg_cls = np.mean([t['classification_ms'] for t in times])
        img_per_sec = len(grayscale_images) * 1000 / avg_total
        total_crops = times[0]['total_crops']
        cls_batches = times[0]['classification_batches']

        # Count detections and classifications
        total_detections = sum(r['num_particles'] for r in results)
        class_counts = {}
        for result in results:
            for cls in result['classifications']:
                if cls is not None:
                    name = cls['class_name']
                    class_counts[name] = class_counts.get(name, 0) + 1

        print(f"\nResults (excluding warmup):")
        print(f"  Total: {avg_total:.1f} +/- {std_total:.1f} ms")
        print(f"  Preprocessing: {avg_prep:.1f} ms ({100*avg_prep/avg_total:.1f}%)")
        print(f"  Inference: {avg_inf:.1f} ms ({100*avg_inf/avg_total:.1f}%)")
        print(f"  Postprocessing: {avg_post:.1f} ms ({100*avg_post/avg_total:.1f}%)")
        print(f"  Crop extraction: {avg_crop:.1f} ms ({100*avg_crop/avg_total:.1f}%)")
        print(f"  Classification: {avg_cls:.1f} ms ({100*avg_cls/avg_total:.1f}%)")
        print(f"  Images/second: {img_per_sec:.1f}")
        print(f"  ms/image: {avg_total/len(grayscale_images):.2f}")
        print(f"  Total detections: {total_detections} ({total_detections/len(grayscale_images):.1f} per image)")
        print(f"  Classification batches: {cls_batches} (batch size up to {pipeline.classifier.max_supported_batch_size})")
        print(f"  Classification breakdown:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count}")

        return {
            'num_images': len(grayscale_images),
            'avg_total_ms': avg_total,
            'std_total_ms': std_total,
            'images_per_second': img_per_sec,
            'total_detections': total_detections,
            'class_counts': class_counts
        }
    else:
        results, timing = pipeline.process_batch_with_timing(
            grayscale_images, original_images,
            detection_batch_size, classification_batch_size, detailed
        )

        total_detections = sum(r['num_particles'] for r in results)
        class_counts = {}
        for result in results:
            for cls in result['classifications']:
                name = cls['class_name']
                class_counts[name] = class_counts.get(name, 0) + 1

        print(f"\nResults:")
        print(f"  Total time: {timing['total_ms']:.1f} ms")
        print(f"  Images/second: {timing['images_per_second']:.1f}")
        print(f"  Total detections: {total_detections}")
        print(f"  Classification breakdown:")
        for class_name, count in sorted(class_counts.items()):
            print(f"    {class_name}: {count}")

        return {
            'num_images': len(grayscale_images),
            'timing': timing,
            'total_detections': total_detections,
            'class_counts': class_counts,
            'results': results
        }


def main():
    parser = argparse.ArgumentParser(description="Jetson-Optimized Detection + Classification Pipeline")
    parser.add_argument('input', type=str, help='Input directory containing images')
    parser.add_argument('--heatmap-model', type=str, required=True,
                       help='Path to TensorRT heatmap engine (.engine)')
    parser.add_argument('--classifier-model', type=str, required=True,
                       help='Path to YOLO classifier model (.pt)')
    parser.add_argument('--detection-batch-size', type=int, default=None,
                       help='Batch size for detection (default: auto)')
    parser.add_argument('--classification-batch-size', type=int, default=None,
                       help='Batch size for classification (default: auto)')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Peak detection threshold (default: 0.3)')
    parser.add_argument('--detection-max-batch-size', type=int, default=16,
                       help='Maximum detection batch size (default: 16)')
    parser.add_argument('--classification-max-batch-size', type=int, default=32,
                       help='Maximum classification batch size (default: 32)')
    parser.add_argument('--crop-size', type=int, default=128,
                       help='Crop size for classification (default: 128, must match classifier export size)')
    parser.add_argument('--single-norm', action='store_true',
                       help='Use single normalization for grayscale (faster)')
    parser.add_argument('--topk-k', type=int, default=50,
                       help='Fixed k for topk peak detection (default: 50)')
    parser.add_argument('--speedtest', action='store_true',
                       help='Run speedtest with multiple iterations')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed timing breakdown')
    parser.add_argument('--output', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--test-classifier', action='store_true',
                       help='Test classifier model only (debugging)')

    args = parser.parse_args()

    # Test classifier only mode
    if args.test_classifier:
        print(f"Testing classifier: {args.classifier_model}")
        print(f"Crop size: {args.crop_size}")

        from ultralytics import YOLO
        is_trt = args.classifier_model.endswith('.engine') or args.classifier_model.endswith('.trt')

        # TensorRT needs explicit task='classify'
        if is_trt:
            model = YOLO(args.classifier_model, task='classify')
        else:
            model = YOLO(args.classifier_model)

        print(f"Model type: {type(model)}")
        print(f"Model names: {model.names if hasattr(model, 'names') else 'N/A'}")
        print(f"Is TensorRT: {is_trt}")

        # Create test image
        test_img = np.random.randint(0, 255, (args.crop_size, args.crop_size, 3), dtype=np.uint8)
        print(f"Test image shape: {test_img.shape}")

        try:
            if is_trt:
                results = model.predict([test_img], verbose=True, device='cuda', imgsz=args.crop_size)
            else:
                results = model([test_img], verbose=True)

            print(f"Results type: {type(results)}")
            print(f"Results length: {len(results)}")

            if results:
                r = results[0]
                print(f"Result[0] type: {type(r)}")
                print(f"Result[0] attributes: {[a for a in dir(r) if not a.startswith('_')]}")

                if hasattr(r, 'probs'):
                    print(f"  probs: {r.probs}")
                    if r.probs is not None:
                        print(f"  probs.data: {r.probs.data}")
                        print(f"  probs.data.shape: {r.probs.data.shape}")

                if hasattr(r, 'boxes'):
                    print(f"  boxes: {r.boxes}")

                if hasattr(r, 'orig_shape'):
                    print(f"  orig_shape: {r.orig_shape}")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()

        sys.exit(0)

    # Validate
    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory")
        sys.exit(1)

    if not Path(args.heatmap_model).exists():
        print(f"Error: Heatmap model not found: {args.heatmap_model}")
        sys.exit(1)

    if not Path(args.classifier_model).exists():
        print(f"Error: Classifier model not found: {args.classifier_model}")
        sys.exit(1)

    # Initialize pipeline
    pipeline = JetsonClassifierPipeline(
        heatmap_model_path=args.heatmap_model,
        classifier_model_path=args.classifier_model,
        peak_threshold=args.threshold,
        detection_max_batch_size=args.detection_max_batch_size,
        classification_max_batch_size=args.classification_max_batch_size,
        single_norm=args.single_norm,
        topk_k=args.topk_k,
        crop_size=args.crop_size
    )

    # Process
    results = process_folder(
        pipeline, input_path,
        args.detection_batch_size, args.classification_batch_size,
        args.speedtest, args.detailed
    )

    # Save results
    if args.output:
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
            elif isinstance(obj, list):
                return [clean(x) for x in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        with open(args.output, 'w') as f:
            json.dump(clean(results), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
