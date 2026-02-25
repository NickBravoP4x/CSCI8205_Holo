#!/usr/bin/env python3
"""
HoloVision Inference Service - Batched Detection + Classification Edition

Optimized for:
- 200 FPS input with batch size 32 for detection and classification
- Display at 50-100 FPS with bounding boxes (frame skipping)
- Accurate PD/EC counts for ALL frames
- Delay up to 0.5s acceptable for display

Architecture:
- Frame ingestion thread: buffers frames from camera
- Batch processing thread: detection + classification in batches of 32
- Display: sends results to GUI with frame skipping
"""
import zmq
import cv2
import numpy as np
import struct
import time
import torch
import torch.nn.functional as F
from pathlib import Path
import queue
import threading
import csv
import json
from datetime import datetime
from collections import deque
import psutil

# Enable cuDNN benchmark mode for faster convolutions
torch.backends.cudnn.benchmark = True

# RAM monitoring thresholds
RAM_DROP_THRESHOLD = 85.0
RAM_CHECK_INTERVAL = 50

def check_ram_critical():
    """Check if RAM usage exceeds threshold."""
    mem = psutil.virtual_memory()
    return mem.percent >= RAM_DROP_THRESHOLD

# Import local modules
from moving_window_gpu import MovingWindowEnhancer
from static_centerpoint_filter import StaticCenterpointFilter

# Import pipeline components
from heatmap_classifier_pipeline_jetson import (
    TensorRTModel,
    JetsonPreprocessor,
    JetsonPeakDetector,
    YOLOClassifier
)

# TensorRT support
TRT_AVAILABLE = False
try:
    import sys
    sys.path.insert(0, '/usr/lib/python3.10/dist-packages')
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    print("TensorRT not available - heatmap detection requires TensorRT")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Batch processing configuration
BATCH_SIZE = 32                    # Detection batch size
CLASSIFICATION_BATCH_SIZE = 32    # Crops per classification batch
DISPLAY_SKIP = 2                   # Display 1 of every N processed frames
CROP_SIZE = 128                    # Classifier input size

# Queue sizes
PROCESSING_QUEUE_SIZE = 4          # Max batches waiting to be processed
RESULTS_QUEUE_SIZE = 4             # Max result batches waiting for display

# Default model paths (can be overridden via control socket)
DEFAULT_HEATMAP_MODEL = "/home/holomicrobe/Documents/Software/holovision-embeded/services/inference/heatmap_448_dynamic_32.engine"
DEFAULT_CLASSIFIER_MODEL = "/home/holomicrobe/Documents/Software/holovision-embeded/services/inference/bacteria_class.engine"

# =============================================================================
# GLOBAL STATE
# =============================================================================

# Enhancement settings
enhancement_enabled = False  # Controlled by GUI
ENHANCEMENT_METHOD = 'ema'
ENHANCEMENT_WINDOW_SIZE = 10
ENHANCEMENT_EMA_ALPHA = 0.05
ENHANCEMENT_SIZE = 448  # Enhance at model input size (448x448) - one size for all
enhancer = None
current_shape = None
gpu_input_buffer = None

# Detection/Classification state
detection_enabled = False
confidence_threshold = 0.3
classification_threshold = 0.25  # Classification confidence threshold
heatmap_model_path = None
classifier_model_path = None
heatmap_model = None
classifier = None
preprocessor = None
peak_detector = None
pipeline_initialized = False
pipeline_warmed_up = False

# Counts (thread-safe with lock)
counts_lock = threading.Lock()
total_particle_count = 0
pd_count = 0
ec_count = 0
detection_frame_count = 0

# Static centerpoint filter for removing persistent detections (dust, artifacts)
static_filter = None
static_filter_enabled = False

# Frame skip settings
FRAME_SKIP_N = 1  # Process every Nth frame (1 = no skip, 2 = skip half)
raw_frame_counter = 0

# Performance tracking
performance_mode = True
pipeline_fps = 0.0
pipeline_times = deque(maxlen=100)

# Recording state
NUM_WRITER_THREADS = 4
recording = False
recording_folder = None
recording_target_count = 0
recording_frame_counter = 0
recording_start_time = None
writer_queues = []
writer_threads = []
metadata_queue = queue.Queue()
metadata_thread = None
recording_lock = threading.Lock()
fast_recording_mode = False
save_percentage = 1.0
save_start = 0
save_end = 0
saved_frame_counter = 0

# Save mode flags
save_raw = True
save_enhanced = False
save_crops = False
crop_expansion_factor = 1.5

# Writer threads for each mode
enhanced_writer_queues = []
enhanced_writer_threads = []
crops_writer_queue = None
crops_writer_thread = None
crop_counter = 0

# Batch processing queues and threads
processing_queue = None
results_queue = None
batch_processing_thread = None
batch_processing_running = False

# Verbose timing
VERBOSE_TIMING = False
VERBOSE_TIMING_INTERVAL = 100
timing_accum = {
    'batch_total': [], 'detection': [], 'classification': [],
}
timing_frame_count = 0

# Performance diagnostics
PERF_DIAG_ENABLED = True
PERF_DIAG_INTERVAL = 100  # Print every N frames
perf_diag = {
    'frames_received': 0,
    'frames_enhanced': 0,
    'batches_submitted': 0,
    'batches_processed': 0,
    'frames_displayed': 0,
    'enhancement_time_ms': [],
    'batch_pipeline_time_ms': [],  # Detection + classification time per batch
    'batch_wait_frames': 0,
    'last_print_time': 0,
    'image_shape': None,
}

# Display rate limiting to smooth output (prevent bursts)
DISPLAY_TARGET_FPS = 60  # Target display rate
DISPLAY_MIN_INTERVAL = 1.0 / DISPLAY_TARGET_FPS  # Minimum time between display frames
last_display_time = 0

# Display buffer for smoothing output
DISPLAY_BUFFER_SIZE = 16  # Max frames to buffer for display
display_buffer = None  # Initialized in main()


def print_timing_summary():
    global timing_accum, timing_frame_count
    if timing_frame_count == 0:
        return

    print(f"\n{'='*60}")
    print(f"  TIMING SUMMARY (last {timing_frame_count} batches)")
    print(f"{'='*60}")

    total_avg = sum(timing_accum['batch_total']) / len(timing_accum['batch_total']) if timing_accum['batch_total'] else 1

    for stage in ['detection', 'classification']:
        times = timing_accum[stage]
        if times:
            avg = sum(times) / len(times)
            pct = (avg / total_avg * 100) if total_avg > 0 else 0
            print(f"  {stage:<15} {avg:>8.2f}ms ({pct:>5.1f}%)")

    if timing_accum['batch_total']:
        avg_total = sum(timing_accum['batch_total']) / len(timing_accum['batch_total'])
        fps = BATCH_SIZE * 1000 / avg_total if avg_total > 0 else 0
        print(f"  {'TOTAL':<15} {avg_total:>8.2f}ms ({fps:.1f} frames/s)")
    print(f"{'='*60}\n")

    for key in timing_accum:
        timing_accum[key] = []
    timing_frame_count = 0


# =============================================================================
# RECORDING THREADS
# =============================================================================

def image_writer_thread(queue_id, write_queue):
    frames_written = 0
    while True:
        try:
            item = write_queue.get(timeout=1.0)
            if item is None:
                print(f"Writer thread {queue_id} stopping (wrote {frames_written} frames)")
                break
            frame, filename = item
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames_written += 1
        except queue.Empty:
            continue


def metadata_writer_thread(output_folder):
    csv_path = Path(output_folder) / "frame_metadata.csv"
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame_id', 'timestamp_unix', 'timestamp_iso', 'filename'])
            while True:
                try:
                    item = metadata_queue.get(timeout=1.0)
                    if item is None:
                        break
                    frame_id, timestamp_unix, timestamp_str, filename = item
                    writer.writerow([frame_id, f"{timestamp_unix:.6f}", timestamp_str, filename])
                    csvfile.flush()
                except queue.Empty:
                    continue
    except Exception as e:
        print(f"Metadata writer error: {e}")


def crop_writer_thread(write_queue):
    """Thread for writing particle crop images with metadata."""
    crops_written = 0
    while True:
        try:
            item = write_queue.get(timeout=1.0)
            if item is None:
                break
            crop, filename, metadata = item
            cv2.imwrite(filename, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Write metadata sidecar JSON
            if metadata:
                meta_path = filename.replace('.jpg', '.json')
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            crops_written += 1
        except queue.Empty:
            continue
    print(f"Crop writer: {crops_written} crops saved")


def initialize_recording_threads(images_folder, dataset_folder, do_save_raw=True, do_save_enhanced=False, do_save_crops=False):
    global writer_queues, writer_threads, metadata_thread
    global enhanced_writer_queues, enhanced_writer_threads
    global crops_writer_queue, crops_writer_thread
    global crop_counter

    crop_counter = 0

    # Raw image writers
    if do_save_raw:
        writer_queues = [queue.Queue(maxsize=25) for _ in range(NUM_WRITER_THREADS)]
        writer_threads = []
        for i, q in enumerate(writer_queues):
            thread = threading.Thread(target=image_writer_thread, args=(i, q), daemon=True)
            thread.start()
            writer_threads.append(thread)
        print(f"Initialized {NUM_WRITER_THREADS} raw writer threads")
    else:
        writer_queues = []
        writer_threads = []

    # Enhanced image writers
    if do_save_enhanced:
        enhanced_writer_queues = [queue.Queue(maxsize=25) for _ in range(NUM_WRITER_THREADS)]
        enhanced_writer_threads = []
        for i, q in enumerate(enhanced_writer_queues):
            thread = threading.Thread(target=image_writer_thread, args=(i, q), daemon=True)
            thread.start()
            enhanced_writer_threads.append(thread)
        print(f"Initialized {NUM_WRITER_THREADS} enhanced writer threads")
    else:
        enhanced_writer_queues = []
        enhanced_writer_threads = []

    # Crop writer (single thread - smaller images)
    if do_save_crops:
        crops_writer_queue = queue.Queue(maxsize=100)
        crops_writer_thread = threading.Thread(
            target=crop_writer_thread,
            args=(crops_writer_queue,),
            daemon=True
        )
        crops_writer_thread.start()
        print("Initialized crop writer thread")
    else:
        crops_writer_queue = None
        crops_writer_thread = None

    # Metadata writer (always)
    metadata_thread = threading.Thread(target=metadata_writer_thread, args=(dataset_folder,), daemon=True)
    metadata_thread.start()


def stop_recording_threads():
    global writer_queues, writer_threads, metadata_thread
    global enhanced_writer_queues, enhanced_writer_threads
    global crops_writer_queue, crops_writer_thread

    # Stop raw writers
    for q in writer_queues:
        q.put(None)
    for thread in writer_threads:
        thread.join(timeout=2.0)

    # Stop enhanced writers
    for q in enhanced_writer_queues:
        q.put(None)
    for thread in enhanced_writer_threads:
        thread.join(timeout=2.0)

    # Stop crop writer
    if crops_writer_queue:
        crops_writer_queue.put(None)
    if crops_writer_thread:
        crops_writer_thread.join(timeout=2.0)

    # Stop metadata writer
    metadata_queue.put(None)
    if metadata_thread:
        metadata_thread.join(timeout=2.0)

    # Clear references
    writer_queues.clear()
    writer_threads.clear()
    enhanced_writer_queues.clear()
    enhanced_writer_threads.clear()
    crops_writer_queue = None
    crops_writer_thread = None
    metadata_thread = None


# =============================================================================
# BATCH PROCESSING THREAD
# =============================================================================

def batch_processing_worker():
    """
    Worker thread that processes batches of frames through detection + classification.

    Input queue: (batch_images, batch_metadata, batch_originals)
    - batch_images: list of grayscale numpy arrays for detection
    - batch_metadata: list of (timestamp, frame_id, frame_counter, raw_frame, enhanced_frame) tuples
    - batch_originals: list of original numpy arrays for classification crops

    Output queue: (batch_metadata, batch_results, batch_originals, timing_info)

    Note: Status publishing is done in main thread for ZMQ thread-safety.
    """
    global total_particle_count, pd_count, ec_count, detection_frame_count
    global pipeline_fps, pipeline_times, timing_accum, timing_frame_count
    global batch_processing_running, classification_threshold

    print("Batch processing worker started")

    while batch_processing_running:
        try:
            # Get batch from queue
            item = processing_queue.get(timeout=0.5)
            if item is None:
                break

            batch_images, batch_metadata, batch_originals = item
            batch_size = len(batch_images)

            # Always track batch time for FPS calculation
            t_batch_start = time.perf_counter()

            if VERBOSE_TIMING:
                torch.cuda.synchronize()

            # Stage 1: Batched heatmap detection
            if VERBOSE_TIMING:
                torch.cuda.synchronize()
                t_det_start = time.perf_counter()

            batch_tensor = preprocessor.preprocess_batch_optimized(batch_images)

            with torch.no_grad():
                heatmaps = heatmap_model(batch_tensor)

            peak_detector.peak_threshold = confidence_threshold
            peak_results = peak_detector.find_peaks_batch_fused(heatmaps, heatmap_model.heatmap_size)

            # Convert to detection format and apply static filter
            all_detections = []
            for idx, (coords, confidences) in enumerate(peak_results):
                # Apply static filter if enabled
                if static_filter_enabled and static_filter is not None and len(coords) > 0:
                    coords_np = coords.cpu().numpy()
                    conf_np = confidences.cpu().numpy()
                    coords_np, mask = static_filter.update_and_filter(coords_np)
                    conf_np = conf_np[mask]
                else:
                    coords_np = coords.cpu().numpy()
                    conf_np = confidences.cpu().numpy()

                detections = [
                    {'x': float(coords_np[j, 0]), 'y': float(coords_np[j, 1]),
                     'confidence': float(conf_np[j])}
                    for j in range(len(coords_np))
                ]
                all_detections.append(detections)

            if VERBOSE_TIMING:
                torch.cuda.synchronize()
                t_det_end = time.perf_counter()

            # Stage 2: Batched classification across ALL images in batch
            if VERBOSE_TIMING:
                torch.cuda.synchronize()
                t_cls_start = time.perf_counter()

            # Collect all crops with their image indices
            all_crops = []
            crop_to_image_map = []  # (image_idx, detection_idx) for each crop

            # Scale factor: detection coords are in 640x640 space, crops from 448x448
            crop_scale = ENHANCEMENT_SIZE / 640.0  # 448/640 = 0.7

            for img_idx, (orig_image, detections) in enumerate(zip(batch_originals, all_detections)):
                # Scale detection coordinates to crop image space (448x448)
                scaled_detections = []
                for det in detections:
                    scaled_detections.append({
                        'x': det['x'] * crop_scale,
                        'y': det['y'] * crop_scale,
                        'confidence': det['confidence']
                    })
                crops = classifier.extract_crops_from_detections(orig_image, scaled_detections)
                for det_idx, crop in enumerate(crops):
                    all_crops.append(crop)
                    crop_to_image_map.append((img_idx, det_idx))

            # Classify all crops in batches
            all_classifications = []
            if all_crops:
                num_cls_batches = (len(all_crops) + CLASSIFICATION_BATCH_SIZE - 1) // CLASSIFICATION_BATCH_SIZE
                for cls_batch_idx in range(num_cls_batches):
                    start = cls_batch_idx * CLASSIFICATION_BATCH_SIZE
                    end = min(start + CLASSIFICATION_BATCH_SIZE, len(all_crops))
                    batch_crops = all_crops[start:end]
                    batch_classifications = classifier._classify_crops_batch(batch_crops)
                    all_classifications.extend(batch_classifications)

            if VERBOSE_TIMING:
                torch.cuda.synchronize()
                t_cls_end = time.perf_counter()

            # Stage 3: Map classifications back to images and build results
            batch_results = []
            for i in range(batch_size):
                result = {
                    'detections': all_detections[i],
                    'classifications': [None] * len(all_detections[i]),
                    'combined_results': []
                }
                batch_results.append(result)

            for crop_idx, (img_idx, det_idx) in enumerate(crop_to_image_map):
                if crop_idx < len(all_classifications):
                    batch_results[img_idx]['classifications'][det_idx] = all_classifications[crop_idx]

            # Build combined results and update counts
            batch_pd_count = 0
            batch_ec_count = 0
            batch_particle_count = 0

            for result in batch_results:
                for det, cls in zip(result['detections'], result['classifications']):
                    if cls is not None:
                        cls_confidence = cls.get('confidence', 0.0)
                        class_name = cls.get('class_name', 'unknown')

                        # Filter by classification confidence threshold
                        if cls_confidence >= classification_threshold:
                            batch_particle_count += 1
                            result['combined_results'].append({
                                'x': det['x'],
                                'y': det['y'],
                                'detection_confidence': det['confidence'],
                                'class_id': cls.get('class_id', -1),
                                'class_name': class_name,
                                'classification_confidence': cls_confidence
                            })
                            if class_name.upper() == 'PD':
                                batch_pd_count += 1
                            elif class_name.upper() == 'EC':
                                batch_ec_count += 1
                    # Skip detections with cls=None or below threshold

            # Update global counts (thread-safe)
            with counts_lock:
                total_particle_count += batch_particle_count
                pd_count += batch_pd_count
                ec_count += batch_ec_count
                detection_frame_count += batch_size

            # Calculate timing - always measure batch time for FPS
            t_batch_end = time.perf_counter()
            batch_time_ms = (t_batch_end - t_batch_start) * 1000

            if VERBOSE_TIMING:
                torch.cuda.synchronize()
                det_time_ms = (t_det_end - t_det_start) * 1000
                cls_time_ms = (t_cls_end - t_cls_start) * 1000

                timing_accum['detection'].append(det_time_ms)
                timing_accum['classification'].append(cls_time_ms)
                timing_accum['batch_total'].append(batch_time_ms)
                timing_frame_count += 1

                if timing_frame_count >= VERBOSE_TIMING_INTERVAL:
                    print_timing_summary()

                timing_info = {
                    'detection_ms': det_time_ms,
                    'classification_ms': cls_time_ms,
                    'batch_total_ms': batch_time_ms,
                    'num_crops': len(all_crops)
                }
            else:
                timing_info = {'batch_total_ms': batch_time_ms}

            # Track pipeline FPS (frames per second based on batch processing time)
            if batch_time_ms > 0:
                pipeline_times.append(batch_size * 1000.0 / batch_time_ms)
            else:
                pipeline_times.append(0)

            # Note: Status publishing moved to main thread for ZMQ thread-safety

            # Put results in output queue
            try:
                results_queue.put_nowait((batch_metadata, batch_results, batch_originals, timing_info))
            except queue.Full:
                # Drop oldest result to make room
                try:
                    results_queue.get_nowait()
                except queue.Empty:
                    pass
                results_queue.put_nowait((batch_metadata, batch_results, batch_originals, timing_info))

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Batch processing error: {e}")
            import traceback
            traceback.print_exc()

    print("Batch processing worker stopped")


def start_batch_processing():
    """Start the batch processing thread."""
    global processing_queue, results_queue, batch_processing_thread, batch_processing_running

    processing_queue = queue.Queue(maxsize=PROCESSING_QUEUE_SIZE)
    results_queue = queue.Queue(maxsize=RESULTS_QUEUE_SIZE)
    batch_processing_running = True

    batch_processing_thread = threading.Thread(
        target=batch_processing_worker,
        daemon=True
    )
    batch_processing_thread.start()
    print("Batch processing thread started")


def stop_batch_processing():
    """Stop the batch processing thread."""
    global batch_processing_running, batch_processing_thread, processing_queue, results_queue

    print("Stopping batch processing thread...")
    batch_processing_running = False

    # Drain processing queue to make room for stop signal
    if processing_queue:
        try:
            while not processing_queue.empty():
                processing_queue.get_nowait()
        except:
            pass
        try:
            processing_queue.put_nowait(None)  # Signal thread to stop
        except queue.Full:
            pass  # Thread will stop on next timeout anyway

    # Drain results queue
    if results_queue:
        try:
            while not results_queue.empty():
                results_queue.get_nowait()
        except:
            pass

    if batch_processing_thread:
        batch_processing_thread.join(timeout=2.0)
        if batch_processing_thread.is_alive():
            print("Warning: Batch processing thread did not stop cleanly")
        batch_processing_thread = None

    # Clear queues
    processing_queue = None
    results_queue = None
    print("Batch processing thread stopped")


# =============================================================================
# PIPELINE INITIALIZATION
# =============================================================================

def initialize_pipeline(heatmap_path, classifier_path, device='cuda'):
    """Initialize the detection + classification pipeline."""
    global heatmap_model, classifier, preprocessor, peak_detector
    global pipeline_initialized, pipeline_warmed_up
    global heatmap_model_path, classifier_model_path

    print(f"\nInitializing Detection + Classification Pipeline...")
    print(f"  Heatmap model: {heatmap_path}")
    print(f"  Classifier model: {classifier_path}")

    heatmap_model_path = heatmap_path
    classifier_model_path = classifier_path

    # Load heatmap model
    heatmap_model = TensorRTModel(heatmap_path, device)

    # Initialize preprocessor
    preprocessor = JetsonPreprocessor(
        device=device,
        input_size=heatmap_model.input_size,
        single_norm=False
    )

    # Initialize peak detector
    peak_detector = JetsonPeakDetector(
        device=device,
        peak_threshold=confidence_threshold,
        min_distance=5,
        output_scale=640.0,
        topk_k=50
    )

    # Load classifier
    classifier = YOLOClassifier(
        model_path=classifier_path,
        device=device,
        crop_size=CROP_SIZE,
        max_batch_size=CLASSIFICATION_BATCH_SIZE
    )

    pipeline_initialized = True
    pipeline_warmed_up = False

    print("Pipeline initialized successfully")
    return True


def warmup_pipeline():
    """Warmup the pipeline models."""
    global pipeline_warmed_up

    if not pipeline_initialized:
        return False

    print("\nWarming up pipeline...")
    heatmap_model.warmup(20)
    pipeline_warmed_up = True
    print("Pipeline warmup complete")
    return True


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def draw_detections_with_classification(image, detections, classifications, combined_results):
    """Draw bounding boxes with classification labels on image."""
    if len(image.shape) == 2:
        display_frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display_frame = image.copy()

    img_h, img_w = display_frame.shape[:2]
    scale_x = img_w / 640.0
    scale_y = img_h / 640.0

    box_size = 20

    # Color mapping for classes
    class_colors = {
        'PD': (0, 255, 0),      # Green for PD
        'EC': (0, 0, 255),      # Red for EC
        'unknown': (128, 128, 128),  # Gray for unknown
    }

    for result in combined_results:
        x = int(result['x'] * scale_x)
        y = int(result['y'] * scale_y)
        class_name = result.get('class_name', 'unknown').upper()
        det_conf = result.get('detection_confidence', 0)
        cls_conf = result.get('classification_confidence', 0)

        x1 = max(0, x - box_size)
        y1 = max(0, y - box_size)
        x2 = min(img_w - 1, x + box_size)
        y2 = min(img_h - 1, y + box_size)

        color = class_colors.get(class_name, (128, 128, 128))

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

        # Label with class name and confidence
        label = f"{class_name} {cls_conf:.2f}"
        cv2.putText(display_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return display_frame


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    global enhancement_enabled, detection_enabled, confidence_threshold
    global heatmap_model_path, classifier_model_path
    global heatmap_model, classifier, preprocessor, peak_detector
    global pipeline_initialized, pipeline_warmed_up
    global enhancer, current_shape, gpu_input_buffer
    global total_particle_count, pd_count, ec_count, detection_frame_count
    global recording, recording_folder, recording_target_count, recording_frame_counter
    global recording_start_time, fast_recording_mode, performance_mode
    global pipeline_fps, pipeline_times
    global save_start, save_end, save_percentage, saved_frame_counter
    global timing_accum, timing_frame_count
    global static_filter, static_filter_enabled
    global save_raw, save_enhanced, save_crops, crop_expansion_factor, crop_counter
    global enhanced_writer_queues, enhanced_writer_threads, crops_writer_queue, crops_writer_thread
    global raw_frame_counter
    global last_display_time
    global display_buffer

    # Load config for recording parameters
    import yaml
    config_path = Path(__file__).parent.parent.parent / "configs" / "HoloMicrobe.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        crop_expansion_factor = config.get('recording', {}).get('crop_expansion_factor', 1.5)
        print(f"Loaded crop_expansion_factor: {crop_expansion_factor}")

    context = zmq.Context()

    # Subscribe to camera feed
    camera_sub = context.socket(zmq.SUB)
    camera_sub.setsockopt(zmq.SUBSCRIBE, b"")
    camera_sub.setsockopt(zmq.CONFLATE, 0)
    camera_sub.setsockopt(zmq.RCVHWM, 100)
    camera_sub.connect("tcp://localhost:5558")

    # Publish processed frames
    processed_pub = context.socket(zmq.PUB)
    processed_pub.setsockopt(zmq.SNDHWM, 10)
    processed_pub.bind("tcp://*:5559")

    # Publish detection status
    status_pub = context.socket(zmq.PUB)
    status_pub.bind("tcp://*:5561")

    # Control socket
    control_rep = context.socket(zmq.REP)
    control_rep.bind("tcp://*:5560")

    print("HoloVision Inference Service - Batched Pipeline Edition")
    print("=" * 60)
    print(f"Batch size: {BATCH_SIZE} frames")
    print(f"Classification batch: {CLASSIFICATION_BATCH_SIZE} crops")
    print(f"Display skip: 1/{DISPLAY_SKIP}")
    print("=" * 60)
    print("Camera input: port 5558 (SUB)")
    print("Processed output: port 5559 (PUB)")
    print("Status output: port 5561 (PUB)")
    print("Control: port 5560 (REP)")
    print("=" * 60)

    frame_count = 0
    dropped_frames = 0
    FRAME_BUFFER_MAX = 200
    frame_buffer = deque(maxlen=FRAME_BUFFER_MAX)
    ram_check_counter = 0
    frames_dropped_ram = 0

    # Batch accumulation
    batch_images = []
    batch_metadata = []
    batch_originals = []
    batch_frame_counter = 0

    # Display counter for frame skipping
    display_counter = 0

    # Initialize display buffer for smooth output
    display_buffer = deque(maxlen=DISPLAY_BUFFER_SIZE)

    while True:
        try:
            # Handle control commands
            if control_rep.poll(0):
                msg = control_rep.recv_json()
                cmd = msg.get("cmd")

                if cmd == "set_enhancement":
                    enhancement_enabled = msg.get("enabled", False)
                    control_rep.send_json({"status": "ok", "enhancement": enhancement_enabled})
                    print(f"Enhancement {'enabled' if enhancement_enabled else 'disabled'}")
                    if not enhancement_enabled:
                        enhancer = None
                        current_shape = None

                elif cmd == "set_performance_mode":
                    performance_mode = msg.get("enabled", True)
                    control_rep.send_json({"status": "ok", "performance_mode": performance_mode})

                elif cmd == "set_detection":
                    prev_state = detection_enabled
                    detection_enabled = msg.get("enabled", False)
                    confidence_threshold = msg.get("conf", 0.3)
                    classification_threshold = msg.get("cls_conf", 0.25)

                    # Always use hardcoded model paths (ignore GUI model selection)
                    new_heatmap_path = DEFAULT_HEATMAP_MODEL
                    new_classifier_path = DEFAULT_CLASSIFIER_MODEL

                    if detection_enabled and not prev_state:
                        with counts_lock:
                            total_particle_count = 0
                            pd_count = 0
                            ec_count = 0
                            detection_frame_count = 0
                        # Auto-enable static filter when detection starts
                        static_filter_enabled = True
                        static_filter = StaticCenterpointFilter()
                        print("Static centerpoint filter auto-enabled")

                    print(f"\nDetection: {'ENABLE' if detection_enabled else 'DISABLE'}")
                    print(f"  Detection threshold: {confidence_threshold}")
                    print(f"  Classification threshold: {classification_threshold}")
                    print(f"  Heatmap model: {new_heatmap_path}")
                    print(f"  Classifier model: {new_classifier_path}")

                    if detection_enabled and TRT_AVAILABLE:
                        # Initialize pipeline if needed
                        if not pipeline_initialized or new_heatmap_path != heatmap_model_path or new_classifier_path != classifier_model_path:
                            try:
                                initialize_pipeline(new_heatmap_path, new_classifier_path)
                            except Exception as e:
                                print(f"Failed to initialize pipeline: {e}")
                                import traceback
                                traceback.print_exc()
                                detection_enabled = False
                                pipeline_initialized = False

                        # Start batch processing thread (always when enabling detection)
                        if pipeline_initialized and not batch_processing_running:
                            start_batch_processing()

                    elif detection_enabled and not TRT_AVAILABLE:
                        print("TensorRT not available")
                        detection_enabled = False
                    elif not detection_enabled:
                        # Stop batch processing when detection disabled
                        if batch_processing_running:
                            stop_batch_processing()

                    control_rep.send_json({
                        "status": "ok",
                        "detection": detection_enabled,
                        "conf": confidence_threshold,
                        "cls_conf": classification_threshold,
                        "heatmap_model": str(heatmap_model_path) if heatmap_model_path else None,
                        "classifier_model": str(classifier_model_path) if classifier_model_path else None,
                        "pipeline_initialized": pipeline_initialized,
                        "needs_warmup": pipeline_initialized and not pipeline_warmed_up
                    })

                elif cmd == "set_prefilter":
                    # Prefilter not needed - heatmap model is fast
                    control_rep.send_json({
                        "status": "ok",
                        "prefilter_enabled": False,
                        "prefilter_loaded": False
                    })

                elif cmd == "start_recording":
                    folder = msg.get("folder", "data/recordings")
                    count = msg.get("count", 100)
                    fast_mode = msg.get("fast_mode", False)
                    pct = msg.get("save_percentage", 1.0)
                    save_raw = msg.get("save_raw", True)
                    save_enhanced = msg.get("save_enhanced", False)
                    save_crops = msg.get("save_crops", False)

                    with recording_lock:
                        if recording:
                            control_rep.send_json({"status": "error", "message": "Already recording"})
                        else:
                            output_folder = Path(folder)

                            # Create folders based on selected modes
                            if save_raw:
                                (output_folder / "images").mkdir(parents=True, exist_ok=True)
                            if save_enhanced:
                                (output_folder / "enhanced").mkdir(parents=True, exist_ok=True)
                            if save_crops:
                                (output_folder / "crops").mkdir(parents=True, exist_ok=True)
                            (output_folder / "labels").mkdir(parents=True, exist_ok=True)

                            recording = True
                            recording_folder = str(output_folder)
                            recording_target_count = count
                            recording_frame_counter = 0
                            recording_start_time = time.time()
                            fast_recording_mode = fast_mode

                            save_percentage = max(0.01, min(1.0, pct))
                            frames_to_save = int(count * save_percentage)
                            save_start = 0
                            save_end = frames_to_save
                            saved_frame_counter = 0

                            initialize_recording_threads(str(output_folder / "images"), str(output_folder), save_raw, save_enhanced, save_crops)

                            save_modes = []
                            if save_raw:
                                save_modes.append("raw")
                            if save_enhanced:
                                save_modes.append("enhanced")
                            if save_crops:
                                save_modes.append("crops")

                            control_rep.send_json({
                                "status": "started",
                                "folder": str(output_folder),
                                "target_count": count,
                                "fast_mode": fast_mode,
                                "save_percentage": save_percentage,
                                "save_modes": save_modes
                            })
                            print(f"\nRECORDING STARTED - Target: {count} frames, Modes: {save_modes}")

                elif cmd == "stop_recording":
                    with recording_lock:
                        if not recording:
                            control_rep.send_json({"status": "error", "message": "Not recording"})
                        else:
                            recording = False
                            fast_recording_mode = False
                            elapsed_time = time.time() - recording_start_time
                            stop_recording_threads()
                            control_rep.send_json({
                                "status": "stopped",
                                "frames_saved": recording_frame_counter,
                                "elapsed_time": elapsed_time,
                                "folder": recording_folder
                            })
                            print(f"RECORDING STOPPED - {recording_frame_counter} frames in {elapsed_time:.2f}s")

                elif cmd == "recording_status":
                    with recording_lock:
                        control_rep.send_json({
                            "status": "ok",
                            "recording": recording,
                            "frames_recorded": recording_frame_counter,
                            "target_count": recording_target_count,
                            "complete": recording_frame_counter >= recording_target_count if recording else False
                        })

                elif cmd == "get_detection_summary":
                    with counts_lock:
                        current_pd = pd_count
                        current_ec = ec_count
                        current_total = total_particle_count
                    control_rep.send_json({
                        "status": "ok",
                        "detections": {
                            "total": current_total,
                            "pd_count": current_pd,
                            "ec_count": current_ec,
                            "by_class": {"PD": current_pd, "EC": current_ec}
                        },
                        "prefilter": {
                            "enabled": False,
                            "passed": 0,
                            "skipped": 0,
                            "total": 0,
                            "pass_rate": 0.0
                        }
                    })

                elif cmd == "reset_detection_counts":
                    with counts_lock:
                        total_particle_count = 0
                        pd_count = 0
                        ec_count = 0
                        detection_frame_count = 0
                    control_rep.send_json({"status": "ok"})

                elif cmd == "set_static_filter":
                    static_filter_enabled = msg.get("enabled", False)
                    max_history = msg.get("max_history", 500)
                    distance_threshold = msg.get("distance_threshold", 50.0)
                    min_hits = msg.get("min_hits", 5)

                    if static_filter_enabled:
                        static_filter = StaticCenterpointFilter(
                            max_history=max_history,
                            distance_threshold=distance_threshold,
                            min_hits=min_hits
                        )
                        print(f"Static filter enabled: history={max_history}, dist={distance_threshold}, hits={min_hits}")
                    else:
                        static_filter = None
                        print("Static filter disabled")

                    control_rep.send_json({
                        "status": "ok",
                        "static_filter_enabled": static_filter_enabled,
                        "max_history": max_history,
                        "distance_threshold": distance_threshold,
                        "min_hits": min_hits
                    })

                else:
                    control_rep.send_json({"status": "unknown command"})

            # Drain display buffer at steady rate (smooth output)
            if display_buffer:
                current_display_time = time.perf_counter()
                time_since_last = current_display_time - last_display_time
                if time_since_last >= DISPLAY_MIN_INTERVAL:
                    display_frame = display_buffer.popleft()
                    last_display_time = current_display_time

                    # Send buffered frame
                    height, width = display_frame.shape[:2]
                    channels = display_frame.shape[2] if display_frame.ndim == 3 else 1

                    meta_header = struct.pack("dI", time.time(), frame_count)
                    meta = np.array([height, width, channels], dtype=np.uint16).tobytes()
                    processed_pub.send(meta_header + meta + display_frame.astype(np.uint8).tobytes())
                    frame_count += 1

                    if PERF_DIAG_ENABLED:
                        perf_diag['frames_displayed'] += 1

            # Buffer incoming frames
            while camera_sub.poll(0):
                try:
                    incoming_msg = camera_sub.recv(zmq.NOBLOCK)
                    if len(incoming_msg) >= 18:
                        frame_buffer.append(incoming_msg)
                except zmq.Again:
                    break

            # RAM check
            ram_check_counter += 1
            if ram_check_counter >= RAM_CHECK_INTERVAL:
                ram_check_counter = 0
                if check_ram_critical():
                    drop_count = len(frame_buffer) // 2
                    for _ in range(drop_count):
                        if frame_buffer:
                            frame_buffer.popleft()
                            frames_dropped_ram += 1

            # Get next frame
            if frame_buffer:
                msg = frame_buffer.popleft()
            elif camera_sub.poll(10):
                msg = camera_sub.recv()
                if len(msg) < 18:
                    continue
            else:
                # No frame available, check for results to display
                if detection_enabled and results_queue and not results_queue.empty():
                    pass  # Will handle results below
                else:
                    continue

            if len(msg) < 18:
                continue

            # Parse message
            timestamp, frame_id = struct.unpack("dI", msg[:12])
            height, width, channels = np.frombuffer(msg[12:18], dtype=np.uint16)
            payload = msg[18:]

            # Parse frame
            if channels == 1:
                frame = np.frombuffer(payload, dtype=np.uint8).reshape((height, width)).copy()
            elif channels == 3:
                frame = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3)).copy()
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue

            # Keep original for classification (convert gray to 3-channel if needed)
            if channels == 1:
                gray_frame = frame
                original_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                original_frame = frame

            # Fast recording mode (only saves raw frames, no processing)
            if recording and fast_recording_mode:
                with recording_lock:
                    if recording and recording_frame_counter < recording_target_count:
                        current_time = time.time()
                        timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"frame_{recording_frame_counter:06d}_{timestamp_str}.jpg"
                        filepath = str(Path(recording_folder) / "images" / filename)

                        # Only save if raw writers are enabled
                        if writer_queues:
                            thread_id = recording_frame_counter % NUM_WRITER_THREADS
                            try:
                                writer_queues[thread_id].put_nowait((gray_frame.copy(), filepath))
                                metadata_queue.put_nowait((recording_frame_counter, current_time, timestamp_str, filename))
                            except queue.Full:
                                pass

                        recording_frame_counter += 1

                        if recording_frame_counter % 100 == 0:
                            print(f"Fast recording: {recording_frame_counter}/{recording_target_count}")

                        if recording_frame_counter >= recording_target_count:
                            recording = False
                            fast_recording_mode = False
                            elapsed = time.time() - recording_start_time
                            print(f"FAST RECORDING COMPLETE: {recording_frame_counter} frames in {elapsed:.2f}s")
                            stop_recording_threads()
                            status_pub.send_json({"recording_complete": True, "frames_saved": recording_frame_counter})

                meta_header = struct.pack("dI", time.time(), frame_count)
                meta = np.array([height, width, 1], dtype=np.uint16).tobytes()
                processed_pub.send(meta_header + meta + gray_frame.tobytes())
                frame_count += 1
                continue

            # Frame skip for input
            raw_frame_counter += 1
            if FRAME_SKIP_N > 1 and (raw_frame_counter % FRAME_SKIP_N) != 0:
                continue

            # Performance diagnostics
            if PERF_DIAG_ENABLED:
                perf_diag['frames_received'] += 1
                perf_diag['image_shape'] = gray_frame.shape
                t_enhance_start = time.perf_counter()

            # Enhancement (background subtraction) - GPU path with pre-allocated buffers
            # Enhance at 448x448 - same size for detection, crops, everything
            enhanced_448_np = None  # Enhanced at 448x448 for detection and crops
            enhanced_frame_np = None  # For display (resized to original)
            original_h, original_w = gray_frame.shape[:2]

            if enhancement_enabled:
                # Pre-allocate GPU buffer if needed (reuse across frames - fast!)
                if gpu_input_buffer is None or gpu_input_buffer.shape != (ENHANCEMENT_SIZE, ENHANCEMENT_SIZE):
                    gpu_input_buffer = torch.empty((ENHANCEMENT_SIZE, ENHANCEMENT_SIZE), dtype=torch.float32, device='cuda')
                    print(f"Pre-allocated GPU buffer: {ENHANCEMENT_SIZE}x{ENHANCEMENT_SIZE}")

                # CPU resize to 448x448 (one resize only)
                if original_h != ENHANCEMENT_SIZE or original_w != ENHANCEMENT_SIZE:
                    resized_frame = cv2.resize(gray_frame, (ENHANCEMENT_SIZE, ENHANCEMENT_SIZE), interpolation=cv2.INTER_AREA)
                else:
                    resized_frame = gray_frame

                # Copy to pre-allocated GPU buffer (in-place, no allocation)
                gpu_input_buffer.copy_(torch.from_numpy(resized_frame), non_blocking=True)

                # Initialize or update enhancer
                target_shape = (ENHANCEMENT_SIZE, ENHANCEMENT_SIZE)
                if enhancer is None or current_shape != target_shape:
                    enhancer = MovingWindowEnhancer(
                        window_size=ENHANCEMENT_WINDOW_SIZE,
                        method=ENHANCEMENT_METHOD,
                        ema_alpha=ENHANCEMENT_EMA_ALPHA
                    )
                    current_shape = target_shape
                    print(f"Initialized enhancer for shape {ENHANCEMENT_SIZE}x{ENHANCEMENT_SIZE}")

                # Apply enhancement (input: 0-255 GPU tensor, output: 0-1 float)
                enhanced_tensor = enhancer.enhance(gpu_input_buffer, return_float=True)

                # Convert to numpy at 448x448 - used for detection AND crops
                enhanced_448_np = (enhanced_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

            # Performance diagnostics - enhancement time (before display resize)
            if PERF_DIAG_ENABLED:
                perf_diag['frames_enhanced'] += 1
                enhance_time = (time.perf_counter() - t_enhance_start) * 1000
                perf_diag['enhancement_time_ms'].append(enhance_time)

            if enhancement_enabled:
                # For display: resize to original (outside timing - not part of core pipeline)
                enhanced_frame_np = cv2.resize(enhanced_448_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            else:
                # No enhancement - use raw frame
                enhanced_frame_np = gray_frame

            # Detection mode: accumulate frames for batch processing
            if detection_enabled and pipeline_initialized:
                # Warmup on first batch
                if not pipeline_warmed_up:
                    print("\nWarming up pipeline...")
                    warmup_pipeline()
                    # Flush stale frames
                    while camera_sub.poll(0):
                        _ = camera_sub.recv(zmq.NOBLOCK)
                    continue

                # Accumulate frames for batch
                # Use 448x448 enhanced image for detection (already at model size, skips resize)
                if enhancement_enabled and enhanced_448_np is not None:
                    batch_images.append(enhanced_448_np)
                else:
                    batch_images.append(gray_frame)  # Will be resized by preprocessor

                # Metadata includes raw frame for display and enhanced for reference
                batch_metadata.append((timestamp, frame_id, batch_frame_counter, gray_frame.copy(), enhanced_frame_np.copy() if enhanced_frame_np is not None else gray_frame.copy()))

                # For classification crops: use 448x448 enhanced image (BGR)
                if enhancement_enabled and enhanced_448_np is not None:
                    crop_source_frame = cv2.cvtColor(enhanced_448_np, cv2.COLOR_GRAY2BGR)
                else:
                    # No enhancement: resize original to 448x448 for consistent crop coordinates
                    crop_source_frame = cv2.resize(original_frame, (ENHANCEMENT_SIZE, ENHANCEMENT_SIZE), interpolation=cv2.INTER_AREA)
                batch_originals.append(crop_source_frame)
                batch_frame_counter += 1

                # Submit batch when full
                if len(batch_images) >= BATCH_SIZE:
                    if processing_queue is not None:
                        try:
                            processing_queue.put_nowait((
                                batch_images.copy(),
                                batch_metadata.copy(),
                                batch_originals.copy()
                            ))
                            if PERF_DIAG_ENABLED:
                                perf_diag['batches_submitted'] += 1
                        except queue.Full:
                            # Drop batch if queue full (processing can't keep up)
                            print("Warning: Processing queue full, dropping batch")
                        except AttributeError:
                            # Queue was set to None during toggle
                            pass

                    batch_images.clear()
                    batch_metadata.clear()
                    batch_originals.clear()
                else:
                    if PERF_DIAG_ENABLED:
                        perf_diag['batch_wait_frames'] += 1

                # Process results from batch processing thread
                while results_queue and not results_queue.empty():
                    try:
                        result_batch_metadata, result_batch_results, result_batch_originals, timing_info = results_queue.get_nowait()

                        # Track batch pipeline timing
                        if PERF_DIAG_ENABLED and 'batch_total_ms' in timing_info:
                            perf_diag['batch_pipeline_time_ms'].append(timing_info['batch_total_ms'])

                        if PERF_DIAG_ENABLED:
                            perf_diag['batches_processed'] += 1

                        # Display and recording for each frame in batch
                        for i, (meta, result, orig_frame) in enumerate(zip(result_batch_metadata, result_batch_results, result_batch_originals)):
                            # Unpack metadata: (timestamp, frame_id, counter, raw_frame, enhanced_frame)
                            ts, fid, fc, raw_frame, enhanced_frame = meta
                            display_counter += 1

                            # Recording
                            if recording and recording_frame_counter < recording_target_count:
                                current_time = time.time()
                                timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y%m%d_%H%M%S_%f")

                                should_save = (save_start <= recording_frame_counter < save_end)
                                if should_save:
                                    # Save RAW frames
                                    if writer_queues:
                                        filename = f"frame_{recording_frame_counter:06d}_{timestamp_str}.jpg"
                                        filepath = str(Path(recording_folder) / "images" / filename)
                                        thread_id = saved_frame_counter % NUM_WRITER_THREADS
                                        try:
                                            writer_queues[thread_id].put_nowait((raw_frame.copy(), filepath))
                                            metadata_queue.put_nowait((recording_frame_counter, current_time, timestamp_str, filename))
                                        except queue.Full:
                                            pass

                                    # Save ENHANCED frames
                                    if enhanced_writer_queues and enhanced_frame is not None:
                                        filename = f"enhanced_{recording_frame_counter:06d}_{timestamp_str}.jpg"
                                        filepath = str(Path(recording_folder) / "enhanced" / filename)
                                        thread_id = saved_frame_counter % NUM_WRITER_THREADS
                                        try:
                                            enhanced_writer_queues[thread_id].put_nowait((enhanced_frame.copy(), filepath))
                                        except queue.Full:
                                            pass

                                    # Save CROPS with classification
                                    if crops_writer_queue and result['combined_results']:
                                        img_h, img_w = orig_frame.shape[:2]
                                        scale_x = img_w / 640.0
                                        scale_y = img_h / 640.0

                                        for det_idx, cr in enumerate(result['combined_results']):
                                            cx = int(cr['x'] * scale_x)
                                            cy = int(cr['y'] * scale_y)

                                            base_size = 40
                                            crop_half = int(base_size * crop_expansion_factor / 2)

                                            x1 = max(0, cx - crop_half)
                                            y1 = max(0, cy - crop_half)
                                            x2 = min(img_w, cx + crop_half)
                                            y2 = min(img_h, cy + crop_half)

                                            if x2 > x1 and y2 > y1:
                                                crop = orig_frame[y1:y2, x1:x2].copy()
                                                crop_filename = f"crop_{crop_counter:08d}_f{recording_frame_counter:06d}_d{det_idx:02d}_{cr['class_name']}.jpg"
                                                crop_filepath = str(Path(recording_folder) / "crops" / crop_filename)

                                                crop_metadata = {
                                                    "frame_id": recording_frame_counter,
                                                    "detection_idx": det_idx,
                                                    "detection_confidence": cr['detection_confidence'],
                                                    "class_name": cr['class_name'],
                                                    "classification_confidence": cr['classification_confidence'],
                                                    "center_x": cx,
                                                    "center_y": cy,
                                                    "crop_bbox": [x1, y1, x2, y2],
                                                    "expansion_factor": crop_expansion_factor,
                                                    "timestamp": timestamp_str
                                                }

                                                try:
                                                    crops_writer_queue.put_nowait((crop, crop_filepath, crop_metadata))
                                                    crop_counter += 1
                                                except queue.Full:
                                                    pass

                                    # Save YOLO format labels with classification
                                    if result['combined_results']:
                                        label_filename = f"frame_{recording_frame_counter:06d}_{timestamp_str}.txt"
                                        label_filepath = Path(recording_folder) / "labels" / label_filename

                                        img_h, img_w = orig_frame.shape[:2]
                                        bbox_size = 40
                                        bbox_w_norm = bbox_size / img_w
                                        bbox_h_norm = bbox_size / img_h
                                        scale_x = img_w / 640.0
                                        scale_y = img_h / 640.0

                                        # Class mapping: PD=0, EC=1
                                        class_map = {'PD': 0, 'EC': 1}

                                        try:
                                            with open(label_filepath, 'w') as f:
                                                for cr in result['combined_results']:
                                                    cx = cr['x'] * scale_x
                                                    cy = cr['y'] * scale_y
                                                    cx_norm = cx / img_w
                                                    cy_norm = cy / img_h
                                                    class_id = class_map.get(cr['class_name'].upper(), 0)
                                                    f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {bbox_w_norm:.6f} {bbox_h_norm:.6f}\n")
                                        except Exception as e:
                                            print(f"Error writing label: {e}")

                                    saved_frame_counter += 1

                                recording_frame_counter += 1

                                if recording_frame_counter % 100 == 0:
                                    status_pub.send_json({
                                        "recording_progress": recording_frame_counter,
                                        "recording_target": recording_target_count,
                                        "frames_saved": saved_frame_counter
                                    })

                                if recording_frame_counter >= recording_target_count:
                                    recording = False
                                    elapsed_time = time.time() - recording_start_time
                                    stop_recording_threads()
                                    status_pub.send_json({
                                        "recording_complete": True,
                                        "frames_saved": saved_frame_counter,
                                        "elapsed_time": elapsed_time,
                                        "folder": recording_folder
                                    })
                                    print(f"\nRecording complete: {saved_frame_counter} frames in {elapsed_time:.2f}s")

                            # Display skip: only buffer every DISPLAY_SKIP frames
                            if display_counter % DISPLAY_SKIP != 0:
                                continue

                            # Use enhanced frame for display when enhancement is enabled
                            if enhancement_enabled and enhanced_frame is not None:
                                # Convert grayscale enhanced frame to BGR for display
                                display_base = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
                            else:
                                display_base = orig_frame

                            # Draw bounding boxes with classification
                            display_frame = draw_detections_with_classification(
                                display_base,
                                result['detections'],
                                result['classifications'],
                                result['combined_results']
                            )

                            # Buffer frame for smooth display (don't send immediately)
                            if len(display_buffer) < DISPLAY_BUFFER_SIZE:
                                display_buffer.append(display_frame)

                    except queue.Empty:
                        break

                # Send status update from main thread (ZMQ thread-safety)
                with counts_lock:
                    current_pd = pd_count
                    current_ec = ec_count
                    current_total = total_particle_count
                    current_frames = detection_frame_count

                avg_fps = sum(pipeline_times) / len(pipeline_times) if pipeline_times else 0

                status_pub.send_json({
                    "particle_count": current_total,
                    "pd_count": current_pd,
                    "ec_count": current_ec,
                    "generic_count": current_pd,      # PD = generic (non-bacteria)
                    "bacteria_count": current_ec,     # EC = bacteria
                    "pipeline_fps": avg_fps,
                    "frames_processed": current_frames
                })

            else:
                # No detection - pass through enhanced frame if available
                send_frame = enhanced_frame_np if enhanced_frame_np is not None else gray_frame
                status_pub.send_json({
                    "particle_count": total_particle_count,
                    "pd_count": pd_count,
                    "ec_count": ec_count,
                    "generic_count": pd_count,        # PD = generic (non-bacteria)
                    "bacteria_count": ec_count,       # EC = bacteria
                    "pipeline_fps": 0.0,
                    "frames_processed": detection_frame_count
                })

                # Recording (non-detection mode)
                if recording and recording_frame_counter < recording_target_count:
                    current_time = time.time()
                    timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y%m%d_%H%M%S_%f")

                    should_save = (save_start <= recording_frame_counter < save_end)
                    if should_save:
                        # Save RAW frames
                        if writer_queues:
                            filename = f"frame_{recording_frame_counter:06d}_{timestamp_str}.jpg"
                            filepath = str(Path(recording_folder) / "images" / filename)
                            thread_id = saved_frame_counter % NUM_WRITER_THREADS
                            try:
                                writer_queues[thread_id].put_nowait((gray_frame.copy(), filepath))
                                metadata_queue.put_nowait((recording_frame_counter, current_time, timestamp_str, filename))
                            except queue.Full:
                                pass

                        # Save ENHANCED frames
                        if enhanced_writer_queues and enhanced_frame_np is not None:
                            filename = f"enhanced_{recording_frame_counter:06d}_{timestamp_str}.jpg"
                            filepath = str(Path(recording_folder) / "enhanced" / filename)
                            thread_id = saved_frame_counter % NUM_WRITER_THREADS
                            try:
                                enhanced_writer_queues[thread_id].put_nowait((enhanced_frame_np.copy(), filepath))
                            except queue.Full:
                                pass

                        # Note: Crops not saved in non-detection mode (no detections to crop)

                        saved_frame_counter += 1

                    recording_frame_counter += 1

                    if recording_frame_counter % 100 == 0:
                        status_pub.send_json({
                            "recording_progress": recording_frame_counter,
                            "recording_target": recording_target_count,
                            "frames_saved": saved_frame_counter
                        })

                    if recording_frame_counter >= recording_target_count:
                        recording = False
                        elapsed_time = time.time() - recording_start_time
                        stop_recording_threads()
                        status_pub.send_json({
                            "recording_complete": True,
                            "frames_saved": saved_frame_counter,
                            "elapsed_time": elapsed_time,
                            "folder": recording_folder
                        })
                        print(f"\nRecording complete: {saved_frame_counter} frames in {elapsed_time:.2f}s")

                # Display skip
                display_counter += 1
                if DISPLAY_SKIP > 1 and (display_counter % DISPLAY_SKIP) != 0:
                    continue

                height, width = send_frame.shape[:2]
                channels = send_frame.shape[2] if send_frame.ndim == 3 else 1

                meta_header = struct.pack("dI", time.time(), frame_count)
                meta = np.array([height, width, channels], dtype=np.uint16).tobytes()
                processed_pub.send(meta_header + meta + send_frame.astype(np.uint8).tobytes())
                frame_count += 1

            # Periodic logging
            log_interval = 5000 if performance_mode else 1000
            if frame_count % log_interval == 0:
                with counts_lock:
                    current_pd = pd_count
                    current_ec = ec_count
                    current_total = total_particle_count
                print(f"Frames: {frame_count} | PD: {current_pd} | EC: {current_ec} | Total: {current_total}")

            # Performance diagnostics output
            if PERF_DIAG_ENABLED and perf_diag['frames_received'] > 0:
                current_time = time.time()
                if perf_diag['frames_received'] % PERF_DIAG_INTERVAL == 0:
                    elapsed = current_time - perf_diag['last_print_time'] if perf_diag['last_print_time'] > 0 else 1.0

                    avg_enhance_ms = sum(perf_diag['enhancement_time_ms']) / len(perf_diag['enhancement_time_ms']) if perf_diag['enhancement_time_ms'] else 0
                    avg_batch_ms = sum(perf_diag['batch_pipeline_time_ms']) / len(perf_diag['batch_pipeline_time_ms']) if perf_diag['batch_pipeline_time_ms'] else 0
                    batch_fps = (BATCH_SIZE * 1000.0 / avg_batch_ms) if avg_batch_ms > 0 else 0

                    print(f"\n{'='*60}")
                    print(f"PERF DIAG (last {PERF_DIAG_INTERVAL} frames, {elapsed:.2f}s)")
                    if perf_diag['image_shape']:
                        print(f"  Image size:         {perf_diag['image_shape'][1]}x{perf_diag['image_shape'][0]} (enhance @ {ENHANCEMENT_SIZE}x{ENHANCEMENT_SIZE})")
                    print(f"  Frames received:    {perf_diag['frames_received']}")
                    print(f"  Frames enhanced:    {perf_diag['frames_enhanced']}")
                    print(f"  Batches submitted:  {perf_diag['batches_submitted']}")
                    print(f"  Batches processed:  {perf_diag['batches_processed']}")
                    print(f"  Frames displayed:   {perf_diag['frames_displayed']}")
                    print(f"  Batch wait frames:  {perf_diag['batch_wait_frames']}")
                    print(f"  Avg enhance time:   {avg_enhance_ms:.2f}ms")
                    print(f"  Avg batch time:     {avg_batch_ms:.2f}ms ({BATCH_SIZE} frames -> {batch_fps:.1f} FPS)")
                    print(f"  Input FPS:          {PERF_DIAG_INTERVAL / elapsed:.1f}")
                    print(f"  Display FPS:        {perf_diag['frames_displayed'] / elapsed:.1f}")
                    if processing_queue:
                        print(f"  Processing Q size:  {processing_queue.qsize()}")
                    if results_queue:
                        print(f"  Results Q size:     {results_queue.qsize()}")
                    print(f"  Display buffer:     {len(display_buffer)}/{DISPLAY_BUFFER_SIZE}")
                    print(f"{'='*60}\n")

                    # Reset counters
                    perf_diag['frames_received'] = 0
                    perf_diag['frames_enhanced'] = 0
                    perf_diag['batches_submitted'] = 0
                    perf_diag['batches_processed'] = 0
                    perf_diag['frames_displayed'] = 0
                    perf_diag['batch_wait_frames'] = 0
                    perf_diag['enhancement_time_ms'] = []
                    perf_diag['batch_pipeline_time_ms'] = []
                    perf_diag['last_print_time'] = current_time

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    # Cleanup
    if batch_processing_running:
        stop_batch_processing()


if __name__ == "__main__":
    main()
