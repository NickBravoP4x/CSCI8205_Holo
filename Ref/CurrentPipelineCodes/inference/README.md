# HoloVision Inference Service - Live Detection

## Overview

The inference service provides real-time holographic reconstruction and YOLO-based particle detection for live camera feeds.

## Features

### ✅ Implemented
- **GPU Enhancement**: Moving window background subtraction (20-frame buffer)
- **Holographic Reconstruction**: 3-plane intensity stack generation
- **YOLO Detection**: Real-time particle detection on reconstructed stacks
- **Laser Spot Filtering**: IoU-based filtering of persistent false positives
- **Detection Overlay**: Green bounding boxes drawn on live feed
- **Model Warmup**: First-frame initialization to prevent lag spikes

### 🎯 Detection Pipeline

```
Camera Frame (grayscale)
    ↓
Enhancement (optional, GPU)
    ↓
Holographic Reconstruction (3 planes @ 0, 6, 12μm)
    ↓
YOLO Inference (GPU)
    ↓
Laser Spot Filter (IoU tracking)
    ↓
Bounding Box Overlay
    ↓
GUI Display
```

## Architecture

### Files
- **`inference_service.py`**: Main service (ZMQ communication + processing loop)
- **`holographic_reconstruction.py`**: Standalone holographic reconstruction
- **`laser_spot_filter.py`**: Persistent false positive filtering
- **`moving_window_gpu.py`**: GPU-accelerated background enhancement

### ZMQ Ports
- **5558 (SUB)**: Receives raw camera frames from camera service
- **5559 (PUB)**: Publishes processed frames to GUI
- **5560 (REP)**: Control commands from GUI

### Control Commands

#### Enable Enhancement
```python
{
    "cmd": "set_enhancement",
    "enabled": True
}
```

#### Enable Detection
```python
{
    "cmd": "set_detection",
    "enabled": True,
    "conf": 0.5,  # Confidence threshold
    "model": "models/yolo_particles.pt"  # Model path
}
```

## Configuration

### Reconstruction Parameters
```python
HolographicReconstructor(
    resolution=0.087,        # microns per pixel (depends on objective)
    wavelength=0.304,        # 405nm / 1.33 (water) / 1000
    z_start=0,               # starting depth (μm)
    z_step=6,                # depth between planes (μm)
    num_planes=3,            # number of planes for YOLO
    shift_mean=105,          # background normalization
    shift_value=105,
    padding=32,              # FFT padding
    use_fp16=False           # float16 for speed (requires tensor cores)
)
```

### Laser Spot Filter Parameters
```python
LaserSpotFilter(
    max_history=100,         # frames to track
    iou_threshold=0.8,       # 80% overlap for matching
    min_hits=10              # boxes seen ≥10 times are filtered
)
```

### YOLO Model
- **Format**: PyTorch (.pt) or ONNX (.onnx)
- **Input**: 3-channel RGB-style stack (reconstruction planes)
- **Output**: Bounding boxes [x1, y1, x2, y2, confidence, class]
- **Resize**: Automatic to model's expected input size

## Model Requirements

### Directory Structure
```
HoloVision/
└── models/
    ├── yolo_particles.pt      # Your trained YOLO model
    └── yolo11n.pt             # Alternative model
```

### Training Data Format
YOLO models should be trained on 3-plane holographic reconstructions:
- **Channel 0**: Focal plane at z=0μm
- **Channel 1**: Focal plane at z=6μm
- **Channel 2**: Focal plane at z=12μm

## Usage

### 1. Start Services

```bash
# Terminal 1: Camera Service
conda activate holovision_camera
cd /home/p4x/Documents/Software/HoloVision/services/camera
python camera_service.py

# Terminal 2: Inference Service
conda activate holovision_infer
cd /home/p4x/Documents/Software/HoloVision/services/inference
python inference_service.py

# Terminal 3: GUI
conda activate holovision
python gui_launcher.py
```

### 2. Enable Detection in GUI

1. Click "📹 Start Live Camera"
2. Check "🎯 Real-time Detection"
3. Adjust confidence threshold (default 0.5)

### 3. Monitor Performance

**Inference Service Output:**
```
🧠 HoloVision Inference Service
==================================================
📡 Camera input: port 5558 (SUB)
📤 Processed output: port 5559 (PUB)
🎛️  Control: port 5560 (REP)
==================================================
Ready to process frames...
🎯 Detection enabled (conf=0.5)
📦 Loading YOLO model from models/yolo_particles.pt...
✅ YOLO model loaded successfully
🧠 Initialized reconstructor for shape (720, 540)
⚙️  Warming up YOLO model...
🧹 Flushed 15 stale frames
✅ YOLO warmup complete
🎯 Detections: 5 → 3 (filtered)
📊 Processed 100 frames (est. 45.2 FPS)
```

## Performance

### Typical Timings (RTX 3080)
- **Enhancement**: ~1ms
- **Reconstruction (3 planes)**: ~15ms
- **YOLO Inference**: ~10ms
- **Laser Filter + Overlay**: ~1ms
- **Total**: ~27ms → **~37 FPS**

### Optimization Tips
1. **Use FP16**: Set `use_fp16=True` for 2x speedup (requires Ampere+ GPU)
2. **Reduce Planes**: Use 2 planes instead of 3 for faster reconstruction
3. **Smaller Model**: Use YOLO Nano instead of Medium/Large
4. **Lower Resolution**: Downsample camera feed before processing

## Troubleshooting

### "❌ Failed to load YOLO model"
- Check model path exists: `ls models/yolo_particles.pt`
- Verify model format (should be .pt file)
- Check ultralytics installation: `pip install ultralytics`

### "⚠️  System lag: 0.8s"
- GPU is overloaded - reduce resolution or disable enhancement
- Check GPU utilization: `nvidia-smi`
- Consider downsampling camera feed in camera_service.py

### No Detections
- Check confidence threshold (try lowering to 0.1)
- Verify model is trained on holographic reconstructions
- Enable enhancement for better signal-to-noise
- Check laser filter settings (may be too aggressive)

### False Positives (Laser Spots)
- Increase `min_hits` in LaserSpotFilter (default 10 → 20)
- Increase `iou_threshold` (default 0.8 → 0.9)
- Reduce `max_history` for faster adaptation

## Development

### Testing Without Camera
```python
# Create fake frame source for testing
import zmq
import numpy as np
import struct
import time

context = zmq.Context()
pub = context.socket(zmq.PUB)
pub.bind("tcp://*:5558")

while True:
    # Generate test image
    frame = np.random.randint(0, 255, (720, 540), dtype=np.uint8)

    # Send with proper header
    timestamp = time.time()
    frame_id = int(time.time() * 100) % 10000
    meta_header = struct.pack("dI", timestamp, frame_id)
    meta = np.array([720, 540, 1], dtype=np.uint16).tobytes()

    pub.send(meta_header + meta + frame.tobytes())
    time.sleep(0.01)  # 100 FPS
```

### Custom Reconstruction Parameters
Edit `inference_service.py` lines 177-188 to match your optical setup:
```python
reconstructor = HolographicReconstructor(
    resolution=YOUR_MICRONS_PER_PIXEL,
    wavelength=YOUR_WAVELENGTH / YOUR_REFRACTIVE_INDEX / 1000,
    z_start=0,
    z_step=YOUR_Z_STEP,
    num_planes=3,
    ...
)
```

## Next Steps

### Planned Features
- [ ] Particle tracking (assign IDs across frames)
- [ ] Size estimation (from segmentation)
- [ ] Recording with detections (save boxes to JSON)
- [ ] Live statistics (particle count, velocity)
- [ ] Multi-class detection support
- [ ] Confidence histogram visualization

### Integration
- GUI already has detection toggle in `gui/widgets/data_widget.py`
- Statistics tab can display detection counts
- Recording can save both frames and detection metadata
