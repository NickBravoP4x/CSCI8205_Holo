"""
Virtual camera / frame source for pipeline benchmarking.

Pre-loads hologram images into GPU memory as float32 tensors, eliminating
disk I/O from the benchmark loop to reveal true compute bottlenecks.

Modes:
    sync    (default): yield pre-loaded GPU frames as fast as pipeline consumes
    fixed:             yield at --camera-fps with time.sleep() rate limiting
    threaded:          background thread pushes into deque at target FPS

Usage in pipeline:
    source = FrameSource(data_dir="Data/MA/run1", device="cuda", max_frames=200)
    for i in range(n):
        frame = source.next_frame()   # float32 [0,1] tensor on GPU, shape (448,448)
"""

import time
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


class FrameSource:
    """GPU-resident frame source with ring-buffer cycling.

    Pre-loads up to `max_frames` images into GPU memory as float32 [0,1] tensors.
    Ring-buffer cycles for runs longer than max_frames.

    Memory: 200 frames x 448x448 x 4 bytes = ~153 MB GPU.
    """

    def __init__(
        self,
        data_dir: str,
        device: str = 'cuda',
        max_frames: int = 200,
        mode: str = 'sync',
        target_fps: float = 0,
        num_images: Optional[int] = None,
        img_size: int = 448,
    ):
        """
        Args:
            data_dir: Directory containing hologram images (jpg/png)
            device: CUDA device string
            max_frames: Number of frames to pre-load into GPU memory
            mode: 'sync' (unlimited), 'fixed' (rate-limited), 'threaded' (bg thread)
            target_fps: Target FPS for fixed/threaded modes (0 = unlimited)
            num_images: Total images to process (for ring-buffer sizing)
            img_size: Resize target (default 448)
        """
        self.device = torch.device(device)
        self.mode = mode
        self.target_fps = target_fps
        self.img_size = img_size

        # Collect image paths
        data_path = Path(data_dir)
        paths = sorted(data_path.glob('*.jpg'))
        if not paths:
            paths = sorted(data_path.glob('*.png'))
        if num_images is not None:
            paths = paths[:num_images]

        # Pre-load into GPU
        n_load = min(max_frames, len(paths))
        print(f"FrameSource: pre-loading {n_load} frames to GPU "
              f"({n_load * img_size * img_size * 4 / 1e6:.0f} MB)...")

        self._frames = []
        for i, p in enumerate(paths[:n_load]):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(img.astype(np.float32) / 255.0).to(self.device)
            self._frames.append(t)

        self._n_frames = len(self._frames)
        self._idx = 0
        self._frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        self._last_yield_time = 0.0

        # Threaded mode state
        self._thread = None
        self._queue: deque = deque(maxlen=64)
        self._stop_event = threading.Event()
        self._queue_depths = []

        if mode == 'threaded' and target_fps > 0:
            self._start_thread()

        print(f"FrameSource: {self._n_frames} frames loaded, mode={mode}, "
              f"target_fps={target_fps}")

    def _start_thread(self):
        """Start background producer thread for threaded mode."""
        def _producer():
            interval = self._frame_interval
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                frame = self._frames[self._idx % self._n_frames]
                self._idx += 1
                self._queue.append(frame)
                self._queue_depths.append(len(self._queue))
                elapsed = time.perf_counter() - t0
                if interval > elapsed:
                    time.sleep(interval - elapsed)

        self._thread = threading.Thread(target=_producer, daemon=True)
        self._thread.start()

    def next_frame(self) -> torch.Tensor:
        """Get next frame from the source.

        Returns:
            float32 tensor on GPU, shape (img_size, img_size), values in [0, 1]
        """
        if self.mode == 'threaded':
            # Wait for frame from producer thread
            while len(self._queue) == 0:
                time.sleep(0.0001)
            return self._queue.popleft()

        # Rate limiting for fixed mode
        if self.mode == 'fixed' and self._frame_interval > 0:
            now = time.perf_counter()
            wait = self._frame_interval - (now - self._last_yield_time)
            if wait > 0:
                time.sleep(wait)
            self._last_yield_time = time.perf_counter()

        # Ring-buffer cycling
        frame = self._frames[self._idx % self._n_frames]
        self._idx += 1
        return frame

    def stop(self):
        """Stop background thread (threaded mode only)."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)

    def stats(self) -> dict:
        """Return frame source statistics."""
        result = {
            'mode': self.mode,
            'target_fps': self.target_fps,
            'frames_loaded': self._n_frames,
            'frames_served': self._idx,
            'gpu_memory_mb': self._n_frames * self.img_size * self.img_size * 4 / 1e6,
        }
        if self._queue_depths:
            depths = np.array(self._queue_depths)
            result['queue_depth_max'] = int(np.max(depths))
            result['queue_depth_mean'] = float(np.mean(depths))
        return result

    def __del__(self):
        self.stop()
