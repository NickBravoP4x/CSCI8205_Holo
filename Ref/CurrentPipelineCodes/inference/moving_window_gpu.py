import torch


class MovingWindowEnhancer:
    """
    GPU-accelerated background subtraction enhancer.

    Supports two methods:
    - 'ema': Exponential Moving Average (FAST ~2-3ms)
    - 'median': Median background (SLOW ~8-12ms but more robust)
    """

    def __init__(self, window_size=10, scale_factor=2, mean_intensity=105,
                 method='ema', ema_alpha=0.05):
        """
        Args:
            window_size: Number of frames for median method (ignored for EMA)
            scale_factor: Contrast scaling factor
            mean_intensity: Target background intensity
            method: 'ema' (fast) or 'median' (robust)
            ema_alpha: EMA smoothing factor (0.01-0.1, lower = smoother)
        """
        self.window_size = window_size
        self.scale_factor = scale_factor
        self.mean_intensity = mean_intensity
        self.method = method
        self.ema_alpha = ema_alpha

        # EMA state
        self.ema_background = None
        self.ema_initialized = False
        self.ema_warmup_frames = 0
        self.ema_warmup_target = 5  # Frames before EMA is considered stable

        # Median state
        self.stack = None
        self.current_idx = 0
        self.is_filled = False

    def enhance(self, image, return_float=False):
        """
        Enhance frame using background subtraction.

        Args:
            image: Tensor in [0, 255] range
            return_float: If True, return float32 [0,1] instead of uint8

        Returns:
            Enhanced tensor
        """
        if self.method == 'ema':
            return self._enhance_ema(image, return_float)
        else:
            return self._enhance_median(image, return_float)

    def _enhance_ema(self, image, return_float=False):
        """Fast EMA-based enhancement (~2-3ms)."""
        # Convert to float32 if needed
        if image.dtype != torch.float32:
            image = image.to(dtype=torch.float32)

        # Initialize EMA background on first frame
        if self.ema_background is None:
            self.ema_background = image.clone()
            self.ema_initialized = True
            self.ema_warmup_frames = 1
            # Return zeros during initial warmup
            if return_float:
                return torch.zeros(image.shape, device=image.device, dtype=torch.float32)
            return torch.zeros(image.shape, device=image.device, dtype=torch.uint8)

        # Update EMA background: bg = alpha * new + (1 - alpha) * bg
        self.ema_background = (self.ema_alpha * image +
                               (1 - self.ema_alpha) * self.ema_background)

        self.ema_warmup_frames += 1

        # Return zeros during warmup period
        if self.ema_warmup_frames < self.ema_warmup_target:
            if return_float:
                return torch.zeros(image.shape, device=image.device, dtype=torch.float32)
            return torch.zeros(image.shape, device=image.device, dtype=torch.uint8)

        # Compute enhancement
        enhanced = torch.abs(
            (image - self.ema_background) * self.scale_factor + self.mean_intensity
        )
        enhanced = torch.clamp(enhanced, 0, 255)

        if return_float:
            return (enhanced / 255.0).to(dtype=torch.float32)
        return enhanced.to(dtype=torch.uint8)

    def _enhance_median(self, image, return_float=False):
        """Median-based enhancement (robust but slower ~8-12ms)."""
        # Convert to float32 if needed
        if image.dtype != torch.float32:
            image = image.to(dtype=torch.float32)

        # Initialize stack on first frame
        if self.stack is None:
            shape = image.shape
            self.stack = torch.zeros(
                (self.window_size, *shape),
                dtype=torch.float32,
                device=image.device
            )

        # Add to circular buffer
        self.stack[self.current_idx] = image
        self.current_idx += 1

        # Check if buffer is filled
        if self.current_idx >= self.window_size:
            self.is_filled = True
            self.current_idx = 0

        # Only compute enhancement once buffer is full
        if not self.is_filled:
            if return_float:
                return torch.zeros(image.shape, device=image.device, dtype=torch.float32)
            return torch.zeros(image.shape, device=image.device, dtype=torch.uint8)

        # Compute median using kthvalue (faster than torch.median)
        k = self.window_size // 2 + 1
        median_background = torch.kthvalue(self.stack, k, dim=0).values

        # Get the most recent image (previous index in circular buffer)
        recent_idx = (self.current_idx - 1) % self.window_size
        recent_image = self.stack[recent_idx]

        # Compute enhancement
        enhanced = torch.abs(
            (recent_image - median_background) * self.scale_factor + self.mean_intensity
        )
        enhanced = torch.clamp(enhanced, 0, 255)

        if return_float:
            return (enhanced / 255.0).to(dtype=torch.float32)
        return enhanced.to(dtype=torch.uint8)
