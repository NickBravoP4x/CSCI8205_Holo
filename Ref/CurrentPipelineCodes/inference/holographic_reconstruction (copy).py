"""
Holographic Reconstruction for Live Detection
Standalone implementation - no external dependencies
"""
import torch
import math
import numpy as np


class HolographicReconstructor:
    """
    GPU-accelerated holographic reconstruction for live particle detection.
    Generates 3-plane intensity stacks optimized for YOLO input.
    """

    def __init__(
        self,
        resolution: float,
        wavelength: float,
        z_start: float,
        z_step: float,
        num_planes: int,
        im_x: int,
        im_y: int,
        shift_mean: float = 105,
        shift_value: float = 105,
        padding: int = 32,
        use_fp16: bool = False,
    ):
        """
        Initialize holographic reconstructor.

        Args:
            resolution: Microns per pixel (e.g., 0.087 for 40x objective)
            wavelength: Wavelength in microns (e.g., 405nm / 1.33 / 1000 = 0.304)
            z_start: Starting depth in microns (typically 0)
            z_step: Depth step between planes in microns (e.g., 6)
            num_planes: Number of reconstruction planes (typically 3 for YOLO)
            im_x: Image width in pixels
            im_y: Image height in pixels
            shift_mean: Background intensity normalization target
            shift_value: Background intensity shift value
            padding: Padding for FFT (reduces edge artifacts)
            use_fp16: Use float16 for speed (requires GPU with tensor cores)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        # Force FP16 for speed if GPU supports it
        if self.device.type == "cuda" and not use_fp16:
            try:
                # Check if GPU supports FP16
                test_tensor = torch.zeros(1, dtype=torch.float16, device=self.device)
                self.use_fp16 = True
                print("  ⚡ Enabled FP16 for 2x speedup")
            except:
                pass

        dtype = torch.float16 if self.use_fp16 else torch.float32
        ctype = torch.complex32 if self.use_fp16 else torch.complex64

        # Resolution scaling for very small pixels
        if resolution < 0.5:
            scale = 0.5 / resolution
            resolution *= scale
            z_start *= scale ** 2
            z_step *= scale ** 2

        self.resolution = torch.tensor(resolution, device=self.device, dtype=dtype)
        self.wavelength = torch.tensor(wavelength, device=self.device, dtype=dtype)
        self.padding = padding
        self.shift_mean = shift_mean
        self.shift_value = shift_value

        # Depth planes
        self.z_start = z_start
        self.z_step = z_step
        self.num_planes = num_planes
        self.z_list = np.arange(z_start, z_start + num_planes * z_step, z_step)

        # Padded image dimensions
        self.im_x_padded = im_x + 2 * padding
        self.im_y_padded = im_y + 2 * padding
        self.im_x = im_x
        self.im_y = im_y

        # Precompute spatial frequency grid
        # Note: Image is (H, W) but we need grid in (Y, X) order for proper FFT
        y = torch.arange(self.im_y_padded, device=self.device, dtype=dtype)
        x = torch.arange(self.im_x_padded, device=self.device, dtype=dtype)
        yy, xx = torch.meshgrid(
            y / self.im_y_padded - 0.5,
            x / self.im_x_padded - 0.5,
            indexing="ij"
        )
        f2 = torch.fft.fftshift(xx ** 2 + yy ** 2)

        # Precompute propagation kernel components
        sqrt_input = 1 - f2 * (self.wavelength / self.resolution) ** 2
        sqrt_input = torch.clamp(sqrt_input, min=0)
        self.f2_new = torch.sqrt(sqrt_input)

        # Transfer function
        self.H = -2 * math.pi * 1j * self.f2_new / self.wavelength
        self.H = torch.fft.fftshift(self.H).to(dtype=ctype)

    def reconstruct_3d(self, hologram_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct hologram at multiple depth planes.

        Args:
            hologram_tensor: Input hologram (H, W) on GPU, float32, range [0, 1]

        Returns:
            3D complex field (H, W, num_planes)
        """
        # Validate input dimensions match initialized size
        actual_h, actual_w = hologram_tensor.shape
        if actual_h != self.im_y or actual_w != self.im_x:
            raise ValueError(
                f"Input tensor size ({actual_h}, {actual_w}) doesn't match "
                f"initialized size ({self.im_y}, {self.im_x}). "
                f"Reconstructor must be reinitialized for new image size."
            )

        # Pad hologram
        im_padded = torch.full(
            (self.im_y_padded, self.im_x_padded),
            torch.median(hologram_tensor).item(),
            device=hologram_tensor.device,
            dtype=hologram_tensor.dtype
        )
        im_padded[
            self.padding : self.padding + self.im_y,
            self.padding : self.padding + self.im_x
        ] = hologram_tensor

        # FFT of hologram
        Fholo = torch.fft.fft2(im_padded)

        # Reconstruct each plane
        reconstructed_planes = []
        for z in self.z_list:
            # Propagation kernel for this depth
            Hz = torch.exp(-2 * math.pi * 1j * z * self.f2_new / self.wavelength).to(
                hologram_tensor.device
            )
            phase_shift = torch.exp(2 * math.pi * 1j * z / self.wavelength).to(
                hologram_tensor.device
            )

            # Propagate and inverse FFT
            Fplane = torch.multiply(Fholo, Hz)
            rec_plane = torch.fft.ifft2(Fplane)
            rec_plane = rec_plane * phase_shift

            reconstructed_planes.append(rec_plane.unsqueeze(-1))

        # Stack planes
        rec3d = torch.cat(reconstructed_planes, dim=-1)

        # Remove padding
        rec3d = rec3d[
            self.padding : self.im_y_padded - self.padding,
            self.padding : self.im_x_padded - self.padding,
            :
        ]

        return rec3d

    def reconstruct_intensity_stack(self, hologram_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct hologram and return normalized intensity stack for YOLO.

        Args:
            hologram_tensor: Input hologram (H, W) on GPU, float32

        Returns:
            Normalized intensity stack (H, W, num_planes), range [0, 1]
        """
        # Complex field reconstruction
        rec3d = self.reconstruct_3d(hologram_tensor)

        # Intensity = |field|^2
        intensity = rec3d.abs().pow(2)

        # Normalize to [0, 1]
        min_val = intensity.min()
        intensity = intensity - min_val
        max_val = intensity.max()
        intensity = intensity / (max_val + 1e-8)

        return intensity
