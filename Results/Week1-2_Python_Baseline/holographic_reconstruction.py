"""
Holographic reconstruction utilities for HoloVision preprocessing pipeline.
Extracted and adapted from PreviousCode/AutoFocus/tools/dataProcessing/holography.py
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class HolographicReconstructor:
    """
    Physics-based holographic reconstruction for particle detection.
    Implements Fresnel propagation for multi-plane reconstruction.
    """
    
    def __init__(
        self,
        resolution: float = 0.087,  # microns per pixel
        wavelength: float = 405 / 1.33 / 1000,  # wavelength in microns (405nm in water)
        z_start: float = 0,  # starting depth in microns
        z_step: float = 6,   # depth step in microns
        num_planes: int = 3, # number of reconstruction planes
        shift_mean: float = 105,  # background intensity shift
        shift_value: float = 105, # intensity shift value
        padding: int = 32,   # padding for reconstruction
        device: Optional[str] = None
    ):
        self.resolution = resolution
        self.wavelength = wavelength
        self.z_start = z_start
        self.z_step = z_step
        self.num_planes = num_planes
        self.shift_mean = shift_mean
        self.shift_value = shift_value
        self.padding = padding
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize reconstruction parameters
        self._initialized = False
        self.H = None  # propagation filter
        self.f2_new = None  # frequency domain coordinates
        self.z_list = None  # list of z values
        
    def _initialize_propagation(self, im_height: int, im_width: int):
        """Initialize propagation filters for given image dimensions."""
        # Check if already initialized for these dimensions
        if (hasattr(self, '_last_height') and hasattr(self, '_last_width') and 
            self._last_height == im_height and self._last_width == im_width and 
            self._initialized):
            return
            
        # Store current dimensions
        self._last_height = im_height
        self._last_width = im_width
            
        # Adjust for resolution scaling if needed
        resolution = self.resolution
        z_start = self.z_start
        z_step = self.z_step

        resolution_threshold = 0.5  # Apply scaling for resolutions finer than this (μm/pixel)
        if resolution < resolution_threshold:
            scale_factor = resolution_threshold / resolution
            resolution = scale_factor * resolution
            z_start = scale_factor * scale_factor * z_start
            z_step = scale_factor * scale_factor * z_step
            
        # Add padding
        padded_height = im_height + 2 * self.padding
        padded_width = im_width + 2 * self.padding
        
        # Create frequency domain coordinates  
        x = torch.arange(padded_width, device=self.device)
        y = torch.arange(padded_height, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Transpose like in original code
        xx = torch.t(xx)
        yy = torch.t(yy)
        
        fx = xx / padded_width - 0.5
        fy = yy / padded_height - 0.5
        f2 = fx**2 + fy**2
        self.f2_new = torch.fft.fftshift(f2)
        
        # Create propagation filter
        sqrt_input = 1 - f2 * (self.wavelength / resolution)**2
        sqrt_input = torch.clamp(sqrt_input, min=0)
        self.H = -2 * math.pi * 1j * torch.sqrt(sqrt_input) / self.wavelength
        self.H = torch.fft.fftshift(self.H)

        # Update f2_new for depth calculation
        temp = (self.wavelength / resolution) ** 2
        self.f2_new = temp * self.f2_new
        self.f2_new = torch.sqrt(1 - self.f2_new)
        
        # Create z list
        z_end = self.num_planes * z_step + z_start
        self.z_list = np.arange(z_start, z_end, z_step)
        
        self._initialized = True
        
    def reconstruct_3d_batched(self, hologram: torch.Tensor) -> torch.Tensor:
        """
        Perform multi-plane reconstruction in a single batched pass.
        
        Args:
            hologram: Input hologram tensor of shape [H, W]
            
        Returns:
            Reconstructed intensity volume of shape [H, W, num_planes]
        """
        # Ensure hologram is on correct device
        if hologram.device != self.device:
            hologram = hologram.to(self.device)
            
        # Initialize propagation filters if needed
        H_in, W_in = hologram.shape
        self._initialize_propagation(H_in, W_in)
        
        # Pad image with median values
        median_val = torch.median(hologram)
        im_new = median_val * torch.ones(
            (H_in + 2*self.padding, W_in + 2*self.padding),
            device=self.device,
            dtype=hologram.dtype
        )
        im_new[self.padding:self.padding + H_in, 
               self.padding:self.padding + W_in] = hologram
               
        # FFT of padded image
        Fholo = torch.fft.fft2(im_new.float())
        H_pad, W_pad = im_new.shape
        
        # Create batched propagation filters for all planes
        z_tensor = torch.as_tensor(self.z_list, device=self.device, dtype=Fholo.dtype)
        num_planes = z_tensor.shape[0]
        
        # Batched propagation: shape (num_planes, H_pad, W_pad)
        Hz_stack = torch.exp(
            -2j * math.pi * z_tensor[:, None, None] * self.f2_new / self.wavelength
        )

        # Expand hologram for batch processing
        Fholo_batched = Fholo.unsqueeze(0).expand(num_planes, H_pad, W_pad)

        # Multiply and inverse FFT
        Fplane = Fholo_batched * Hz_stack
        rec_batched = torch.fft.ifft2(Fplane, dim=(-2, -1))
        
        # Apply phase correction
        phase_stack = torch.exp(2j * math.pi * z_tensor / self.wavelength)
        rec_batched = rec_batched * phase_stack[:, None, None]
        
        # Permute to [H_pad, W_pad, num_planes] and remove padding
        rec_batched = rec_batched.permute(1, 2, 0)
        rec3 = rec_batched[
            self.padding:H_pad - self.padding,
            self.padding:W_pad - self.padding,
            :
        ]
        
        return rec3
        
    def reconstruct_intensity(self, hologram: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct intensity volume with background normalization.
        
        Args:
            hologram: Input hologram tensor of shape [H, W]
            
        Returns:
            Intensity volume of shape [H, W, num_planes] with values in [0, 1]
        """
        # Get complex reconstruction
        rec3 = self.reconstruct_3d_batched(hologram)
        
        # Calculate intensity as magnitude squared
        intensity = torch.abs(rec3)**2
        
        # Normalize to [0, 1]
        min_val = torch.min(intensity)
        max_val = torch.max(intensity)
        intensity = (intensity - min_val) / (max_val - min_val + 1e-8)
        
        # Apply background shift
        mean_background = torch.mean(intensity)
        scale_factor = (self.shift_mean / 255) / (mean_background + 1e-8)
        intensity *= scale_factor
        
        # Clamp to [0, 1]
        intensity = torch.clamp(intensity, 0, 1)
        
        return intensity
        
    def generate_rgb_stack(self, hologram: torch.Tensor) -> torch.Tensor:
        """
        Generate 3-channel RGB stack from holographic reconstruction.
        
        Args:
            hologram: Input hologram tensor of shape [H, W]
            
        Returns:
            RGB stack tensor of shape [3, H, W]
        """
        # Get intensity reconstruction
        intensity_3d = self.reconstruct_intensity(hologram)
        
        # Use the 3 reconstruction planes as RGB channels
        if self.num_planes >= 3:
            # Take first 3 planes
            rgb_stack = intensity_3d[:, :, :3]
        else:
            # Repeat planes if we have fewer than 3
            rgb_stack = intensity_3d.repeat(1, 1, 3//self.num_planes + 1)[:, :, :3]
            
        # Permute to [C, H, W] format
        rgb_stack = rgb_stack.permute(2, 0, 1)
        
        return rgb_stack


def generate_rgb_stack_images(batch_images: torch.Tensor, 
                            resolution: float = 0.087,
                            wavelength: float = 405 / 1.33 / 1000,
                            z_start: float = 0,
                            z_step: float = 6,
                            num_planes: int = 3,
                            shift_mean: float = 105,
                            shift_value: float = 105) -> torch.Tensor:
    """
    Generate RGB stacks for a batch of hologram images.
    
    Args:
        batch_images: Batch of hologram tensors of shape [N, 1, H, W]
        
    Returns:
        Batch of RGB stacks of shape [N, 3, H, W]
    """
    device = batch_images.device
    batch_size = batch_images.shape[0]
    
    # Get image dimensions (assuming all images same size)
    im_height = batch_images.shape[2]
    im_width = batch_images.shape[3]
    
    # Create reconstructor
    reconstructor = HolographicReconstructor(
        resolution=resolution,
        wavelength=wavelength,
        z_start=z_start,
        z_step=z_step,
        num_planes=num_planes,
        shift_mean=shift_mean,
        shift_value=shift_value,
        device=device
    )
    
    # Process each image
    batch_stack = []
    for i in range(batch_size):
        # Extract single image and remove channel dimension
        image_2d = batch_images[i].squeeze(0)  # [H, W]
        
        # Generate RGB stack
        rgb_stack = reconstructor.generate_rgb_stack(image_2d)  # [3, H, W]
        batch_stack.append(rgb_stack)
    
    # Stack into batch
    batch_stack = torch.stack(batch_stack, dim=0)  # [N, 3, H, W]
    
    return batch_stack