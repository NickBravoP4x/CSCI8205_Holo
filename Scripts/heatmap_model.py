"""
Heatmap detector: MobileNetV3-Small backbone + FPN head.

Architecture:
    - Backbone: torchvision MobileNetV3-Small with conv-bn fusion applied
    - FPN: 4 lateral 1x1 convs tapping features at strides 2/8/16/32,
      bottom-up upsample+add, then head Conv(32,32,3,stride=2)+ReLU+Conv(32,1,1)+Sigmoid
    - Input: (B, 3, 448, 448) float32
    - Output: (B, 1, 112, 112) float32 heatmap in [0, 1]

Usage:
    model = load_heatmap_detector("path/to/fast_bacteria_hm_optimized.pt", "cuda")
    heatmap = model(input_tensor)  # (B, 1, 112, 112)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FPNHead(nn.Module):
    """Feature Pyramid Network head for heatmap prediction.

    Taps MobileNetV3-Small backbone at 4 scales and fuses bottom-up.

    Tap points (for 448x448 input):
        features.0  -> 16ch @ 224x224 (stride 2)
        features.2  -> 24ch @ 56x56  (stride 8)
        features.7  -> 48ch @ 28x28  (stride 16)
        features.12 -> 576ch @ 14x14 (stride 32)
    """

    def __init__(self, fpn_channels: int = 32):
        super().__init__()
        # Lateral 1x1 convolutions to project to fpn_channels
        self.lateral0 = nn.Conv2d(16, fpn_channels, 1)
        self.lateral1 = nn.Conv2d(24, fpn_channels, 1)
        self.lateral2 = nn.Conv2d(48, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(576, fpn_channels, 1)

        # Detection head: stride-2 conv reduces 224->112, then 1x1 to single channel
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat0, feat2, feat7, feat12):
        # Bottom-up FPN fusion
        p3 = self.lateral3(feat12)                                        # 14x14
        p2 = self.lateral2(feat7) + F.interpolate(p3, size=feat7.shape[-2:], mode='nearest')   # 28x28
        p1 = self.lateral1(feat2) + F.interpolate(p2, size=feat2.shape[-2:], mode='nearest')   # 56x56
        p0 = self.lateral0(feat0) + F.interpolate(p1, size=feat0.shape[-2:], mode='nearest')   # 224x224

        return self.head(p0)  # 112x112


class HeatmapDetector(nn.Module):
    """MobileNetV3-Small backbone + FPN heatmap head."""

    TAP_INDICES = (0, 2, 7, 12)

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        # Only keep the feature extraction layers; discard avgpool + classifier
        self.backbone = backbone.features
        self.head = FPNHead()

    def forward(self, x):
        # Unrolled backbone loop (required for torch.jit.trace / TensorRT export).
        # MobileNetV3-Small has 13 blocks (indices 0-12).
        # Features are tapped at TAP_INDICES = (0, 2, 7, 12).
        out = self.backbone[0](x)
        feat0 = out
        out = self.backbone[1](out)
        out = self.backbone[2](out)
        feat2 = out
        out = self.backbone[3](out)
        out = self.backbone[4](out)
        out = self.backbone[5](out)
        out = self.backbone[6](out)
        out = self.backbone[7](out)
        feat7 = out
        out = self.backbone[8](out)
        out = self.backbone[9](out)
        out = self.backbone[10](out)
        out = self.backbone[11](out)
        out = self.backbone[12](out)
        feat12 = out
        return self.head(feat0, feat2, feat7, feat12)


def _fuse_conv_bn_in_module(module: nn.Module):
    """Recursively fuse Conv2d + BatchNorm2d pairs in nn.Sequential containers."""
    for name, child in module.named_children():
        if isinstance(child, nn.Sequential):
            children = list(child.children())
            for i in range(len(children) - 1):
                if isinstance(children[i], nn.Conv2d) and isinstance(children[i + 1], nn.BatchNorm2d):
                    fused = torch.nn.utils.fusion.fuse_conv_bn_eval(children[i], children[i + 1])
                    child[i] = fused
                    child[i + 1] = nn.Identity()
        _fuse_conv_bn_in_module(child)


def load_heatmap_detector(path: str, device: str = 'cuda') -> HeatmapDetector:
    """Load a heatmap detector with conv-bn fused weights.

    Args:
        path: Path to .pt checkpoint (expects keys: model_state_dict, optimizations)
        device: Target device

    Returns:
        HeatmapDetector in eval mode on the specified device
    """
    model = HeatmapDetector()
    model.eval()

    # Fuse Conv+BN pairs to match the saved state dict (which has fusion applied)
    _fuse_conv_bn_in_module(model)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Remap keys: saved dict uses "backbone.features.X" but our model uses "backbone.X"
    remapped = {}
    for k, v in state_dict.items():
        new_key = k.replace('backbone.features.', 'backbone.')
        remapped[new_key] = v
    model.load_state_dict(remapped)
    model.to(device)
    return model


if __name__ == '__main__':
    import os
    import sys

    # Find model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, 'Ref', 'CurrentPipelineCodes', 'inference',
                              'fast_bacteria_hm_optimized.pt')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading heatmap detector on {device}...")
    model = load_heatmap_detector(model_path, device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Test forward pass
    dummy = torch.randn(1, 3, 448, 448, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input shape:  {list(dummy.shape)}")
    print(f"Output shape: {list(out.shape)}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    assert out.shape == (1, 1, 112, 112), f"Expected (1,1,112,112), got {out.shape}"
    print("PASS: output shape is correct")
