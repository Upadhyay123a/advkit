"""Bit-depth reduction defense (a.k.a. feature squeezing).

Reduces color precision per channel. Small adversarial perturbations often
rely on fine-grained pixel values to fool the model; quantizing pixel values
to fewer bits can destroy that signal while leaving visible image content
largely intact. This is a real technique from adversarial ML research
(Xu et al., "Feature Squeezing", 2017).
"""
from __future__ import annotations

import torch
from torch import Tensor

from .base import Defense

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class BitDepthReduction(Defense):
    """Quantizes normalized image tensors to a reduced bit depth."""

    def __init__(self, bits: int = 4) -> None:
        if not (1 <= bits <= 8):
            raise ValueError("bits must be between 1 and 8")
        self.bits = bits

    def apply(self, image_tensor: Tensor) -> Tensor:
        squeeze = image_tensor.ndim == 4
        if squeeze:
            image_tensor = image_tensor.squeeze(0)

        device = image_tensor.device
        mean = torch.tensor(MEAN, device=device).view(3, 1, 1)
        std = torch.tensor(STD, device=device).view(3, 1, 1)

        # Undo normalization to get pixel values back into [0, 1] where
        # quantization makes intuitive sense (real pixel intensity space).
        pixel_space = image_tensor * std + mean
        pixel_space = torch.clamp(pixel_space, 0.0, 1.0)

        # Quantize: map [0,1] -> 2^bits discrete levels -> back to [0,1].
        levels = 2 ** self.bits
        quantized = torch.round(pixel_space * (levels - 1)) / (levels - 1)

        # Re-normalize back into the model's expected input space.
        result = (quantized - mean) / std

        return result.unsqueeze(0) if squeeze else result
