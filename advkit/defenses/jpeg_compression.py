"""JPEG compression defense.

Lossy JPEG compression discards high-frequency detail. Adversarial
perturbations are typically small, high-frequency signals spread across
many pixels — compression tends to smooth these out while preserving
the low-frequency structure a human recognizes as "the image."
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .base import Defense

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class JPEGCompression(Defense):
    """Applies lossy JPEG compression as an adversarial defense."""

    def __init__(self, quality: int = 75) -> None:
        if not (1 <= quality <= 100):
            raise ValueError("quality must be between 1 and 100")
        self.quality = quality

    def apply(self, image_tensor: Tensor) -> Tensor:
        squeeze = image_tensor.ndim == 4
        if squeeze:
            image_tensor = image_tensor.squeeze(0)

        device = image_tensor.device
        mean = torch.tensor(MEAN, device=device).view(3, 1, 1)
        std = torch.tensor(STD, device=device).view(3, 1, 1)

        # Convert to a real image so we can actually JPEG-encode it.
        pixel_space = torch.clamp(image_tensor * std + mean, 0.0, 1.0)
        array = (pixel_space.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        pil_image = Image.fromarray(array)

        # Compress and immediately decompress in memory — no disk I/O needed.
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        # Convert back into the model's normalized tensor space.
        compressed_array = np.array(compressed, dtype=np.float32) / 255.0
        compressed_tensor = torch.from_numpy(compressed_array).permute(2, 0, 1).to(device)
        result = (compressed_tensor - mean) / std

        return result.unsqueeze(0) if squeeze else result
