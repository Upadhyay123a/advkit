from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

PathLike = Union[str, Path]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_image(path: PathLike) -> Tensor:
    """Load an image from disk, resize it, and return a normalized tensor."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        image_array = np.array(image, dtype=np.float32) / 255.0

    # PyTorch models expect tensors shaped as [C, H, W] with ImageNet-style normalization.
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    tensor = (tensor - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return tensor


def tensor_to_image(tensor: Tensor) -> Image.Image:
    """Convert a normalized image tensor back into a PIL image for display or saving."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError("Expected a tensor with shape [C, H, W] or [1, C, H, W].")

    # Undo normalization so the pixel values are in the display-friendly [0, 1] range.
    tensor = tensor.detach().cpu().float()
    tensor = tensor * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    image_array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(image_array)
