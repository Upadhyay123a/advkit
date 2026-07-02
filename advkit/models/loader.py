from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
from torchvision import models


def load_model(name: str = "resnet18") -> nn.Module:
    """Load a pretrained torchvision model and place it in evaluation mode."""
    if name.lower() != "resnet18":
        raise ValueError("Only resnet18 is supported in this starter project.")

    # Pretrained weights give the network a useful feature basis before we attack it.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()  # Disable dropout and use deterministic inference for prediction.
    return model


def predict(model: nn.Module, image_tensor: Tensor) -> Tuple[int, float]:
    """Return the predicted class index and confidence for a single image tensor."""
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        # The logits are raw scores for each class before softmax turns them into probabilities.
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, predicted_class_idx = torch.max(probabilities, dim=0)
    return int(predicted_class_idx.item()), float(confidence.item())
