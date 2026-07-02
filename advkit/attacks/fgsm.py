from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import Attack


class FGSM(Attack):
    """Fast Gradient Sign Method attack for image classifiers."""

    def __init__(self, epsilon: float = 0.03) -> None:
        self.epsilon = epsilon

    def generate(self, model: nn.Module, image_tensor: Tensor, true_label: int, **kwargs) -> Tensor:
        """Craft an adversarial image by moving in the direction of the loss gradient."""
        image = image_tensor.clone().detach().requires_grad_(True)

        # We need a differentiable input so the model can tell us which pixels matter most.
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(image)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([true_label], dtype=torch.long))

            # Backpropagation gives the gradient of the loss with respect to each pixel.
            loss.backward()
            gradient = image.grad.detach()

        # The sign of the gradient is used because it gives the direction that increases loss
        # the most per pixel while keeping the update simple and bounded.
        adversarial = image + self.epsilon * gradient.sign()

        # Keep the result inside the valid normalized-image range so it still looks like an image.
        adversarial = torch.clamp(adversarial, 0.0, 1.0)
        return adversarial.detach()
