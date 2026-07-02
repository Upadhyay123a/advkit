from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import Attack


class PGD(Attack):
    """Projected Gradient Descent attack for image classifiers."""

    def __init__(self, epsilon: float = 0.03, alpha: float = 0.005, num_steps: int = 10) -> None:
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def generate(self, model: nn.Module, image_tensor: Tensor, true_label: int, **kwargs) -> Tensor:
        """Craft an adversarial image via iterative gradient-based perturbation."""
        original = image_tensor.clone().detach()
        adversarial = original.clone().detach()

        for step in range(self.num_steps):
            adversarial.requires_grad_(True)
            model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                logits = model(adversarial)
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([true_label], dtype=torch.long))
                loss.backward()
                gradient = adversarial.grad.detach()

            # Take a small step in the gradient direction (alpha controls step size).
            adversarial = adversarial.detach() + self.alpha * gradient.sign()

            # Project back into the epsilon-ball around the original image. This keeps the total
            # perturbation bounded and is what makes PGD "projected" — we iteratively optimize
            # while staying constrained, unlike FGSM's single unconstrained step.
            adversarial = torch.clamp(adversarial, original - self.epsilon, original + self.epsilon)

            # Clamp to valid normalized-image range so pixel values stay in [0, 1].
            adversarial = torch.clamp(adversarial, 0.0, 1.0)

        return adversarial.detach()
