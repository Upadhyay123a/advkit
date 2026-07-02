from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import Attack


class PGD(Attack):
    """Projected Gradient Descent attack for image classifiers."""

    def __init__(
        self,
        epsilon: float = 0.03,
        alpha: float = 0.005,
        num_steps: int = 10,
        random_start: bool = True,
        early_stop: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        self.early_stop = early_stop

    def generate(self, model: nn.Module, image_tensor: Tensor, true_label: int, **kwargs) -> Tensor:
        """Craft an adversarial image via iterative gradient-based perturbation."""

        original = image_tensor.clone().detach()
        adversarial = original.clone().detach()

        # Random start within epsilon ball
        if self.random_start:
            noise = torch.empty_like(adversarial).uniform_(-self.epsilon, self.epsilon)
            adversarial = adversarial + noise

        best_adversarial = adversarial.clone()
        best_confidence = float("inf")

        for step in range(self.num_steps):
            adversarial.requires_grad_(True)
            model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                logits = model(adversarial)
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    torch.tensor([true_label], dtype=torch.long, device=image_tensor.device),
                )
                loss.backward()
                gradient = adversarial.grad.detach()

            # Gradient step
            adversarial = adversarial.detach() + self.alpha * gradient.sign()

            # ---- PGD projection (epsilon constraint) ----
            adversarial = torch.max(
                torch.min(adversarial, original + self.epsilon),
                original - self.epsilon,
            )

            # ---- ImageNet normalization bounds ----
            mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)

            min_val = (0.0 - mean) / std
            max_val = (1.0 - mean) / std

            adversarial = torch.max(
                torch.min(adversarial, max_val),
                min_val,
            )

            # Evaluate current adversarial
            with torch.no_grad():
                logits_best = model(adversarial)
                probs_best = torch.softmax(logits_best, dim=1)[0]
                confidence_true = probs_best[true_label].item()

                if confidence_true < best_confidence:
                    best_confidence = confidence_true
                    best_adversarial = adversarial.clone()

            # Early stopping
            if self.early_stop:
                with torch.no_grad():
                    predicted_class = torch.argmax(logits, dim=1)[0].item()
                    if predicted_class != true_label:
                        return best_adversarial.detach()

        return best_adversarial.detach()