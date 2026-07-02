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

        # Random start: perturb the initial adversarial example within the epsilon-ball.
        # This avoids starting from a poor local gradient direction and leads to stronger attacks.
        if self.random_start:
            noise = torch.empty_like(adversarial).uniform_(-self.epsilon, self.epsilon)
            adversarial = adversarial + noise
            adversarial = torch.clamp(adversarial, 0.0, 1.0)

        best_adversarial = adversarial.clone()
        best_confidence = float("inf")

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

            # Track the best adversarial image by confidence on the true label. Attacks are not
            # always monotonic — later steps may increase confidence again due to gradient noise.
            with torch.no_grad():
                logits_best = model(adversarial)
                probs_best = torch.softmax(logits_best, dim=1)[0]
                confidence_true = probs_best[true_label].item()

                if confidence_true < best_confidence:
                    best_confidence = confidence_true
                    best_adversarial = adversarial.clone()

            # Early stop: if prediction changed, stop the attack for speed.
            if self.early_stop:
                with torch.no_grad():
                    predicted_class = torch.argmax(logits, dim=1)[0].item()
                    if predicted_class != true_label:
                        return best_adversarial.detach()

        return best_adversarial.detach()
