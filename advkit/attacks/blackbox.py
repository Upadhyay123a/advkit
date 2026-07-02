from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from .base import Attack


class SimBA(Attack):
    """Simple Black-box Attack (SimBA) — gradient-free adversarial perturbation.

    This attack queries the model's output probabilities without accessing
    internal gradients, simulating a realistic API-only threat model where
    the attacker has no visibility into model internals.
    """

    def __init__(self, epsilon: float = 0.02, max_queries: int = 5000, early_stop: bool = True) -> None:
        self.epsilon = epsilon
        self.max_queries = max_queries
        self.early_stop = early_stop

    def generate(
        self, model: nn.Module, image_tensor: Tensor, true_label: int, **kwargs
    ) -> Tuple[Tensor, int]:
        """Craft an adversarial image via query-based random search.

        Returns:
            (adversarial_tensor, query_count) — the attacked image and number of model queries used.
        """
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)

        original = image_tensor.clone().detach()
        current_image = original.clone()
        query_count = 0

        # The first model call counts as a query and establishes the starting confidence.
        with torch.no_grad():
            logits = model(current_image.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            current_confidence = probs[true_label].item()
        query_count += 1

        # Flatten the image into a list of pixel coordinates and shuffle them.
        # This defines the random search order for the black-box attack.
        C, H, W = current_image.shape
        pixel_coords = [(c, h, w) for c in range(C) for h in range(H) for w in range(W)]
        indices = torch.randperm(len(pixel_coords))
        pixel_coords = [pixel_coords[i] for i in indices]

        for coord in pixel_coords:
            if query_count >= self.max_queries:
                break

            c, h, w = coord
            original_value = original[c, h, w].item()

            # Try the PLUS direction first. This is a simple, query-efficient test.
            current_image[c, h, w] = original_value + self.epsilon
            current_image = torch.clamp(current_image, original - self.epsilon, original + self.epsilon)

            with torch.no_grad():
                logits = model(current_image.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)[0]
                new_confidence = probs[true_label].item()
            query_count += 1

            if new_confidence < current_confidence:
                current_confidence = new_confidence
            else:
                # If PLUS did not help, try the opposite direction.
                current_image[c, h, w] = original_value - self.epsilon
                current_image = torch.clamp(current_image, original - self.epsilon, original + self.epsilon)

                with torch.no_grad():
                    logits = model(current_image.unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)[0]
                    new_confidence = probs[true_label].item()
                query_count += 1

                if new_confidence < current_confidence:
                    current_confidence = new_confidence
                else:
                    # Neither direction improved the confidence, so revert to the original pixel value.
                    current_image[c, h, w] = original_value

            if self.early_stop:
                with torch.no_grad():
                    logits = model(current_image.unsqueeze(0))
                    predicted_class = torch.argmax(logits, dim=1)[0].item()
                    if predicted_class != true_label:
                        return current_image.detach(), query_count

        return current_image.detach(), query_count
