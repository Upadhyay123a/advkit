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

    def __init__(self, epsilon: float = 0.05, max_queries: int = 1000, early_stop: bool = True) -> None:
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
        # Remove batch dimension if present (image_tensor is typically [1, C, H, W])
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
        
        original = image_tensor.clone().detach()
        current_image = original.clone()
        query_count = 0

        # Get initial confidence on the true label (query #1).
        with torch.no_grad():
            logits = model(current_image.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
            current_confidence = probs[true_label].item()
        query_count += 1

        # Flatten the image into a list of (C, H, W) coordinates and shuffle them.
        # This random ordering ensures we sample different pixel directions per attack run.
        C, H, W = current_image.shape
        pixel_coords = [(c, h, w) for c in range(C) for h in range(H) for w in range(W)]
        torch.manual_seed(torch.seed())  # Different seed per run
        indices = torch.randperm(len(pixel_coords))
        pixel_coords = [pixel_coords[i] for i in indices]

        # Loop through shuffled pixels, trying to decrease true-label confidence.
        # We test both +epsilon and -epsilon directions per pixel (the "simple" part of SimBA).
        for coord in pixel_coords:
            if query_count >= self.max_queries:
                break

            c, h, w = coord
            original_value = current_image[c, h, w].item()

            # Try PLUS direction: add epsilon.
            current_image[c, h, w] = original_value + self.epsilon
            current_image = torch.clamp(current_image, 0.0, 1.0)

            with torch.no_grad():
                logits = model(current_image.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)[0]
                new_confidence = probs[true_label].item()
            query_count += 1

            if new_confidence < current_confidence:
                # Confidence decreased — keep the change.
                current_confidence = new_confidence
            else:
                # PLUS didn't help; try MINUS direction: subtract 2*epsilon to go the other way.
                current_image[c, h, w] = original_value - self.epsilon
                current_image = torch.clamp(current_image, 0.0, 1.0)

                with torch.no_grad():
                    logits = model(current_image.unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)[0]
                    new_confidence = probs[true_label].item()
                query_count += 1

                if new_confidence < current_confidence:
                    # Confidence decreased — keep the change.
                    current_confidence = new_confidence
                else:
                    # Neither direction helped; revert to original pixel value.
                    current_image[c, h, w] = original_value

            # Early stop: check if prediction changed.
            if self.early_stop:
                with torch.no_grad():
                    logits = model(current_image.unsqueeze(0))
                    predicted_class = torch.argmax(logits, dim=1)[0].item()
                    if predicted_class != true_label:
                        return current_image.detach(), query_count

        return current_image.detach(), query_count
