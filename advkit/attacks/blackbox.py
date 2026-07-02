from __future__ import annotations

import logging
from typing import Tuple

import torch
from torch import Tensor, nn

from .base import Attack

logger = logging.getLogger(__name__)


class SimBA(Attack):
    """Simple Black-box Attack (SimBA) — gradient-free adversarial perturbation.

    This attack queries the model's output probabilities without accessing
    internal gradients, simulating a realistic API-only threat model where
    the attacker has no visibility into model internals.

    Optimization: Plus and minus perturbation candidates are batched into a
    single forward pass per pixel, reducing wall-clock latency by ~50% while
    maintaining accurate query count (still increments by 2 per pixel, since
    2 distinct images are evaluated).
    """

    def __init__(self, epsilon: float = 0.02, max_queries: int = 5000, early_stop: bool = True) -> None:
        self.epsilon = epsilon
        self.max_queries = max_queries
        self.early_stop = early_stop

    def generate(
        self, model: nn.Module, image_tensor: Tensor, true_label: int, **kwargs
    ) -> Tuple[Tensor, int]:
        """Craft an adversarial image via query-based random search with batched perturbations.

        Batches plus and minus perturbation candidates for each pixel into a single
        forward pass, reducing model calls from 2 per pixel to 1 while maintaining
        accurate query accounting (still counts as 2 queries per pixel).

        Returns:
            (adversarial_tensor, query_count) — the attacked image and number of model queries used.
        """
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)

        original = image_tensor.clone().detach()
        current_image = original.clone()
        query_count = 0
        stop_reason = "budget_exhausted"

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

        for pixel_idx, coord in enumerate(pixel_coords):
            if query_count >= self.max_queries:
                break

            c, h, w = coord
            original_value = original[c, h, w].item()

            # Construct plus and minus candidates and batch them into a single forward pass.
            plus_candidate = current_image.clone()
            plus_candidate[c, h, w] = original_value + self.epsilon
            plus_candidate = torch.clamp(plus_candidate, original - self.epsilon, original + self.epsilon)

            minus_candidate = current_image.clone()
            minus_candidate[c, h, w] = original_value - self.epsilon
            minus_candidate = torch.clamp(minus_candidate, original - self.epsilon, original + self.epsilon)

            # Stack into [2, C, H, W] and run ONE forward pass instead of two separate ones.
            batch = torch.stack([plus_candidate, minus_candidate])
            with torch.no_grad():
                batch_logits = model(batch)
                batch_probs = torch.softmax(batch_logits, dim=1)
                plus_confidence = batch_probs[0, true_label].item()
                minus_confidence = batch_probs[1, true_label].item()
            query_count += 2

            # Decide which candidate to keep (if any improved).
            best_candidate = current_image
            best_confidence = current_confidence
            if plus_confidence < current_confidence:
                best_candidate = plus_candidate
                best_confidence = plus_confidence
            if minus_confidence < best_confidence:
                best_candidate = minus_candidate
                best_confidence = minus_confidence

            # Update if either candidate improved.
            if best_confidence < current_confidence:
                current_image = best_candidate
                current_confidence = best_confidence

            # Progress logging every 100 queries.
            if query_count % 100 == 0:
                logger.info("Queries: %d, Confidence: %.4f", query_count, current_confidence)

            # Early stop: check if either candidate flipped the predicted class.
            if self.early_stop:
                plus_pred = torch.argmax(batch_logits[0]).item()
                minus_pred = torch.argmax(batch_logits[1]).item()
                if plus_pred != true_label or minus_pred != true_label:
                    # Use whichever one flipped the class (prefer the one with lower confidence).
                    if plus_pred != true_label and (minus_pred == true_label or plus_confidence <= minus_confidence):
                        current_image = plus_candidate
                    else:
                        current_image = minus_candidate
                    stop_reason = "early_stop"
                    return current_image.detach(), query_count

        logger.info("Attack finished — stop reason: %s (queries: %d, final confidence: %.4f)", 
                    stop_reason, query_count, current_confidence)
        return current_image.detach(), query_count
