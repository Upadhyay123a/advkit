from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from advkit.attacks.base import Attack
from advkit.models.loader import predict


def evaluate_attack(
    model: nn.Module,
    attack: Attack,
    image_tensors: list[Tensor],
    true_labels: list[int],
) -> Dict[str, float | int]:
    """Evaluate an attack on a batch of images.
    
    Args:
        model: The target model.
        attack: The attack instance with a generate() method.
        image_tensors: List of image tensors to attack.
        true_labels: List of true predicted labels (used as targets for the attack).
    
    Returns:
        A dictionary with:
        - 'total': number of images tested
        - 'fooled': number of successfully attacked images
        - 'success_rate': percentage of attacks that succeeded
        - 'avg_confidence_drop': average (clean_confidence - adversarial_confidence)
    """
    total = len(image_tensors)
    fooled = 0
    total_confidence_drop = 0.0

    for image_tensor, true_label in zip(image_tensors, true_labels):
        image_tensor = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor

        with torch.no_grad():
            clean_label, clean_confidence = predict(model, image_tensor.squeeze(0))

        adversarial_tensor = attack.generate(model, image_tensor, true_label)

        with torch.no_grad():
            adversarial_label, adversarial_confidence = predict(model, adversarial_tensor.squeeze(0))

        if adversarial_label != true_label:
            fooled += 1

        total_confidence_drop += clean_confidence - adversarial_confidence

    success_rate = (fooled / total * 100) if total > 0 else 0.0
    avg_confidence_drop = (total_confidence_drop / total) if total > 0 else 0.0

    return {
        "total": total,
        "fooled": fooled,
        "success_rate": success_rate,
        "avg_confidence_drop": avg_confidence_drop,
    }
