"""Abstract interface for input-transformation defenses.

Defenses sit between the raw input and the model: they transform an image
BEFORE it's classified, aiming to disrupt adversarial perturbations without
requiring the model itself to be retrained. This is cheaper than adversarial
training but generally weaker — a known tradeoff worth documenting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Defense(ABC):
    """Abstract interface for input-transformation defenses."""

    @abstractmethod
    def apply(self, image_tensor: Tensor) -> Tensor:
        """Transform an input tensor to reduce adversarial perturbations.

        Args:
            image_tensor: Normalized image tensor, shape [C, H, W] or [1, C, H, W].

        Returns:
            Transformed tensor of the same shape.
        """
        raise NotImplementedError
