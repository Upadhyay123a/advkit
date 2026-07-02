from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Attack(ABC):
    """Abstract interface for adversarial attacks."""

    @abstractmethod
    def generate(self, model, image_tensor: Tensor, true_label: int, **kwargs) -> Tensor:
        """Create an adversarial example from a clean input tensor."""
        raise NotImplementedError
