from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Attack(ABC):
    """Abstract interface for adversarial attacks.
    
    Note: Most gradient-based attacks return just the adversarial tensor.
    Black-box attacks (e.g. SimBA) may return (tensor, query_count) tuple
    to track the real-world cost of the attack.
    """

    @abstractmethod
    def generate(self, model, image_tensor: Tensor, true_label: int, **kwargs):
        """Create an adversarial example from a clean input tensor.
        
        Returns:
            Tensor or (Tensor, int): Adversarial tensor, or (adversarial tensor, query count).
        """
        raise NotImplementedError
