"""Adversarial attack implementations."""

from .base import Attack
from .fgsm import FGSM

__all__ = ["Attack", "FGSM"]
