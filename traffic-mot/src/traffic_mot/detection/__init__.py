"""Detector interfaces; import the real adapter explicitly to load ML libraries."""

from .base import BaseDetector
from .mock import MockDetector

__all__ = ["BaseDetector", "MockDetector"]
