"""Reproducible seeds without importing ML libraries for mock-only tests."""
import random
import numpy as np


def seed_everything(seed: int, *, use_torch: bool = False) -> None:
    """Seed Python/NumPy and optionally PyTorch; no hardware determinism claim."""
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
