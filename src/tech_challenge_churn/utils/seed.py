"""Rotinas de reprodutibilidade."""

from __future__ import annotations

import os
import random

import numpy as np

from tech_challenge_churn.config import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fixa as principais seeds usadas no projeto."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
