"""Shared seeding helper so every training script in analysis/ is reproducible.

Call set_seed() before constructing models, DataLoaders, or doing any
permutation/shuffling. Model init, DataLoader shuffling, and dropout all draw
from the global torch/numpy/random RNGs, so without this, re-running a script
can silently change which weights (and sometimes which architecture "wins" a
comparison) get produced and saved.
"""

import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
