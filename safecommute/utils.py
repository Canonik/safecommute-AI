"""Shared utilities for SafeCommute AI."""

import hashlib
import random
import numpy as np
import torch

from safecommute.constants import SEED


def seed_everything(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_split(filename):
    """
    Deterministic 70/15/15 train/val/test split based on sha256 hash.

    This is the SINGLE source of truth for hash-based splitting.
    All pipeline scripts must import and use this function.
    """
    h = int(hashlib.sha256(filename.encode()).hexdigest(), 16) % 100
    if h < 70:
        return 'train'
    elif h < 85:
        return 'val'
    else:
        return 'test'


def worker_init_fn(worker_id):
    """
    DataLoader worker init: ensure each worker gets a unique random seed.

    Without this, all workers start with the same seed and produce
    identical augmentation for different samples in the same batch.
    """
    seed = torch.initial_seed() % (2**32) + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32))
