"""
Shared utilities for SafeCommute AI.

Contains reproducibility primitives and the canonical hash-based splitting
function. These are imported by every pipeline script and the training loop.

Key design decisions:
  - sha256 (not Python's hash()) for splitting: Python's hash() is randomized
    per process (PYTHONHASHSEED), so the same filename could land in different
    splits across runs. sha256 is deterministic across platforms and versions.
  - Source-level splitting: all chunks from the same audio file go to the same
    split, preventing data leakage. Datasets with predefined folds (ESC-50,
    UrbanSound8K) use those folds instead; sha256 is for AudioSet, YouTube,
    and violence data where no canonical folds exist.
"""

import hashlib
import random
import numpy as np
import torch

from safecommute.constants import SEED


def seed_everything(seed=SEED):
    """
    Set all random seeds for full reproducibility.

    Covers Python stdlib, numpy, PyTorch CPU, PyTorch CUDA, and cuDNN.
    Setting cudnn.deterministic=True and benchmark=False sacrifices ~5%
    GPU throughput for bitwise-reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_split(filename):
    """
    Deterministic 70/15/15 train/val/test split based on sha256 hash.

    This is the SINGLE source of truth for hash-based splitting. All
    pipeline scripts must import and use this function — never re-implement
    the hash logic locally.

    The hash is computed on the FILENAME (not full path), so moving files
    between directories does not change their split assignment. This also
    means all chunks from the same source file (which share the filename
    prefix) are split by the original filename, preventing leakage.

    Args:
        filename: The source audio filename (e.g., "ABC123_30_40.wav").

    Returns:
        One of 'train', 'val', 'test'.
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
    DataLoader worker initializer: give each forked worker a unique seed.

    PyTorch DataLoader forks workers from the parent process. Without
    explicit re-seeding, all workers inherit the SAME Python random and
    numpy random state, causing identical augmentation patterns across
    workers within a batch. This function derives a unique seed per worker
    from PyTorch's initial seed (which is set differently per worker by
    PyTorch) plus the worker_id.

    Note: torch.rand() in dataset.py is already per-worker safe (PyTorch
    handles it), but random.random() and np.random calls (used in train.py
    strong augmentation) need this explicit re-seeding.

    Pass to DataLoader as: DataLoader(..., worker_init_fn=worker_init_fn)
    """
    seed = torch.initial_seed() % (2**32) + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32))
