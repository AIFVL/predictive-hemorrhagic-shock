"""
Datasets module.

Provides functionality for creating the final training dataset
from cleaned data and feature transformations.
"""

from .make_dataset import (
    make_training_dataset,
    save_training_dataset,
    create_train_test_split,
    save_splits
)

__all__ = [
    'make_training_dataset',
    'save_training_dataset',
    'create_train_test_split',
    'save_splits'
]
