"""
A collection of adaptive psychophysical experimental methods.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from . import paradigm_utils


def staircase(model: torch.nn.Module, test_fun: Callable, dataset_fun: Callable,
              device: torch.device, low_val: float, high_val: float,
              max_attempts: Optional[int] = 20) -> npt.NDArray:
    """Computing the psychometric function following staircase procedure."""
    mid_val = (low_val + high_val) / 2
    results = []
    attempt_num = 0
    while True:
        # creating the dataset and dataloader
        dataset, batch_size, th = dataset_fun(mid_val)
        db_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        # making the test
        test_log = test_fun(model, db_loader, device)
        accuracy = np.mean(test_log['accuracy'])
        if 1 < accuracy < 0:
            raise RuntimeError('Accuracy for staircase procedure must be between 0 and 1.')
        results.append(np.array([mid_val, accuracy]))
        new_low, new_mid, new_high = paradigm_utils.midpoint(
            accuracy, low_val, mid_val, high_val, th=th
        )
        if new_mid is None or attempt_num == max_attempts:
            break
        else:
            low_val, mid_val, high_val = new_low, new_mid, new_high
        attempt_num += 1
    return np.array(results)
