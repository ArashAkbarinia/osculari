"""
A collection of adaptive psychophysical experimental methods.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable, Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

from . import paradigm_utils

__all__ = [
    'staircase'
]


def staircase(model: nn.Module,
              test_fun: Callable[[nn.Module, TorchDataLoader, torch.device], Dict],
              dataset_fun: Callable[[float], Tuple],
              low_val: float, high_val: float, device: Optional[torch.device] = None,
              max_attempts: Optional[int] = 20) -> npt.NDArray:
    """
        Computes the psychometric function following the staircase procedure.

        Args:
            model: The neural network model to be evaluated.
            test_fun: Function for evaluating the model. This function must accept three
             positional arguments (i.e., model, db_loader, device). The output of this function
             should be a dictionary containing the key `accuracy`.
            dataset_fun: Function for creating the dataset and dataloader. This function must
             accept one argument (mid_val, i.e., the current value to be tested). This funtion must
             return a tuple of three elements (i.e., dataset, batch_size, threshold).
            low_val: The lower bound of the stimulus range.
            high_val: The upper bound of the stimulus range.
            device: The device to which the model and data will be transferred
             (default: CUDA if available) (optional).
            max_attempts: The maximum number of attempts allowed in the staircase procedure.

        Returns:
            A NumPy array containing the psychometric function data points containing two columns,
            the first column tested values and the second column obtained accuracies.
        """

    # Set the device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Calculate the midpoint of the initial stimulus range
    mid_val = (low_val + high_val) / 2

    # List to store the psychometric function data points
    results = []

    # Number of attempts to perform the staircase procedure
    attempt_num = 1

    # Perform the staircase procedure until convergence
    while True:
        # Create the dataset and dataloader for the current midpoint
        dataset, batch_size, th = dataset_fun(mid_val)
        db_loader = TorchDataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        # Evaluate the model on the current midpoint
        test_log = test_fun(model, db_loader, device)
        accuracy = np.mean(test_log['accuracy'])

        # Check if accuracy is within the acceptable range
        if 1 < accuracy or accuracy < 0:
            raise RuntimeError('Accuracy for staircase procedure must be between 0 and 1.')

        # Append the current midpoint and accuracy to the results list
        results.append(np.array([mid_val, accuracy]))

        # Calculate the new stimulus range for the next iteration
        new_low, new_mid, new_high = paradigm_utils.midpoint(
            accuracy, low_val, mid_val, high_val, th=th
        )

        # Check if the procedure has converged or reached the maximum number of attempts
        if new_mid is None or attempt_num == max_attempts:
            break
        else:
            # Update the stimulus range
            low_val, mid_val, high_val = new_low, new_mid, new_high

        # Increment the attempt counter
        attempt_num += 1

    # Return the psychometric function data points
    return np.array(results)
