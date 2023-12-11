"""
Generic template for paradigms linear classifiers on top of pretrained networks.
"""

from typing import Dict, Iterator, Optional, Union

import torch
import torch.nn as nn

from . import paradigm_utils

__all__ = [
    'epoch_loop',
    'test_dataset',
    'predict_dataset'
]


def epoch_loop(model: nn.Module, dataset: Iterator, optimiser: Union[torch.optim.Optimizer, None],
               device: Optional[torch.device] = None,
               return_outputs: Optional[bool] = False) -> Dict:
    """
    Executes an epoch of training or testing.

    Args:
        model: The neural network model to be trained or tested.
        dataset: The data loader providing data batches for training or testing.
        optimiser: The optimiser used for training (if None it switches to testing mode).
        device: The device to which the model and data will be transferred
         (default: CUDA if available) (optional).
        return_outputs: Whether to return model's outputs.

    Returns:
        A dictionary containing 'accuracy' and 'loss'.
        If `return_outputs` is True, it will also include the 'output' key, which contains the
        model's outputs for each data batch.
    """

    # Set the device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine whether the model is in training or evaluation mode
    is_train = False if optimiser is None else True

    # Ensure the model is in the appropriate mode (train or eval)
    model.train() if is_train else model.eval()
    model.to(device)

    # Initialise variables to store epoch-level statistics
    accuracies = []  # List of batch-wise accuracies
    losses = []  # List of batch-wise losses
    outputs = []  # List of batch-wise model outputs

    # Iterate over data batches
    with torch.set_grad_enabled(is_train):
        for batch_ind, batch_data in enumerate(dataset):
            # Transfer data to the device
            inputs = [img.to(device) for img in batch_data[:-1]]  # Input images
            target = batch_data[-1].to(device)  # Ground-truth labels

            # Forward pass the model
            output = model(*inputs)

            # Store outputs if requested
            if return_outputs:
                outputs.extend([out.detach().cpu().numpy() for out in output])

            # Compute loss
            loss = model.loss_function(output, target)
            # Append batch-wise losses
            losses.extend([loss.item() for _ in range(inputs[0].size(0))])
            # Compute accuracy
            accuracy = paradigm_utils.accuracy(output, target)
            # Append batch-wise accuracies
            accuracies.extend([accuracy for _ in range(inputs[0].size(0))])

            # Gradient computation and optimisation (if training mode)
            if is_train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

    # Create a dictionary containing epoch-level statistics
    epoch_log = {'accuracy': accuracies, 'loss': losses}

    # Add 'output' key if requested
    if return_outputs:
        epoch_log['output'] = outputs

    return epoch_log


def test_dataset(model: nn.Module, dataset: Iterator,
                 device: Optional[torch.device] = None) -> Dict:
    """
    Executes an epoch of testing.

    Args:
        model: The neural network model to be trained or tested.
        dataset: The data loader providing data batches for training or testing.
        device: The device to which the model and data will be transferred
         (default: CUDA if available) (optional).

    Returns:
        A dictionary containing 'accuracy' and 'loss'.
    """
    return epoch_loop(model, dataset, optimiser=None, device=device)


def predict_dataset(model: nn.Module, dataset: Iterator,
                    device: Optional[torch.device] = None) -> Dict:
    """
    Executes an epoch of prediction.

    Args:
        model: The neural network model to be trained or tested.
        dataset: The data loader providing data batches for training or testing.
        device: The device to which the model and data will be transferred
         (default: CUDA if available) (optional).

    Returns:
        A dictionary containing 'accuracy', 'loss' and 'output' (model's outputs).
    """
    return epoch_loop(model, dataset, optimiser=None, device=device, return_outputs=True)
