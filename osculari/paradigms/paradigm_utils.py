"""
Utility function for paradigms.
"""

import os
import numpy as np
import numpy.typing as npt
from typing import Union, Optional, List, Callable, Dict, Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.optim import lr_scheduler

from ..models.readout import ProbeNet

__all__ = [
    'train_linear_probe'
]


def _accuracy_preds(output: torch.Tensor, target: torch.Tensor,
                    topk: Optional[Sequence] = (1,)) -> (List[float], List[torch.Tensor]):
    """
    Compute accuracy and correct predictions for the top-k thresholds.

    Parameters:
        output (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth labels.
        topk (Optional[Sequence]): Top-k thresholds for accuracy computation. Default is (1,).

    Returns:
        Tuple[List[float], List[torch.Tensor]]: List of accuracies for each top-k threshold,
                                                list of correct predictions for each top-k
                                                threshold.
    """
    with torch.inference_mode():  # Ensure that the model is in inference mode
        maxk = max(topk)  # Extract the maximum top-k value
        batch_size = target.size(0)  # Get the batch size
        # Check if the target is a 2D tensor (i.e., multi-class classification)
        if target.ndim == 2:
            target = target.max(dim=1)[1]  # Extract the single class label for each sample

        _, pred = output.topk(maxk, 1, True, True)  # Extract the top-k predictions
        pred = pred.t()  # Transpose the predictions to match the target format
        correct = pred.eq(target[None])  # Create a tensor indicating the correct predictions

        accuracies = []  # List to store the computed accuracies
        corrects = []  # List to store the correct predictions for each top-k threshold
        for k in topk:  # Iterate over each top-k threshold
            # Extract the correct predictions for the current threshold
            corrects.append(correct[:k])
            # Compute the sum of correct predictions
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            accuracies.append(correct_k / batch_size)

    return accuracies, corrects


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute the accuracy of model predictions.

    Parameters:
        output (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth labels.

    Returns:
        float: Accuracy of the model predictions.
    """
    # Ensure the output has two dimensions (Linear layer output is two-dimensional)
    assert len(output.shape) == 2
    # Ensure output and target have the same number of elements
    assert len(output) == len(target)

    # Check if the model is performing binary classification
    if output.shape[1] == 1:
        # Convert to binary predictions (greater than 0)
        output_class = torch.gt(output, 0).flatten()
        # Compute accuracy for binary classification
        pred = torch.eq(output_class, target)
        return pred.float().mean().item()

    # Otherwise, the model produces multidimensional predictions
    acc, _ = _accuracy_preds(output, target, topk=[1])
    return acc[0].item()  # Extract the top-1 accuracy


def _circular_mean(a: float, b: float) -> float:
    """
    Compute the circular mean of two variables in the range of 0 to 1.

    Parameters:
        a (float): First angle in radians.
        b (float): Second angle in radians.

    Returns:
        float: Circular mean of the two angles.
    """
    # Ensure a and b are in the range of 0 to 1
    assert 0 <= a <= 1
    assert 0 <= b <= 1
    # Calculate the circular mean using a conditional expression
    mu = (a + b + 1) / 2 if abs(a - b) > 0.5 else (a + b) / 2
    # Adjust the result to be in the range [0, 1)
    return mu if mu >= 1 else mu - 1


def _compute_avg(a: Union[float, npt.NDArray], b: Union[float, npt.NDArray],
                 circular_channels: Optional[List] = None) -> Union[float, npt.NDArray]:
    if circular_channels is None:
        circular_channels = []
    if type(a) is np.ndarray:
        a, b = a.copy().squeeze(), b.copy().squeeze()
    c = (a + b) / 2
    for i in circular_channels:
        c[i] = _circular_mean(a[i], b[i])
    return c


def midpoint(
        acc: float, low: Union[float, npt.NDArray], mid: Union[float, npt.NDArray],
        high: Union[float, npt.NDArray], th: float, ep: Optional[float] = 1e-4,
        circular_channels: Optional[List] = None
) -> (
        Union[float, npt.NDArray, None], Union[float, npt.NDArray, None],
        Union[float, npt.NDArray, None]
):
    """
    Compute new midpoints for a given accuracy in a binary search.

    Parameters:
        acc (float): Current accuracy.
        low (Union[float, npt.NDArray]): Low value in the search space.
        mid (Union[float, npt.NDArray]): Midpoint in the search space.
        high (Union[float, npt.NDArray]): High value in the search space.
        th (float): Target accuracy.
        ep (Optional[float]): Acceptable range around the target accuracy. Default is 1e-4.
        circular_channels (Optional[List]): List of circular channels. Default is None.

    Returns:
        (Union[float, npt.NDArray, None], Union[float, npt.NDArray, None], Union[float, npt.NDArray, None]):
        Tuple containing the updated low, mid, and high values.
        If the accuracy is within the acceptable range of the target accuracy, returns
        (None, None, None).
    """
    # Calculate the difference between the current accuracy and the target accuracy
    diff_acc = acc - th

    # Check if the accuracy is within the acceptable range of the target accuracy
    if abs(diff_acc) < ep:
        return None, None, None

    # Check if the current accuracy is above the target accuracy
    if diff_acc > 0:
        # Compute the new midpoint by averaging the current low and mid values
        new_mid = _compute_avg(low, mid, circular_channels)

        # Update the low and mid values
        return low, new_mid, mid

    # Otherwise, the current accuracy is below the target accuracy
    else:
        # Compute the new midpoint by averaging the current mid and high values
        new_mid = _compute_avg(high, mid, circular_channels)

        # Update the mid and high values
        return mid, new_mid, high


def train_linear_probe(
        model: ProbeNet,
        dataset: Union[TorchDataset, TorchDataLoader],
        epoch_loop: Callable[[nn.Module, TorchDataLoader, Any, torch.device], Dict],
        out_dir: str,
        device: Optional[torch.device] = None,
        epochs: Optional[int] = 10,
        optimiser: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[lr_scheduler.LRScheduler] = None
) -> Dict:
    """
    Train a linear probe on top of a frozen backbone model.

    Parameters:
        model (ProbeNet): Linear probe model.
        dataset (Union[TorchDataset, TorchDataLoader]): Training dataset or data loader.
        epoch_loop (Callable): Function defining the training loop for one epoch.  This function
         must accept for positional arguments (i.e., model, train_loader, optimiser, device).
         This function should return a dictionary.
        out_dir (str): Output directory to save checkpoints.
        device (Optional[torch.device]): Device on which to perform training.
        epochs (Optional[int]): Number of training epochs. Default is 10.
        optimiser (Optional[torch.optim.Optimizer]): Optimization algorithm. Default is SGD.
        scheduler (Optional[lr_scheduler.LRScheduler]): Learning rate scheduler. Default is
         MultiStepLR at 50 and 80% of epochs

    Returns:
        Dict: Training logs containing statistics.
    """
    # Data loading
    if isinstance(dataset, TorchDataLoader):
        train_loader = dataset
    else:
        train_loader = TorchDataLoader(
            dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, sampler=None
        )

    # Optimisation
    model.freeze_backbone()
    if optimiser is None:
        # Create an optimiser for the linear probe parameters
        params_to_optimize = [{'params': [p for p in model.fc.parameters()]}]
        optimiser = torch.optim.SGD(params_to_optimize, lr=0.1, momentum=0.9, weight_decay=1e-4)

    if scheduler is None:
        # Create a learning rate scheduler
        milestones = [int(epochs * e) for e in [0.5, 0.8]]
        scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=milestones, gamma=0.1)

    # Training loop
    training_logs = dict()
    for epoch in range(epochs):
        # Run an epoch of training
        train_log = epoch_loop(model, train_loader, optimiser, device)

        # Update the learning rate scheduler
        scheduler.step()

        # Store training statistics
        for log_key, log_val in train_log.items():
            if log_key not in training_logs:
                training_logs[log_key] = []
            training_logs[log_key].append(np.mean(log_val))

        # Print training summary
        log_str = ' '.join('%s=%.3f' % (key, val[-1]) for key, val in training_logs.items())
        print('[%.3d] %s' % (epoch, log_str))

        # Save checkpoints
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, 'checkpoint.pth.tar')
        torch.save({
            'epoch': epoch,
            'network': model.serialisation_params(),
            'optimizer': optimiser.state_dict(),
            'scheduler': scheduler.state_dict(),
            'log': training_logs
        }, file_path)

    return training_logs
