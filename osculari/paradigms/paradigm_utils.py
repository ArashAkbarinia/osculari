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


def accuracy_preds(output: torch.Tensor, target: torch.Tensor,
                   topk: Optional[Sequence] = (1,)) -> (List, List):
    """
    Computes the accuracy over the k top predictions.

    Args:
        output: The model's output tensor containing predictions for each input sample.
        target: The ground-truth labels for each input sample.
        topk: An optional list of top-k accuracy thresholds to be computed (e.g., (1, 5)).

    Returns:
        A tuple containing the computed accuracies and correct predictions for each top-k threshold.
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
    This function computes the accuracy of a model's prediction on a given set of data.

    Args:
        output: The model's predicted output (torch.Tensor).
        target: The ground truth labels (torch.Tensor).

    Returns:
        The accuracy of the model's predictions (float).
    """

    if output.shape[1] == 1:
        # Check if the model produces one-dimensional predictions
        pred = torch.equal(torch.gt(output, 0), target.float())
        return pred.float().mean(0, keepdim=True)[0]

    # Otherwise, the model produces multidimensional predictions
    acc, _ = accuracy_preds(output, target, topk=[1])
    return acc[0].item()  # Extract the top-1 accuracy


def circular_mean(a: float, b: float) -> float:
    """
       Computes the circular mean of two values.

       Args:
           a: The first value (float).
           b: The second value (float).

       Returns:
           The circular mean of the two values (float).
       """
    mu = (a + b + 1) / 2 if abs(a - b) > 0.5 else (a + b) / 2
    return mu if mu >= 1 else mu - 1


def _compute_avg(a: Union[float, npt.NDArray], b: Union[float, npt.NDArray],
                 circular_channels: Optional[List] = None) -> Union[float, npt.NDArray]:
    if circular_channels is None:
        circular_channels = []
    if type(a) is np.ndarray:
        a, b = a.copy().squeeze(), b.copy().squeeze()
    c = (a + b) / 2
    for i in circular_channels:
        c[i] = circular_mean(a[i], b[i])
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
    Finds the midpoint of the stimulus range based on the current accuracy and the target accuracy
     threshold.

    Args:
        acc: The current accuracy value (float).
        low: The lower bound of the stimulus range (float or NumPy array).
        mid: The current midpoint of the stimulus range (float or NumPy array).
        high: The upper bound of the stimulus range (float or NumPy array).
        th: The target accuracy threshold (float).
        ep: The convergence tolerance (float; optional).
        circular_channels: The list of circular channels for applying circular arithmetic when
         computing the average (list; optional).

    Returns:
        The new low, mid, and high values of the stimulus range based on the current accuracy and
         the target accuracy threshold.
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


def train_linear_probe(model: ProbeNet, dataset: Union[TorchDataset, TorchDataLoader],
                       epoch_loop: Callable[[nn.Module, TorchDataLoader, Any, torch.device], Dict],
                       out_dir: str, device: Optional[torch.device] = None,
                       epochs: Optional[int] = 10,
                       optimiser: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[lr_scheduler.LRScheduler] = None) -> Dict:
    """
        Trains the linear probe network on the specified dataset.

        Args:
            model: The linear probe network to train.
            dataset: The dataset or dataloader for training.
            epoch_loop: A function to perform an epoch of training or testing. This function
             must accept for positional arguments (i.e., model, train_loader, optimiser, device).
             This function should return a dictionary.
            out_dir: The output directory for saving checkpoints.
            device: The device to use for training (Optional).
            epochs: The number of epochs to train for (Optional).
            optimiser: The optimiser to use for training (default: SGD) (Optional).
            scheduler: The learning rate scheduler to use
             (default: MultiStepLR at 50 and 80% of epochs) (Optional).

        Returns:
            A dictionary containing training logs.
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
