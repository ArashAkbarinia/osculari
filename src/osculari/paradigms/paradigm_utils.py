"""
Utility function for paradigms.
"""

import numpy as np
import numpy.typing as npt
from typing import Union, Optional, List

import torch


def accuracy_preds(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        accuracies = []
        corrects = []
        for k in topk:
            corrects.append(correct[:k])
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            accuracies.append(correct_k / batch_size)
    return accuracies, corrects


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    # TODO: better documentation and unifying different output shapes
    if output.shape[1] == 1:
        pred = torch.equal(torch.gt(output, 0), target)
        return pred.float().mean(0, keepdim=True)[0]
    acc, _ = accuracy_preds(output, target, topk=[1])
    return acc[0].item()


def circular_mean(a: float, b: float) -> float:
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
    diff_acc = acc - th
    if abs(diff_acc) < ep:
        return None, None, None
    elif diff_acc > 0:
        new_mid = _compute_avg(low, mid, circular_channels)
        return low, new_mid, mid
    else:
        new_mid = _compute_avg(high, mid, circular_channels)
        return mid, new_mid, high
