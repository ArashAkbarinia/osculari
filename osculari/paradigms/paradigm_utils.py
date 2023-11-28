"""
Utility function for paradigms.
"""

import os
import numpy as np
import numpy.typing as npt
from typing import Union, Optional, List, Callable, Dict

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.optim import lr_scheduler

from ..models.readout import ProbeNet

__all__ = [
    'train_linear_probe'
]


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


def train_linear_probe(model: ProbeNet, dataset: Union[TorchDataset, TorchDataLoader],
                       epoch_loop: Callable, out_dir: str,
                       device: Optional[torch.device] = None, epochs: Optional[int] = 10,
                       optimiser: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[lr_scheduler.LRScheduler] = None) -> Dict:
    # dataloader
    train_loader = dataset if type(dataset) is TorchDataLoader else TorchDataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, sampler=None
    )

    # device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # optimiser
    model.freeze_backbone()
    model = model.to(device)
    params_to_optimize = [{'params': [p for p in model.fc.parameters()]}]
    if optimiser is None:
        optimiser = torch.optim.SGD(params_to_optimize, lr=0.1, momentum=0.9, weight_decay=1e-4)

    # scheduler
    if scheduler is None:
        milestones = [int(epochs * e) for e in [0.5, 0.8]]
        scheduler = lr_scheduler.MultiStepLR(optimiser, milestones=milestones, gamma=0.1)

    # doing epoch
    training_logs = dict()
    for epoch in range(epochs):
        train_log = epoch_loop(model, train_loader, optimiser, device)
        scheduler.step()
        # loading the log dict
        for log_key, log_val in train_log.items():
            if log_key not in training_logs:
                training_logs[log_key] = []
            training_logs[log_key].append(np.mean(log_val))
        log_str = ' '.join('%s=%.3f' % (key, val[-1]) for key, val in training_logs.items())
        print('[%.3d] %s' % (epoch, log_str))
        # saving the checkpoint
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
