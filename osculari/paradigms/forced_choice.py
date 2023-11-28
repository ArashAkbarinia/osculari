"""
Generic template for paradigms linear classifiers on top of pretrained networks.
"""

from typing import Dict, Iterator, Optional, Union

import torch
import torch.nn as nn

from . import paradigm_utils


def epoch_loop(model: nn.Module, dataset: Iterator, optimiser: Union[torch.optim.Optimizer, None],
               device: torch.device, return_outputs: Optional[bool] = False) -> Dict:
    # usually the code for train/test has a large overlap.
    is_train = False if optimiser is None else True
    # model should be in train/eval model accordingly
    model.train() if is_train else model.eval()

    accuracies = []
    losses = []
    outputs = []
    with torch.set_grad_enabled(is_train):
        for batch_ind, batch_data in enumerate(dataset):
            # moving the image and GT to device
            inputs = [img.to(device) for img in batch_data[:-1]]
            target = batch_data[-1].to(device)

            # calling the network
            output = model(*inputs)
            if return_outputs:
                outputs.extend([out for out in output])

            # computing the loss function
            loss = model.loss_function(output, target)
            losses.extend([loss.item() for _ in range(inputs[0].size(0))])
            # computing the accuracy
            accuracy = paradigm_utils.accuracy(output, target)
            accuracies.extend([accuracy for _ in range(inputs[0].size(0))])

            # compute gradient and do SGD step
            if is_train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
    epoch_log = {'accuracy': accuracies, 'loss': losses}
    if return_outputs:
        epoch_log['output'] = outputs
    return epoch_log


def test_dataset(model: nn.Module, dataset: Iterator, device: torch.device) -> Dict:
    return epoch_loop(model, dataset, optimiser=None, device=device)


def predict_dataset(model: nn.Module, dataset: Iterator, device: torch.device) -> Dict:
    return epoch_loop(model, dataset, optimiser=None, device=device, return_outputs=True)
