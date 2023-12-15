"""
Unit tests for paradigm_utils.py
"""

import pytest
import numpy as np
import shutil
import os
import torch
from torch.utils.data import Dataset

from osculari.paradigms import paradigm_utils
from osculari import models


def test_accuracy_binary_classification():
    # Test accuracy for binary classification predictions
    output = torch.tensor([0.2, -0.1, 0.8, -0.4]).view(4, 1)
    target = torch.tensor([1, 0, 1, 0])
    acc = paradigm_utils.accuracy(output, target)
    assert acc == 1.0


def test_accuracy_multi_classification():
    # Test accuracy for multi-class predictions
    output = torch.tensor([[0.2, -0.1, 0.8, -0.4], [0.1, 0.3, -0.2, 0.5]])
    target = torch.tensor([2, 0])
    acc = paradigm_utils.accuracy(output, target)
    assert acc == 0.5


def test_accuracy_invalid_input():
    # Test with invalid input (different shapes)
    output = torch.tensor([[0.2, -0.1, 0.8, -0.4], [0.1, 0.3, -0.2, 0.5]])
    target = torch.tensor([2, 0, 1])  # Invalid target shape
    with pytest.raises(AssertionError):
        paradigm_utils.accuracy(output, target)


def test_accuracy_zero_dimensional():
    # Test with zero-dimensional input (should raise an error)
    output = torch.tensor(0.5)
    target = torch.tensor(1)
    with pytest.raises(AssertionError):
        paradigm_utils.accuracy(output, target)


def test_accuracy_one_dimensional_equal():
    # Test accuracy for one-dimensional predictions where output and target are equal
    output = torch.tensor([0.2, -0.1, 0.8, -0.4]).view(4, 1)
    target = torch.tensor([0, 0, 1, 0])
    acc = paradigm_utils.accuracy(output, target)
    assert acc == 0.75


def test_accuracy_multi_target():
    # Test accuracy for multi-class predictions
    output = torch.tensor([[0.2, -0.1, 0.8, -0.4], [0.1, 0.3, -0.2, 0.5]])
    target = torch.tensor([[0.1, 0.1, 0.8, 0.0], [0.6, 0.3, 0.1, 0.0]])
    acc = paradigm_utils.accuracy(output, target)
    assert acc == 0.5


def test_midpoint_within_acceptable_range():
    # Test when the accuracy is within the acceptable range
    acc = 0.5
    low = 0.4
    mid = 0.5
    high = 0.6
    th = 0.5
    ep = 0.1
    circular_channels = None

    updated_low, updated_mid, updated_high = paradigm_utils.midpoint(
        acc, low, mid, high, th, ep, circular_channels
    )

    assert updated_low is None
    assert updated_mid is None
    assert updated_high is None


def test_midpoint_accuracy_above_target():
    # Test when the accuracy is above the target
    acc = 0.7
    low = 0.4
    mid = 0.5
    high = 0.6
    th = 0.5
    ep = 0.1
    circular_channels = None

    updated_low, updated_mid, updated_high = paradigm_utils.midpoint(
        acc, low, mid, high, th, ep, circular_channels
    )

    # Check if the new midpoint is computed correctly
    assert np.isclose(updated_low, low)
    assert np.isclose(updated_mid, (low + mid) / 2)
    assert np.isclose(updated_high, mid)


def test_midpoint_accuracy_below_target():
    # Test when the accuracy is below the target
    acc = 0.3
    low = 0.4
    mid = 0.5
    high = 0.6
    th = 0.5
    ep = 0.1
    circular_channels = None

    updated_low, updated_mid, updated_high = paradigm_utils.midpoint(
        acc, low, mid, high, th, ep, circular_channels
    )

    # Check if the new midpoint is computed correctly
    assert np.isclose(updated_low, mid)
    assert np.isclose(updated_mid, (mid + high) / 2)
    assert np.isclose(updated_high, high)


def test_midpoint_circular_channels():
    # Test with circular channels
    acc = 0.7
    low = np.array([350 / 360, 350])
    mid = np.array([10 / 360, 10])
    high = np.array([20 / 360, 20])
    th = 0.5
    ep = 0.1
    circular_channels = [0]

    updated_low, updated_mid, updated_high = paradigm_utils.midpoint(
        acc, low, mid, high, th, ep, circular_channels
    )

    # Check if the new midpoint is computed correctly for circular channels
    assert np.allclose(updated_low, low)
    assert np.allclose(updated_mid, np.array([(low[0] + mid[0] + 1) / 2, (low[1] + mid[1]) / 2]))
    assert np.allclose(updated_high, mid)


def test_midpoint_circular_channels_invalid_input():
    # Test with circular channels
    acc = 0.7
    low = np.array([350, 350])
    mid = np.array([10, 10])
    high = np.array([20, 20])
    th = 0.5
    ep = 0.1
    circular_channels = [0]

    with pytest.raises(AssertionError):
        _ = paradigm_utils.midpoint(acc, low, mid, high, th, ep, circular_channels)


class DummyDataset(Dataset):
    """Placeholder for a dummy dataset"""

    def __init__(self):
        self.data = torch.randn((10, 3, 224, 224))
        self.labels = torch.randint(0, 2, (10,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def dummy_epoch_loop(model, train_loader, optimiser, device):
    """Placeholder for the epoch_loop function"""
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        output = model(data, data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimiser.step()
    return {'loss': [loss.item()]}


def test_train_linear_probe():
    model = models.paradigm_2afc_merge_concatenate(
        architecture='resnet18', weights=None, layers='block0', img_size=224
    )
    dataset = DummyDataset()
    out_dir = "test_output"
    device = torch.device("cpu")
    epochs = 5

    training_logs = paradigm_utils.train_linear_probe(
        model=model,
        dataset=dataset,
        epoch_loop=dummy_epoch_loop,
        out_dir=out_dir,
        device=device,
        epochs=epochs,
    )

    assert 'loss' in training_logs
    assert len(training_logs['loss']) == epochs

    # Check if the output directory is created and the checkpoint file exists
    assert os.path.exists(out_dir)
    checkpoint_path = os.path.join(out_dir, 'checkpoint.pth.tar')
    assert os.path.exists(checkpoint_path)

    # Check if the checkpoint file is valid
    checkpoint = torch.load(checkpoint_path)
    assert 'epoch' in checkpoint
    assert 'network' in checkpoint
    assert 'optimizer' in checkpoint
    assert 'scheduler' in checkpoint
    assert 'log' in checkpoint

    # Clean up the temporary test output directory
    shutil.rmtree(out_dir)
