"""
Unit tests for adaptive_psychophysics_test.py
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from osculari.paradigms import staircase


class SimpleModel(nn.Module):
    """Placeholder for a simple neural network model"""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(22, 1)

    def forward(self, x):
        return self.fc(x)


class DummyDataset(Dataset):
    """Placeholder for a dummy dataset"""

    def __init__(self):
        self.data = torch.randn((10, 22))
        self.labels = torch.randint(0, 2, (10,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def dummy_test_function(model, dataloader, device):
    """Dummy test function for evaluation"""
    model.to(device)
    outputs = []
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        outputs.extend(output.detach().cpu().numpy())
    accuracy = np.random.uniform()
    return {'accuracy': [accuracy]}


def dummy_dataset_function(_mid_val):
    """Dummy dataset function for creating the dataset and dataloader"""
    batch_size = 10
    th = 0.1
    return DummyDataset(), batch_size, th


def test_staircase():
    # Test the basic functionality of the staircase procedure
    model = SimpleModel()
    low_val = 0.0
    high_val = 1.0
    result = staircase(model, dummy_test_function, dummy_dataset_function, low_val, high_val,
                       max_attempts=5)

    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 2  # The result should have two columns (midpoint, accuracy)
    assert result.shape[0] <= 5  # The result should have at most 5 data points (max_attempts)


def test_staircase_invalid_accuracy():
    # Test when the accuracy is outside the acceptable range
    model = SimpleModel()
    low_val = 0.0
    high_val = 1.0

    def dummy_invalid_test_function(_model, _dataloader, _device):
        """Dummy function for invalid accuracy."""
        return {'accuracy': [1.5]}  # Invalid accuracy value

    with pytest.raises(RuntimeError):
        _ = staircase(model, dummy_invalid_test_function, dummy_dataset_function, low_val, high_val)


def test_staircase_max_attempts():
    # Test when the staircase procedure reaches the maximum number of attempts
    model = SimpleModel()
    low_val = 0.0
    high_val = 1.0
    result = staircase(model, dummy_test_function, dummy_dataset_function, low_val, high_val,
                       max_attempts=1)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1  # The result should have only one data point (max_attempts)
