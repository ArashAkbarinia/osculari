"""
Unit tests for paradigm_utils.py
"""

import pytest
import torch

from osculari.paradigms import paradigm_utils


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
