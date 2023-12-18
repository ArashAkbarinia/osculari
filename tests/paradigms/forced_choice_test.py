"""
Unit tests for forced_choice.py
"""

import pytest
import torch

from osculari.paradigms import forced_choice
from osculari import models


@pytest.fixture
def example_model():
    """Create an example model"""
    return models.paradigm_2afc_merge_concatenate(
        architecture='resnet18', weights=None, layers='block0', img_size=224
    )


class DummyDataset:
    """Placeholder for a dummy dataset"""

    def __init__(self, num_batches=3, batch_size=4, img_size=224, num_classes=2):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_data = (
                torch.randn((self.batch_size, 3, self.img_size, self.img_size)),
                torch.randn((self.batch_size, 3, self.img_size, self.img_size)),
                torch.randint(0, self.num_classes, (self.batch_size,))
            )
            yield batch_data


def test_epoch_loop_train(example_model):
    # Test training mode
    optimiser = torch.optim.SGD(example_model.parameters(), lr=0.01, momentum=0.9)
    dataset = DummyDataset()
    epoch_log = forced_choice.epoch_loop(example_model, dataset, optimiser, return_outputs=False)

    assert 'accuracy' in epoch_log
    assert 'loss' in epoch_log
    assert 'output' not in epoch_log  # output should not be present if return_outputs=False


def test_epoch_loop_eval(example_model):
    # Test evaluation mode
    dataset = DummyDataset()
    epoch_log = forced_choice.epoch_loop(example_model, dataset, optimiser=None,
                                         return_outputs=False)

    assert 'accuracy' in epoch_log
    assert 'loss' in epoch_log
    assert 'output' not in epoch_log  # output should not be present if return_outputs=False


def test_epoch_loop_return_outputs(example_model):
    # Test return_outputs=True
    optimiser = torch.optim.SGD(example_model.parameters(), lr=0.01, momentum=0.9)
    dataset = DummyDataset()
    epoch_log = forced_choice.epoch_loop(example_model, dataset, optimiser, return_outputs=True)

    assert 'accuracy' in epoch_log
    assert 'loss' in epoch_log
    assert 'output' in epoch_log  # output should be present if return_outputs=True


def test_test_dataset(example_model):
    # Test evaluation mode
    dataset = DummyDataset()
    epoch_log = forced_choice.test_dataset(example_model, dataset)

    assert 'accuracy' in epoch_log
    assert 'loss' in epoch_log
    assert 'output' not in epoch_log  # output should not be present


def test_predict_dataset(example_model):
    # Test evaluation mode
    dataset = DummyDataset()
    epoch_log = forced_choice.predict_dataset(example_model, dataset)

    assert 'accuracy' in epoch_log
    assert 'loss' in epoch_log
    assert 'output' in epoch_log  # output should be present
