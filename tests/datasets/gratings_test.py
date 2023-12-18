"""
Unit tests for gratings.py
"""

import pytest
import numpy as np
import torch
import torchvision.transforms as torch_transforms

from osculari.datasets import GratingsDataset


def test_gratings_dataset_len():
    # Test the __len__ method of GratingsDataset
    img_size = 64
    dataset = GratingsDataset(img_size=img_size)
    expected_length = len(dataset.thetas) * len(dataset.sfs)
    assert len(dataset) == expected_length


def test_gratings_dataset_make_grating():
    # Test the make_grating method of GratingsDataset
    img_size = 64
    dataset = GratingsDataset(img_size=img_size)
    idx = 0
    amplitude = 1.0
    channels = 3
    grating = dataset.make_grating(idx, amplitude, channels)
    assert isinstance(grating, np.ndarray)
    assert grating.shape == (img_size, img_size, channels)


def test_gratings_dataset_getitem():
    # Test the __getitem__ method of GratingsDataset
    img_size = 64
    dataset = GratingsDataset(img_size=img_size)

    # Test without transformation
    idx = 0
    grating = dataset[idx]
    assert isinstance(grating, np.ndarray)
    assert grating.shape == (img_size, img_size, 3)

    # Test with transformation
    transform = torch_transforms.Compose([torch_transforms.ToTensor()])
    dataset.transform = transform
    grating = dataset[idx]
    assert isinstance(grating, torch.Tensor)
    assert grating.shape == (3, img_size, img_size)


def test_gratings_dataset_with_gaussian():
    # Test the make_grating method of GratingsDataset
    img_size = 64
    dataset = GratingsDataset(img_size=img_size, gaussian_sigma=0.5)
    idx = 0
    amplitude = 1.0
    channels = 3
    grating = dataset.make_grating(idx, amplitude, channels)
    assert isinstance(grating, np.ndarray)
    assert grating.shape == (img_size, img_size, channels)
