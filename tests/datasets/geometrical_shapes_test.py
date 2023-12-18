"""
Unit tests for geometrical_shapes.py
"""

import pytest
import numpy as np
import torch

from osculari.datasets import ShapeAppearanceDataset


def test_shape_appearance_dataset_len():
    # Test the __len__ method of ShapeAppearanceDataset
    num_samples = 100
    dataset = ShapeAppearanceDataset(num_samples=num_samples, num_images=2, img_size=64,
                                     background='uniform_achromatic',
                                     merge_fg_bg=lambda x, y: (x, y))
    assert len(dataset) == num_samples


def test_shape_appearance_dataset_make_fg_masks():
    # Test the make_fg_masks method of ShapeAppearanceDataset
    num_samples = 100
    num_images = 2
    img_size = 64
    dataset = ShapeAppearanceDataset(num_samples=num_samples, num_images=num_images,
                                     img_size=img_size, background='uniform_achromatic',
                                     merge_fg_bg=lambda x, y: (x, y))
    fg_masks = dataset.make_fg_masks()
    assert len(fg_masks) == num_images
    assert all(
        isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (img_size, img_size)
        for mask in fg_masks)


def test_shape_appearance_dataset_make_bg_images():
    # Test the make_bg_images method of ShapeAppearanceDataset
    num_samples = 100
    num_images = 2
    img_size = 64
    dataset = ShapeAppearanceDataset(num_samples=num_samples, num_images=num_images,
                                     img_size=img_size, background='uniform_achromatic',
                                     merge_fg_bg=lambda x, y: (x, y))
    bg_images = dataset.make_bg_images()
    assert len(bg_images) == num_images
    assert all(isinstance(img, np.ndarray) and img.dtype == np.float32 and img.shape == (
        img_size, img_size, 3) for img in bg_images)


def test_shape_appearance_dataset_getitem():
    # Test the __getitem__ method of ShapeAppearanceDataset
    num_samples = 100
    num_images = 2
    img_size = 64
    dataset = ShapeAppearanceDataset(num_samples=num_samples, num_images=num_images,
                                     img_size=img_size, background='uniform_achromatic',
                                     merge_fg_bg=lambda x, y: (x, 0))
    idx = 0
    data = dataset[idx]
    assert len(data[:-1]) == num_images
    assert all(isinstance(item, np.ndarray) for item in data[:-1])
    assert data[-1] == 0  # Ground is 0


def test_shape_appearance_dataset_invalid_bg():
    # Test with an invalid background type
    num_samples = 100
    num_images = 2
    img_size = 64
    with pytest.raises(RuntimeError):
        dataset = ShapeAppearanceDataset(num_samples=num_samples, num_images=num_images,
                                         img_size=img_size, background='invalid_bg',
                                         merge_fg_bg=lambda x, y: (x, y))
        _ = dataset[0]


def test_shape_appearance_dataset_invalid_merge_fg_bg():
    # Test with an invalid merge_fg_bg function
    num_samples = 100
    num_images = 2
    img_size = 64
    with pytest.raises(AssertionError):
        _ = ShapeAppearanceDataset(num_samples=num_samples, num_images=num_images,
                                   img_size=img_size, background='uniform_achromatic',
                                   merge_fg_bg='invalid_func')
