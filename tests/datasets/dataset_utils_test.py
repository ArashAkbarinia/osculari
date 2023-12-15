"""
Unit tests for dataset_utils.py
"""

import pytest
import numpy as np

from osculari.datasets import dataset_utils


def test_background_uniform_achromatic():
    bg_size = (256, 256)
    bg_img = dataset_utils.background_img('uniform_achromatic', bg_size)
    assert bg_img.shape == (*bg_size, 3)
    assert np.allclose(bg_img, bg_img[0, 0, :])


def test_background_uniform_colour():
    bg_size = (256, 256)
    bg_img = dataset_utils.background_img('uniform_colour', bg_size)
    assert bg_img.shape == (*bg_size, 3)
    assert np.allclose(bg_img, bg_img[0, 0, :])


def test_background_random_achromatic():
    bg_size = (256, 256)
    bg_img = dataset_utils.background_img('random_achromatic', bg_size)
    assert bg_img.shape == (*bg_size, 3)
    assert np.unique(bg_img).shape[0] > 1


def test_background_random_achromatic_pixelwise():
    bg_size = (256, 256)
    bg_img = dataset_utils.background_img('random_achromatic', bg_size)
    assert bg_img.shape == (*bg_size, 3)
    assert np.unique(bg_img).shape[0] > 1
    assert np.all(np.equal(bg_img[..., 0], bg_img[..., 1]))


def test_background_random_colour():
    bg_size = (256, 256)
    bg_img = dataset_utils.background_img('random_colour', bg_size)
    assert bg_img.shape == (*bg_size, 3)
    assert np.unique(bg_img).shape[0] > 1


def test_background_patch_achromatic():
    bg_size = (256, 256)
    bg_img = dataset_utils.background_img('patch_achromatic', bg_size)
    assert bg_img.shape == (*bg_size, 3)
    assert np.unique(bg_img).shape[0] > 1


def test_background_patch_achromatic_pixelwise():
    bg_size = (256, 256)
    bg_img = dataset_utils.background_img('patch_achromatic', bg_size)
    assert bg_img.shape == (*bg_size, 3)
    assert np.unique(bg_img).shape[0] > 1
    assert np.all(np.equal(bg_img[..., 0], bg_img[..., 1]))


def test_background_patch_colour():
    bg_size = (256, 256)
    bg_img = dataset_utils.background_img('patch_colour', bg_size)
    assert bg_img.shape == (*bg_size, 3)
    assert np.unique(bg_img).shape[0] > 1


def test_background_uniform_value():
    bg_size = (256, 256)
    bg_value = 0.5
    bg_img = dataset_utils.background_img(bg_value, bg_size)
    assert bg_img.shape == (*bg_size, 3)
    assert np.allclose(bg_img, bg_value)


def test_background_invalid_type():
    with pytest.raises(RuntimeError, match='Unsupported background type'):
        _ = dataset_utils.background_img('invalid_type', (256, 256))


def test_background_invalid_value_type():
    with pytest.raises(RuntimeError, match='Unsupported background type'):
        _ = dataset_utils.background_img(None, (256, 256))
