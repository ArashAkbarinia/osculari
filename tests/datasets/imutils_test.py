"""
Unit tests for imutils_test.py
"""

import pytest
import numpy as np

from osculari.datasets import imutils


@pytest.fixture
def sample_image():
    """Create a sample grayscale image for testing"""
    return np.array([[50, 100, 150],
                     [75, 125, 175],
                     [100, 150, 200]], dtype='uint8')


def test_michelson_contrast_valid_input(sample_image):
    contrast_factor = 0.5
    result = imutils.michelson_contrast(sample_image, contrast_factor)

    # Ensure that the output has the same shape as the input
    assert result.shape == sample_image.shape

    # Ensure that the output is a NumPy array
    assert isinstance(result, np.ndarray)

    # Ensure that the contrast is applied correctly
    expected_result = np.array([[108, 120, 133],
                                [114, 126, 139],
                                [120, 133, 145]], dtype='uint8')
    np.testing.assert_almost_equal(result, expected_result)


def test_michelson_contrast_contrast_one(sample_image):
    contrast_factor = 1.0
    result = imutils.michelson_contrast(sample_image, contrast_factor)

    # Ensure that the output has the same shape as the input
    assert result.shape == sample_image.shape

    # Ensure that the output is a NumPy array
    assert isinstance(result, np.ndarray)

    # Ensure that the output is identical to input
    expected_result = sample_image
    np.testing.assert_equal(result, expected_result)


def test_michelson_contrast_invalid_contrast():
    with pytest.raises(AssertionError):
        contrast_factor = 1.5  # Invalid contrast value
        imutils.michelson_contrast(np.array([[1, 2], [3, 4]]), contrast_factor)


def test_gamma_correction_valid_input(sample_image):
    gamma_factor = 0.5
    result = imutils.gamma_correction(sample_image, gamma_factor)

    # Ensure that the output has the same shape as the input
    assert result.shape == sample_image.shape

    # Ensure that the output is a NumPy array
    assert isinstance(result, np.ndarray)

    # Ensure that gamma correction is applied correctly
    expected_result = np.array([[169, 201, 223],
                                [187, 213, 232],
                                [201, 223, 239]], dtype='uint8')
    np.testing.assert_almost_equal(result, expected_result)


def test_gamma_correction_gamma_one(sample_image):
    gamma_factor = 1.0
    result = imutils.gamma_correction(sample_image, gamma_factor)

    # Ensure that when gamma is 1, the output is the same as the input
    np.testing.assert_almost_equal(result, sample_image)


def test_gamma_correction_zero_gamma():
    with pytest.raises(AssertionError):
        gamma_factor = 0.0  # Invalid gamma value
        imutils.gamma_correction(np.array([[1, 2], [3, 4]]), gamma_factor)
