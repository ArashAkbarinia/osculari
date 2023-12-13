"""
Unit tests for imutils_test.py
"""

import numpy as np
import pytest

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


def test_michelson_contrast_invalid_contrast():
    with pytest.raises(AssertionError):
        contrast_factor = 1.5  # Invalid contrast value
        imutils.michelson_contrast(np.array([[1, 2], [3, 4]]), contrast_factor)
