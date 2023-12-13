"""
Image manipulation functions.
"""

import numpy as np
import numpy.typing as npt


def _img_max_val(image):
    max_val = np.maximum(np.max(image), 1)
    for bits in [8, 16, 32, 64]:
        if image.dtype == 'uint%d' % bits:
            max_val = (2 ** bits) - 1
            break
    return max_val


def _im2double(image):
    return np.float32(image) / _img_max_val(image)


def _double2im(image, org_img):
    return (image * _img_max_val(org_img)).astype(org_img.dtype)


def _process_img(fun, in_image, *args, **kwargs):
    image = _im2double(in_image.copy())
    image = fun(image, *args, **kwargs)
    return _double2im(fun(image, *args, **kwargs), in_image)


def michelson_contrast(img: npt.NDArray, contrast: float) -> npt.NDArray:
    """
    Adjust the contrast of an image using the Michelson contrast formula.

    Parameters:
        img (npt.NDArray): Input image as a NumPy array.
        contrast (float): Contrast adjustment factor. Should be in the range [0, 1].

    Returns:
        npt.NDArray: Image with adjusted contrast.
    """
    # Ensure that the contrast value is within the valid range
    assert 0 <= contrast <= 1

    # Check if contrast is already at maximum
    if contrast == 1:
        return img

    # Apply the contrast adjustment
    return _process_img(_adjust_contrast, img, contrast)


def _adjust_contrast(image, amount):
    return (1 - amount) / 2.0 + np.multiply(image, amount)


def gamma_correction(img: npt.NDArray, gamma: float) -> npt.NDArray:
    """
    Adjust the gamma of an image.

    Parameters:
        img (npt.NDArray): Input image as a NumPy array.
        gamma (float): Gamma adjustment factor.

    Returns:
        npt.NDArray: Image with adjusted gamma.
    """
    # Ensure that the gamma value is not zero
    assert gamma != 0

    # Check if gamma is already at default (gamma=1)
    if gamma == 1:
        return img

    # Apply the gamma adjustment
    return _process_img(_adjust_gamma, img, gamma)


def _adjust_gamma(image, amount):
    return image ** amount
