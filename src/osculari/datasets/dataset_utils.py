"""
Set of utility functions in generating datasets.
"""

import numpy as np
import numpy.typing as npt
import random
from typing import Any, Union, Tuple, Optional, Sequence, List


def _unique_colours(num_colours: int, channels: Optional[int] = 3,
                    exclude: [Optional[List]] = None) -> List:
    """Creating a set of unique colours"""
    exclude = [] if exclude is None else exclude
    colours = []
    for i in range(num_colours):
        while True:
            colour = random_colour(channels=channels)
            if colour not in colours and colour not in exclude:
                colours.append(colour)
                break
    return colours


def _patch_img(img_size: Tuple[int, int], num_colours: int, num_patches: int,
               channels: Optional[int] = 3) -> npt.NDArray:
    """Create an image of patched colours, similar to Mondrian images."""
    img = np.zeros((*img_size, channels), dtype='uint8')
    colours = _unique_colours(num_colours, channels=channels)
    patch_rows = int(np.ceil(img_size[0] / num_patches))
    patch_cols = int(np.ceil(img_size[1] / num_patches))
    colour_ind = 0
    for r_ind in range(num_patches):
        srow = r_ind * patch_rows
        erow = min(srow + patch_rows, img.shape[0])
        for c_ind in range(num_patches):
            scol = c_ind * patch_cols
            ecol = min(scol + patch_cols, img.shape[1])
            img[srow:erow, scol:ecol] = colours[colour_ind]
            colour_ind += 1
            if colour_ind == num_colours:
                random.shuffle(colours)
                colour_ind = 0
    return img


def random_colour(channels: Optional[int] = 3) -> Sequence:
    """Generating a random colour."""
    return [np.random.randint(0, 256) for _ in range(channels)]


def michelson_contrast(img: npt.NDArray, contrast: float) -> npt.NDArray:
    """Adjusting the contrast of an image."""
    assert 0 <= contrast <= 1
    return ((1 - contrast) / 2.0 + np.multiply(img, contrast)).astype(img.dtype)


def _uniform_img(img_size: Tuple[int, int], value: Union[Sequence, int],
                 channels: Optional[int] = 3) -> npt.NDArray:
    """Creating an image with uniform colours."""
    if type(value) in (list, tuple, np.ndarray):
        value = np.array(value)
    if type(value) is float or (type(value) is np.ndarray and value.dtype == float):
        if np.max(value) > 1.0:
            raise RuntimeError('Uniform background image: float values must be between 0 to 1.')
        else:
            value *= 255
    return np.zeros((*img_size, channels), dtype='uint8') + value


def repeat_channels(img: npt.NDArray, channels: int) -> npt.NDArray:
    """Adding channel dimension and repeating the image."""
    return np.repeat(img[:, :, np.newaxis], channels, axis=2)


def background_img(bg_type: Any, bg_size: Union[int, Tuple], im2double=True) -> npt.NDArray:
    """Creating a background image."""
    if type(bg_size) not in [tuple, list]:
        bg_size = (bg_size, bg_size)
    if type(bg_type) in [str, np.str_]:
        if bg_type == 'uniform_achromatic':
            bg_img = _uniform_img(bg_size, np.random.randint(0, 256, dtype='uint8'))
        elif bg_type == 'uniform_colour':
            bg_img = _uniform_img(bg_size, random_colour())
        elif bg_type == 'random_achromatic':
            bg_img = repeat_channels(np.random.randint(0, 256, bg_size, dtype='uint8'), 3)
        elif bg_type == 'random_colour':
            bg_img = np.random.randint(0, 256, (*bg_size, 3), dtype='uint8')
        elif 'patch_' in bg_type:
            channels = 3 if 'colour' in bg_type else 1
            num_colours = np.random.randint(3, 25)
            num_patches = np.random.randint(2, bg_size[0] // 20)
            bg_img = _patch_img(bg_size, num_colours, num_patches, channels)
        else:
            raise RuntimeError('Unsupported background type %s.' % bg_type)
    elif type(bg_type) in (list, tuple, np.ndarray, int, float):
        bg_img = _uniform_img(bg_size, bg_type)
    else:
        raise RuntimeError('Unsupported background type %s.' % bg_type)
    return bg_img.astype('float32') / 255 if im2double else bg_img
