"""
Set of utility functions in generating datasets.
"""

import numpy as np
import numpy.typing as npt
import random
from typing import Any, Union, Tuple, Optional, Sequence, List


def _unique_colours(num_colours: int, channels: Optional[int] = 3,
                    exclude: [Optional[List]] = None) -> List:
    """
    Create a set of unique colors with the specified number of colors and channels.

    Parameters:
        num_colours (int): Number of unique colors to generate.
        channels (Optional[int]): Number of color channels in each color (default is 3).
        exclude (Optional[List]): List of colors to exclude from the generated set (default is None).

    Returns:
        List: A list of unique colors, each represented as a sequence of RGB values.
    """
    # Initialize exclude list if not provided
    exclude = [] if exclude is None else exclude
    colours = []

    # Generate unique colors
    for i in range(num_colours):
        while True:
            # Generate a random color
            colour = random_colour(channels=channels)

            # Check if the color is not in the generated list and not in the exclude list
            if colour not in colours and colour not in exclude:
                colours.append(colour)
                break

    return colours


def _patch_img(img_size: Tuple[int, int], num_colours: int, num_patches: int,
               channels: Optional[int] = 3) -> npt.NDArray:
    """
    Create an image of patched colours, similar to Mondrian images.

    Parameters:
        img_size (Tuple[int, int]): Size of the image in pixels (height, width).
        num_colours (int): Number of unique colours to use in the image.
        num_patches (int): Number of patches to divide the image into.
        channels (Optional[int]): Number of color channels in the image (default is 3).

    Returns:
        npt.NDArray: The generated image as a NumPy array.
    """
    # Initialise an empty image with the specified size and channels
    img = np.zeros((*img_size, channels), dtype='uint8')

    # Generate an array of unique colours
    colours = _unique_colours(num_colours, channels=channels)

    # Calculate the number of rows and columns for each patch
    patch_rows = int(np.ceil(img_size[0] / num_patches))
    patch_cols = int(np.ceil(img_size[1] / num_patches))

    colour_ind = 0  # Index for cycling through the unique colours

    # Loop through each patch and fill it with a unique colour
    for r_ind in range(num_patches):
        srow = r_ind * patch_rows
        erow = min(srow + patch_rows, img.shape[0])

        for c_ind in range(num_patches):
            scol = c_ind * patch_cols
            ecol = min(scol + patch_cols, img.shape[1])

            # Assign the current patch to a unique colour
            img[srow:erow, scol:ecol] = colours[colour_ind]

            # Move to the next colour
            colour_ind += 1

            # If all colours have been used, shuffle and reset the index
            if colour_ind == num_colours:
                random.shuffle(colours)
                colour_ind = 0

    return img


def random_colour(channels: Optional[int] = 3) -> Sequence:
    """
    Generate a random color represented as a sequence of RGB values.

    Parameters:
        channels (Optional[int]): Number of color channels in the generated color (default is 3).

    Returns:
        Sequence: A sequence representing a random color, with values in the range [0, 255].
    """
    # Generate random values for each color channel
    colour = [np.random.randint(0, 256) for _ in range(channels)]

    return colour


def _uniform_img(img_size: Tuple[int, int], value: Union[Sequence, int],
                 channels: Optional[int] = 3) -> npt.NDArray:
    """
    Create an image with uniform colors.

    Parameters:
        img_size (Tuple[int, int]): Size of the image in pixels (height, width).
        value (Union[Sequence, int]): Color value for the uniform background.
         It can be a single integer value or a sequence of values (e.g., RGB).
        channels (Optional[int]): Number of color channels in the image (default is 3).

    Returns:
        npt.NDArray: The generated image as a NumPy array.
    """
    # Convert value to NumPy array if it is a list, tuple, or ndarray
    if type(value) in (list, tuple, np.ndarray):
        value = np.array(value)

    # If value is a float, ensure it is in the range [0, 1] and convert to uint8
    if type(value) is float or (type(value) is np.ndarray and value.dtype == float):
        if np.max(value) > 1.0:
            raise RuntimeError('Uniform background image: float values must be between 0 to 1.')
        else:
            value *= 255

    # Create an image filled with the specified uniform color value
    uniform_img = np.zeros((*img_size, channels), dtype='uint8') + value

    return uniform_img


def repeat_channels(img: npt.NDArray, channels: int) -> npt.NDArray:
    """
    Add channel dimension and repeat the image along the new dimension.

    Parameters:
        img (npt.NDArray): Input image as a NumPy array.
        channels (int): Number of times to repeat the image along the new channel dimension.

    Returns:
        npt.NDArray: Image with added channel dimension and repeated along that dimension.
    """
    # Add a new channel dimension and repeat the image along the new dimension
    repeated_img = np.repeat(img[:, :, np.newaxis], channels, axis=2)

    return repeated_img


def background_img(bg_type: Any, bg_size: Union[int, Tuple], im2double=True) -> npt.NDArray:
    """
    Create a background image based on the specified type and size.

    Parameters:
        bg_type (Any): Type of the background. It can be a string representing a predefined type
         ('uniform_achromatic', 'uniform_colour', 'random_achromatic', 'random_colour', or
         'patch_colour', 'patch_achromatic'), or a value (int, float, list, tuple, or ndarray)
         representing a uniform background color.
        bg_size (Union[int, Tuple]): Size of the background image in pixels (height, width).
        im2double (bool): If True, normalize the pixel values to the range [0, 1] (default is True).

    Returns:
        npt.NDArray: The generated background image as a NumPy array.
    """
    # Ensure bg_size is in tuple format
    if type(bg_size) not in [tuple, list]:
        bg_size = (bg_size, bg_size)

    # Handle predefined background types
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
            if 'achromatic' in bg_type:
                bg_img = np.repeat(bg_img, 3, axis=2)
        else:
            raise RuntimeError('Unsupported background type %s.' % bg_type)
    # Handle user-specified background values
    elif type(bg_type) in (list, tuple, np.ndarray, int, float):
        bg_img = _uniform_img(bg_size, bg_type)
    else:
        raise RuntimeError('Unsupported background type %s.' % bg_type)

    # Normalise pixel values to the range [0, 1] if im2double is True
    return bg_img.astype('float32') / 255 if im2double else bg_img
