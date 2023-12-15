"""
A simple generic dataset of geometrical shapes in foreground.
"""

import numpy as np
import numpy.typing as npt
import math
import random
from typing import Optional, List, Tuple, Sequence, Callable, Any, Union

import cv2
import torch
from torch.utils.data import Dataset as TorchDataset

from . import dataset_utils

__all__ = [
    'ShapeAppearanceDataset'
]


def generate_random_polygon(num_sides: int, centre: Optional[Tuple[float, float]] = (0.0, 0.0),
                            radius: Optional[float] = 1.0) -> List[Tuple[float, float]]:
    """
    Generates a random polygon with the specified number of sides.

    Args:
        num_sides: The number of sides of the polygon (int).
        centre: The coordinates of the polygon's center (tuple of floats; optional).
        radius: The radius of the polygon (float; optional). If not specified, defaults to 1.0.

    Returns:
        A list of vertex coordinates for the generated polygon (list of tuples).
    """

    # Set the radius of the shape to 1 if not provided
    if radius is None:
        radius = 1.0

    # Set the centre of the shape to (0.0, 0.0) if not provided
    if centre is None:
        centre = (0.0, 0.0)

    # Calculate the angle between each side
    angle = 2 * math.pi / num_sides

    # Initialize a list to store the coordinates of each vertex
    vertices = []

    # Generate random angles for each vertex
    for i in range(num_sides):
        # Generate random angle for the current vertex
        rand_angle = i * angle + random.uniform(0, angle)

        # Calculate the x and y coordinates of the vertex
        x = radius * math.cos(rand_angle) + centre[0]
        y = radius * math.sin(rand_angle) + centre[1]

        vertices.append((x, y))

    # Return the list of vertex coordinates
    return vertices


def cv2_filled_polygons(img: npt.NDArray, pts: Sequence, color: Sequence[float],
                        thickness: Optional[int] = 1) -> npt.NDArray:
    """
    Draws filled polygons on an image using OpenCV.

    Args:
        img: The image to draw the polygons on (np.ndarray).
        pts: The coordinates of the polygon vertices (list of tuples).
        color: The color of the polygons (list of floats).
        thickness: The thickness of the polygons (int; optional). If negative, filled polygons are
         drawn.

    Returns:
        The modified image with the drawn polygons (np.ndarray).
    """

    # Check the image type
    if not isinstance(img, np.ndarray):
        raise TypeError('img must be of type np.ndarray.')

    # Convert the color to float
    color = np.array(color, dtype=float)

    # Draw the polygons using OpenCV
    img = cv2.polylines(img, pts=pts, color=color, thickness=abs(thickness), isClosed=True)

    # Fill the polygons if the thickness is negative
    if thickness < 0:
        img = cv2.fillPoly(img, pts=pts, color=color)

    return img


def fg_shape_mask(img_size: int) -> npt.NDArray[bool]:
    """
    Generates a foreground mask with a randomly positioned geometric shape.

    Args:
        img_size: The size of the image (int).

    Returns:
        The generated foreground mask (np.ndarray[bool]).
    """

    # Initialise the image.
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # Generate random number of sides of the polygon.
    num_sides = np.random.randint(3, 16)
    # Generate the polygon shape
    shape_vertices = generate_random_polygon(num_sides)

    # Randomly determine the shape diameter and centre shift
    shape_diameter = np.random.randint(int(img_size * 0.10), int(img_size * 0.90))
    centre_shift = np.random.randint(0, img_size - shape_diameter)

    # Adjust the shape vertices based on the chosen parameters
    pts = (np.array(shape_vertices) + 1) / 2
    pts *= shape_diameter
    pts += centre_shift
    pts = pts.astype(int)

    # Create the foreground mask with the shape
    img = cv2_filled_polygons(img, pts=[pts], color=[255], thickness=-1)

    return np.equal(img, 255)


class ShapeAppearanceDataset(TorchDataset):
    """
    A dataset of geometrical shapes whose appearance properties can be altered.

    Args:
        num_samples: The total number of samples to generate.
        num_images: The number of foreground-background pairs to generate for each sample.
        img_size: The size of the image (int).
        background: The background type (str or callable).
        merge_fg_bg: A function to merge the foreground and background images (callable). This
         function should accept two positional arguments (foreground and background images),
         This function should return the merged images and corresponding ground-truth(s).
        unique_fg_shape: Whether to use unique foreground shapes for each sample
         (bool; optional).
        unique_bg: Whether to use unique background images for each sample (bool; optional).
        transform: An optional transformation to be applied to the generated images
         (callable; optional).
    """

    def __init__(
            self, num_samples: int, num_images: int, img_size: int, background: Any,
            merge_fg_bg: Callable[[List[npt.NDArray[bool]], List[npt.NDArray]], Tuple],
            unique_fg_shape: Optional[bool] = True, unique_bg: Optional[bool] = True,
            transform: Optional[Callable] = None
    ) -> None:
        super(ShapeAppearanceDataset, self).__init__()
        self.num_samples = num_samples
        self.num_images = num_images
        self.img_size = img_size
        assert callable(merge_fg_bg)
        self.merge_fg_bg = merge_fg_bg
        self.bg = background
        self.unique_fg_shape = unique_fg_shape
        self.unique_bg = unique_bg
        self.transform = transform

    def __len__(self) -> int:
        """
        Determines the total number of data samples.

        Returns:
            The total number of samples (int).
        """
        return self.num_samples

    def make_fg_masks(self) -> List[npt.NDArray[bool]]:
        """
        Generates foreground masks with randomly positioned shapes.

        Returns:
            A list of foreground masks (List[npt.NDArray[bool]]).
        """
        if self.unique_fg_shape:
            fg_mask = fg_shape_mask(self.img_size)
            return [fg_mask.copy() for _ in range(self.num_images)]
        else:
            return [fg_shape_mask(self.img_size) for _ in range(self.num_images)]

    def make_bg_images(self) -> List[npt.NDArray]:
        """
        Generates background images.

        Returns:
            A list of background images (List[npt.NDArray]).
        """
        if self.unique_bg:
            bg_img = dataset_utils.background_img(self.bg, self.img_size)
            return [bg_img.copy() for _ in range(self.num_images)]
        else:
            return [dataset_utils.background_img(
                self.bg, self.img_size) for _ in range(self.num_images)]

    def __getitem__(self, _idx: int) -> (List[Union[torch.Tensor, npt.NDArray]], Any):
        """
        Retrieve a data sample.

        Args:
            _idx: The index of the sample (int).

        Returns:
            A tuple containing the foreground masks, background images, and ground truth (Tuple).
        """
        fgs = self.make_fg_masks()  # foregrounds
        bgs = self.make_bg_images()  # backgrounds
        images, gt = self.merge_fg_bg(fgs, bgs)
        if self.transform:
            images = [self.transform(img) for img in images]
        return *images, gt
