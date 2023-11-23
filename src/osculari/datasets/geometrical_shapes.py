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


def generate_random_polygon(num_sides: int) -> List[Tuple[float, float]]:
    """Returning the vertices of a random polygon."""
    # Set the radius of the shape to 1
    radius = 1.0
    # Calculate the angle between each side
    angle = 2 * math.pi / num_sides
    # Initialize a list to store the coordinates of each vertex
    vertices = []
    # Generate random angles for each vertex
    for i in range(num_sides):
        rand_angle = i * angle + random.uniform(0, angle)
        # Calculate the x and y coordinates of the vertex
        x = radius * math.cos(rand_angle)
        y = radius * math.sin(rand_angle)
        vertices.append((x, y))
    # Return the list of vertex coordinates
    return vertices


def cv2_filled_polygons(img: npt.NDArray, pts: Sequence, color: Sequence[float],
                        thickness: Optional[int] = 1) -> npt.NDArray:
    """Drawing a filled polygon."""
    img = cv2.polylines(img, pts=pts, color=color, thickness=abs(thickness), isClosed=True)
    if thickness < 0:
        img = cv2.fillPoly(img, pts=pts, color=color)
    return img


def fg_shape_mask(img_size: int) -> npt.NDArray[bool]:
    """Generating a geometrical shape in the foreground."""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    num_sides = np.random.randint(3, 16)
    shape_vertices = generate_random_polygon(num_sides)
    shape_diameter = np.random.randint(int(img_size * 0.10), int(img_size * 0.90))
    centre_shift = np.random.randint(0, img_size - shape_diameter)

    # making the points between 0 and 1
    pts = (np.array(shape_vertices) + 1) / 2
    pts *= shape_diameter
    pts += centre_shift
    pts = pts.astype(int)
    img = cv2_filled_polygons(img, pts=[pts], color=[255], thickness=-1)
    return np.equal(img, 255)


class ShapeAppearanceDataset(TorchDataset):
    """A dataset of geometrical shapes whose appearance properties can be altered."""

    def __init__(self, num_samples: int, num_images: int, img_size: int, background: Any,
                 merge_fg_bg: Callable,
                 unique_fg_shape: Optional[bool] = True, unique_bg: Optional[bool] = True,
                 transform: Optional[Callable] = None) -> None:
        super(ShapeAppearanceDataset, self).__init__()
        self.num_samples = num_samples
        self.num_images = num_images
        self.img_size = img_size
        self.merge_fg_bg = merge_fg_bg
        self.bg = background
        self.unique_fg_shape = unique_fg_shape
        self.unique_bg = unique_bg
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def make_fg_masks(self) -> List[npt.NDArray[bool]]:
        """Generating the foreground images."""
        if self.unique_fg_shape:
            fg_mask = fg_shape_mask(self.img_size)
            return [fg_mask.copy() for _ in range(self.num_images)]
        else:
            return [fg_shape_mask(self.img_size) for _ in range(self.num_images)]

    def make_bg_images(self) -> List[npt.NDArray]:
        """Generating the background images."""
        if self.unique_bg:
            bg_img = dataset_utils.background_img(self.bg, self.img_size)
            return [bg_img.copy() for _ in range(self.num_images)]
        else:
            return [dataset_utils.background_img(
                self.bg, self.img_size) for _ in range(self.num_images)]

    def __getitem__(self, _idx: int) -> (List[Union[torch.Tensor, npt.NDArray]], Any):
        # our routine doesn't need the idx, which is the sample number
        fgs = self.make_fg_masks()  # foregrounds
        bgs = self.make_bg_images()  # backgrounds
        images, gt = self.merge_fg_bg(fgs, bgs)
        if self.transform:
            images = [self.transform(img) for img in images]
        return *images, gt
