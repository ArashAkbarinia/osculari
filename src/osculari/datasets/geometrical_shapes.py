"""
A simple generic dataset of geometrical shapes in foreground.
"""

import numpy as np
import math
import random
from typing import Optional, List, Tuple, Sequence, Callable

import cv2
from torch.utils.data import Dataset as TorchDataset


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


def cv2_filled_polygons(img: np.array, pts: Sequence, color: Sequence[float],
                        thickness: Optional[int] = 1) -> np.array:
    """Drawing a filled polygon."""
    img = cv2.polylines(img, pts=pts, color=color, thickness=abs(thickness), isClosed=True)
    if thickness < 0:
        img = cv2.fillPoly(img, pts=pts, color=color)
    return img


def fg_shape_mask(img_size: int) -> np.array:
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
    return img == 255


class ShapeDataset(TorchDataset):
    """PyTorch dataset of geometrical shapes."""

    def __init__(self, num_samples: int, num_imgs: int, img_size: int,
                 transform: Optional[Callable] = None) -> None:
        super(ShapeDataset, self).__init__()
        self.num_samples = num_samples
        self.num_imgs = num_imgs
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _idx):
        # our routine doesn't need the idx, which is the sample number
        fgs = [fg_shape_mask(self.img_size) for _ in range(self.num_imgs)]

        if self.transform:
            fgs = [self.transform(fg) for fg in fgs]
        return fgs
