"""
Collection of gratings datasets common is psychophysical studies.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Callable, Sequence, Union

import torch
from torch.utils.data import Dataset as TorchDataset

from . import dataset_utils

__all__ = [
    'GratingsDataset'
]


def sinusoid_grating(img_size: int, amplitude: float, theta: float, phase: float,
                     spatial_frequency: int) -> npt.NDArray[float]:
    """Generates sinusoidal grating stimuli.

    Args:
        img_size: The desired size of the grating image.
        amplitude: The amplitude of the sinusoidal modulation.
        theta: The orientation of the grating (float in radians).
        phase: The phase offset of the sinusoidal modulation.
        spatial_frequency: The spatial frequency of the grating (int cycles per image).

    Returns:
        The generated sinusoidal grating stimuli.
    """

    # Generate the grid coordinates
    radius = img_size // 2
    [x, y] = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))

    # Compute the frequency and phase parameters
    omega = [np.cos(theta), np.sin(theta)]
    lambda_wave = (img_size * 0.5) / (np.pi * spatial_frequency)

    # Calculate the sinusoidal modulation
    stimuli = amplitude * np.cos((omega[0] * x + omega[1] * y) / lambda_wave + phase)

    # If the target size is even, the generated stimuli is 1 pixel larger.
    if np.mod(img_size, 2) == 0:
        stimuli = stimuli[:-1, :-1]

    return stimuli


def gaussian_img(img_size: int, sigma: float) -> npt.NDArray[float]:
    """Generates a Gaussian-filtered image.

    Args:
        img_size: The desired size of the Gaussian image (int).
        sigma: The standard deviation of the Gaussian filter (float).

    Returns:
        The generated Gaussian-filtered image (np.ndarray).
    """

    # Generate the grid coordinates
    radius = img_size // 2
    [x, y] = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))

    # Compute the Gaussian filter
    gauss2d = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

    # Make the size odd
    if np.mod(img_size, 2) == 0:
        gauss2d = gauss2d[:-1, :-1]

    # Normalize the intensity values to [0, 1]
    gauss2d = gauss2d / np.max(gauss2d)

    # Return the Gaussian-filtered image
    return gauss2d


class GratingsDataset(TorchDataset):
    """
    A dataset class for generating and storing sinusoidal grating stimuli.

    Args:
        img_size: The desired size of the grating images (int).
        spatial_frequencies: A list of spatial frequencies for the gratings (optional).
        thetas: A list of orientations for the gratings (optional).
        gaussian_sigma: The standard deviation of the Gaussian filter (optional).
        transform: A transformation to be applied to the stimuli (optional).
    """

    def __init__(self, img_size: int, spatial_frequencies: Optional[Sequence[int]] = None,
                 thetas: Optional[Sequence[float]] = None, gaussian_sigma: Optional[float] = None,
                 transform: Optional[Callable] = None) -> None:
        super(GratingsDataset, self).__init__()
        self.img_size = img_size
        self.transform = transform
        self.sfs = [
            i for i in range(1, img_size // 2 + 1) if img_size % i == 0
        ] if spatial_frequencies is None else spatial_frequencies
        self.thetas = np.arange(0, np.pi + 1e-3, np.pi / 12) if thetas is None else thetas
        self.gaussian_sigma = gaussian_sigma

    def __len__(self) -> int:
        return len(self.thetas) * len(self.sfs)

    def make_grating(self, idx: int, amplitude: float, channels=3) -> npt.NDArray[float]:
        """
        Constructs a sinusoidal grating image.

        Args:
            idx: The index of the grating to be generated.
            amplitude: The amplitude of the sinusoidal modulation (float).
            channels: The number of output channels (int; optional).

        Returns:
            The generated sinusoidal grating image (np.ndarray).
        """

        theta_ind, sf_ind = np.unravel_index(idx, (len(self.thetas), len(self.sfs)))
        theta = self.thetas[theta_ind]
        sf = self.sfs[sf_ind]
        phase = 0
        stimuli = sinusoid_grating(self.img_size, amplitude, theta, phase, sf)

        # Apply Gaussian filtering if specified
        if self.gaussian_sigma is not None:
            gauss_img = gaussian_img(self.img_size, self.gaussian_sigma)
            stimuli *= gauss_img

        # Normalise the image intensity to [0, 1]
        stimuli = (stimuli + 1) / 2

        # Repeat the image to multiple channels (if requested)
        stimuli = dataset_utils.repeat_channels(stimuli, channels=channels)
        return stimuli

    def __getitem__(self, idx: int) -> Union[torch.Tensor, npt.NDArray]:
        """
        Retrieves a grating image from the dataset.

        Args:
            idx: The index of the grating to be retrieved.

        Returns:
            The retrieved grating image (torch.Tensor or np.ndarray) after applying the specified
             transformation.
        """
        stimuli = self.make_grating(idx, 1.0)
        if self.transform:
            stimuli = self.transform(stimuli)
        return stimuli
