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
    """Generate Sinusoid gratings."""
    radius = img_size // 2
    [x, y] = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
    omega = [np.cos(theta), np.sin(theta)]
    lambda_wave = (img_size * 0.5) / (np.pi * spatial_frequency)
    stimuli = amplitude * np.cos((omega[0] * x + omega[1] * y) / lambda_wave + phase)
    # if the target size is even, the generated stimuli is 1 pixel larger.
    if np.mod(img_size, 2) == 0:
        stimuli = stimuli[:-1, :-1]
    return stimuli


def gaussian_img(img_size: int, sigma: float) -> npt.NDArray[float]:
    radius = img_size // 2
    [x, y] = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
    gauss2d = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))
    if np.mod(img_size, 2) == 0:
        gauss2d = gauss2d[:-1, :-1]
    gauss2d = gauss2d / np.max(gauss2d)
    return gauss2d


class GratingsDataset(TorchDataset):

    def __init__(self, img_size: int, spatial_frequencies: Optional[Sequence[int]] = None,
                 thetas: Optional[Sequence[int]] = None, gaussian_sigma: Optional[float] = None,
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
        theta_ind, sf_ind = np.unravel_index(idx, (len(self.thetas), len(self.sfs)))
        theta = self.thetas[theta_ind]
        sf = self.sfs[sf_ind]
        phase = 0
        stimuli = sinusoid_grating(self.img_size, amplitude, theta, phase, sf)
        # multiply by a Gaussian
        if self.gaussian_sigma is not None:
            gauss_img = gaussian_img(self.img_size, self.gaussian_sigma)
            stimuli *= gauss_img
        # bringing the image in the range of 0-1
        stimuli = (stimuli + 1) / 2
        # converting it to N channels
        stimuli = dataset_utils.repeat_channels(stimuli, channels=channels)
        return stimuli

    def __getitem__(self, idx: int) -> Union[torch.Tensor, npt.NDArray]:
        stimuli = self.make_grating(idx, 1.0)
        if self.transform:
            stimuli = self.transform(stimuli)
        return stimuli
