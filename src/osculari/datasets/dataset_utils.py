"""
Set of utility functions in generating datasets.
"""

import numpy as np


def michelson_contrast(img: np.array, contrast: float) -> np.array:
    assert 0 <= contrast <= 1
    return (1 - contrast) / 2.0 + np.multiply(img, contrast)
