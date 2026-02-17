"""Input validation utilities.

@author: Pedro Gronda Garrigues
"""

from typing import List

import numpy as np
from numpy.typing import NDArray


def validate_contour(contour: NDArray[np.float64]) -> None:
    """
    Validate that a contour array is properly formatted.

    Args:
        contour: Frequency contour array

    Raises:
        ValueError: If contour is invalid
    """
    if not isinstance(contour, np.ndarray):
        raise ValueError(f"Contour must be a numpy array, got {type(contour)}")

    if contour.ndim != 1:
        raise ValueError(f"Contour must be 1-dimensional, got shape {contour.shape}")

    if len(contour) == 0:
        raise ValueError("Contour cannot be empty")

    if not np.issubdtype(contour.dtype, np.number):
        raise ValueError(f"Contour must contain numeric values, got dtype {contour.dtype}")

    if np.any(np.isnan(contour)):
        raise ValueError("Contour contains NaN values")

    if np.any(np.isinf(contour)):
        raise ValueError("Contour contains infinite values")

    if np.any(contour <= 0):
        raise ValueError("Contour must contain positive frequency values")


def validate_contours(contours: List[NDArray[np.float64]]) -> None:
    """
    Validate a list of contour arrays.

    Args:
        contours: List of frequency contour arrays

    Raises:
        ValueError: If any contour is invalid
    """
    if not isinstance(contours, list):
        raise ValueError(f"Contours must be a list, got {type(contours)}")

    if len(contours) == 0:
        raise ValueError("Contours list cannot be empty")

    for i, contour in enumerate(contours):
        try:
            validate_contour(contour)
        except ValueError as e:
            raise ValueError(f"Invalid contour at index {i}: {str(e)}")


def validate_parameters(
    vigilance: float,
    learning_rate: float,
    bias: float,
    max_categories: int,
    max_iterations: int,
    warp_factor_level: int,
) -> None:
    """
    Validate ARTwarp parameters.

    Args:
        vigilance: Vigilance threshold
        learning_rate: Learning rate
        bias: Activation bias
        max_categories: Maximum number of categories
        max_iterations: Maximum number of iterations
        warp_factor_level: Maximum warping factor

    Raises:
        ValueError: If any parameter is invalid
    """
    if not 1 <= vigilance <= 99:
        raise ValueError(f"Vigilance must be in range [1, 99], got {vigilance}")

    if not 0 < learning_rate <= 1:
        raise ValueError(f"Learning rate must be in range (0, 1], got {learning_rate}")

    if not 0 <= bias <= 1:
        raise ValueError(f"Bias must be in range [0, 1], got {bias}")

    if not isinstance(max_categories, int) or max_categories < 1:
        raise ValueError(f"Max categories must be a positive integer, got {max_categories}")

    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError(f"Max iterations must be a positive integer, got {max_iterations}")

    if not isinstance(warp_factor_level, int) or warp_factor_level <= 1:
        raise ValueError(f"Warp factor level must be an integer > 1, got {warp_factor_level}")
