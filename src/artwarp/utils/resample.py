"""
Resample contours to a new temporal resolution (MATLAB resample option).

Mirrors ARTwarp_Run_Categorisation.m when resample == 1:
  interp1(1:length(contour), contour, 1:sampleInterval/tempres:length(contour))

@author: Pedro Gronda Garrigues
"""

from typing import List, Union
import numpy as np
from numpy.typing import NDArray


def resample_contours(
    contours: List[NDArray[np.float64]],
    tempres: Union[float, List[float]],
    sample_interval_sec: float
) -> List[NDArray[np.float64]]:
    """
    Resample each contour to a new sampling interval (seconds per point).

    Matches MATLAB ARTwarp resample option: each contour is interpolated so that
    the effective time step between points is sample_interval_sec. Useful when
    contours have different temporal resolutions and you want a uniform sampling
    before training.

    Args:
        contours: List of frequency contour arrays (points per contour).
        tempres: Current temporal resolution: seconds per point. Either a single
            float (same for all contours) or a list of length len(contours).
        sample_interval_sec: Desired interval in seconds between consecutive
            points in the output (e.g. 0.01 for 10 ms).

    Returns:
        List of resampled contour arrays (same length as contours).

    Example:
        >>> contours = [np.array([100., 200., 300.]), np.array([150., 250.])]
        >>> tempres = 0.01  # 10 ms per point
        >>> resampled = resample_contours(contours, tempres, 0.02)  # 20 ms output
    """
    if isinstance(tempres, (int, float)):
        tempres_list = [float(tempres)] * len(contours)
    else:
        tempres_list = list(tempres)
    if len(tempres_list) != len(contours):
        raise ValueError(
            f"tempres length ({len(tempres_list)}) must match number of contours ({len(contours)})"
        )
    if sample_interval_sec <= 0:
        raise ValueError("sample_interval_sec must be positive")

    result: List[NDArray[np.float64]] = []
    for c, tr in zip(contours, tempres_list):
        n = len(c)
        if n == 0:
            result.append(np.array([], dtype=np.float64))
            continue
        if n == 1:
            result.append(np.array([c[0]], dtype=np.float64))
            continue
        # MATLAB: 1:(sampleInterval/tempres):length(contour) => new indices
        step = sample_interval_sec / tr
        new_length = max(1, int(round(1 + (n - 1) / step)))
        old_x = np.arange(n, dtype=np.float64)
        new_x = np.linspace(0, n - 1, new_length)
        resampled = np.interp(new_x, old_x, c)
        result.append(resampled.astype(np.float64))
    return result
