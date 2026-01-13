"""
ARTwarp-py: High-performance bioacoustic signal categorization.

This package provides a complete Python implementation of the ARTwarp algorithm,
which combines Dynamic Time Warping (DTW) for contour similarity measurement
with an Adaptive Resonance Theory (ART) neural network for unsupervised clustering.

Main Components:
    - core.dtw: Dynamic Time Warping implementation
    - core.art: ART neural network components
    - core.network: Main ARTwarp algorithm
    - io.loaders: Data loading utilities
    - visualization: Professional plotting functions
    - utils: Helper functions and validation

Example:
    >>> from artwarp import ARTwarp, load_contours
    >>> contours = load_contours('path/to/directory')
    >>> network = ARTwarp(vigilance=85.0, learning_rate=0.1)
    >>> results = network.fit(contours)
    >>> 
    >>> # Visualize results
    >>> from artwarp.visualization import plot_training_summary
    >>> plot_training_summary(results)

@author: Pedro Gronda Garrigues
"""

from artwarp.core.network import ARTwarp
from artwarp.io.loaders import load_contours

# visualization kept separate -> no matplotlib required for core
# from artwarp.visualization import plot_training_summary

__version__ = "2.0.3"
__all__ = ["ARTwarp", "load_contours"]
