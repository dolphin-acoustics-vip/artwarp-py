"""
Visualization module for ARTwarp results.

This module provides professional plotting capabilities for visualizing:
- Training results and convergence
- Category distributions
- Reference contours (category prototypes)
- Individual contour assignments
- Iteration history

All plotting functions use matplotlib and follow scientific visualization best practices.

@author: Pedro Gronda Garrigues
"""

from artwarp.visualization.plotting import (
    plot_training_summary,
    plot_reference_contours,
    plot_category_distribution,
    plot_convergence_history,
    plot_contours_by_category,
    plot_match_distribution,
    create_results_report,
)

__all__ = [
    "plot_training_summary",
    "plot_reference_contours",
    "plot_category_distribution",
    "plot_convergence_history",
    "plot_contours_by_category",
    "plot_match_distribution",
    "create_results_report",
]
