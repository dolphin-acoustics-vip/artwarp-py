"""
Visualization module for ARTwarp results.

This module provides professional plotting capabilities for visualizing:
- Training results and convergence
- Category distributions
- Reference contours (category prototypes)
- Individual contour assignments
- Iteration history
- Discovery curve (cumulative categories vs sample order)
- DTW alignment, ART schematic, warp constraint (algorithm)
- Category similarity matrix, embedding, dendrogram (diagnostics)
- Resampling before/after, contour length/tempres distribution (data)
- Vigilance sweep, run stability (parameter studies)
- Paper-ready figure, confusion matrix, label vs category (reporting)

All plotting functions use matplotlib and follow scientific visualization best practices.

@author: Pedro Gronda Garrigues
"""

from artwarp.visualization.plotting import (
    create_paper_figure,
    create_results_report,
    plot_art_schematic,
    plot_category_distribution,
    plot_category_embedding,
    plot_category_similarity_matrix,
    plot_category_dendrogram,
    plot_confusion_matrix,
    plot_contour_length_distribution,
    plot_contours_by_category,
    plot_convergence_history,
    plot_discovery_curve,
    plot_dtw_alignment,
    plot_label_vs_category,
    plot_match_distribution,
    plot_per_category_match_quality,
    plot_reference_contours,
    plot_resampling_before_after,
    plot_run_stability,
    plot_training_summary,
    plot_vigilance_sweep,
    plot_warp_constraint,
)

__all__ = [
    "create_paper_figure",
    "create_results_report",
    "plot_art_schematic",
    "plot_category_distribution",
    "plot_category_embedding",
    "plot_category_similarity_matrix",
    "plot_category_dendrogram",
    "plot_confusion_matrix",
    "plot_contour_length_distribution",
    "plot_contours_by_category",
    "plot_convergence_history",
    "plot_discovery_curve",
    "plot_dtw_alignment",
    "plot_label_vs_category",
    "plot_match_distribution",
    "plot_per_category_match_quality",
    "plot_reference_contours",
    "plot_resampling_before_after",
    "plot_run_stability",
    "plot_training_summary",
    "plot_vigilance_sweep",
    "plot_warp_constraint",
]
