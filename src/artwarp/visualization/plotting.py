"""
Professional visualization functions for ARTwarp results.

This module provides comprehensive plotting capabilities following scientific
visualization best practices. All functions support customization and export
to various formats.

@author: Pedro Gronda Garrigues

*Note: This plotting module was largely auto-generated and may have some issues.
Please use with caution.*
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from numpy.typing import NDArray

from artwarp.core.network import TrainingResults


def plot_training_summary(
    results: TrainingResults,
    contour_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Create a comprehensive training summary visualization.

    This creates a multi-panel figure showing:
    - Reference contours for each category
    - Category size distribution
    - Match value distribution
    - Convergence history

    Args:
        results: TrainingResults object from network training
        contour_names: Optional list of contour names for labels
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_training_summary(results)
        >>> plt.show()
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # Main title
    fig.suptitle(
        f"ARTwarp Training Summary\n"
        f"{results.num_categories} Categories | "
        f"{results.num_iterations} Iterations | "
        f'{"Converged" if results.converged else "Not Converged"}',
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Reference contours (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_reference_contours_axes(ax1, results.weight_matrix)

    # Plot 2: Category distribution (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    _plot_category_distribution_axes(ax2, results)

    # Plot 3: Match distribution (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_match_distribution_axes(ax3, results)

    # Plot 4: Convergence history (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_convergence_history_axes(ax4, results)

    # Plot 5: Category assignments table (middle right + bottom right)
    ax5 = fig.add_subplot(gs[1:, 2])
    _plot_category_table_axes(ax5, results, contour_names)

    # Plot 6: Match values by category (bottom left + center)
    ax6 = fig.add_subplot(gs[2, :2])
    _plot_match_by_category_axes(ax6, results)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_reference_contours(
    weight_matrix: NDArray[np.float64],
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show_grid: bool = True,
) -> Figure:
    """
    Plot all reference contours (category prototypes).

    Creates a grid of subplots showing the frequency contour for each category.
    Each contour represents the typical shape of that category.

    Args:
        weight_matrix: Weight matrix from trained network (max_features, num_categories)
        figsize: Figure size in inches (width, height). If None, auto-calculated
        save_path: Optional path to save figure
        dpi: Resolution for saved figure
        show_grid: Whether to show grid lines

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_reference_contours(results.weight_matrix)
        >>> plt.show()
    """
    num_categories = weight_matrix.shape[1]

    if num_categories == 0:
        raise ValueError("No categories to plot")

    # grid dimensions
    ncols = min(4, num_categories)
    nrows = (num_categories + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle("Reference Contours (Category Prototypes)", fontsize=14, fontweight="bold")

    for cat_idx in range(num_categories):
        row = cat_idx // ncols
        col = cat_idx % ncols
        ax = axes[row, col]

        # contour without NaN padding
        contour = weight_matrix[:, cat_idx]
        valid_mask = ~np.isnan(contour)
        valid_contour = contour[valid_mask]

        if len(valid_contour) > 0:
            time_points = np.arange(len(valid_contour))
            ax.plot(time_points, valid_contour, "b-", linewidth=2, alpha=0.7)
            ax.fill_between(time_points, valid_contour, alpha=0.3)

            ax.set_xlabel("Time Point", fontsize=10)
            ax.set_ylabel("Frequency (Hz)", fontsize=10)
            ax.set_title(f"Category {cat_idx}", fontsize=11, fontweight="bold")

            if show_grid:
                ax.grid(True, alpha=0.3, linestyle="--")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.text(0.5, 0.5, "Empty Category", ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

    # hide unused subplots
    for idx in range(num_categories, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_category_distribution(
    results: TrainingResults,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Plot the distribution of samples across categories.

    Creates a bar chart showing how many samples were assigned to each category,
    plus uncategorized samples if any.

    Args:
        results: TrainingResults object from network training
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    category_sizes = results.get_category_sizes()
    uncategorized = results.get_uncategorized_count()

    # prep data
    categories = sorted(category_sizes.keys())
    sizes = [category_sizes[cat] for cat in categories]
    bar_labels = [str(c) for c in categories]
    if uncategorized > 0:
        bar_labels.append("Uncategorized")
        sizes.append(uncategorized)

    # bar chart
    cmap = colormaps.get_cmap("Set3")
    colors = cmap(np.linspace(0, 1, len(bar_labels)))
    bars = ax.bar(range(len(bar_labels)), sizes, color=colors, alpha=0.7, edgecolor="black")

    # value labels on bars
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(size)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # labels + title
    n_cats = len(bar_labels)
    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title("Category Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(range(n_cats))
    # many categories => smaller font, vertical labels, every k-th to avoid overlap
    if n_cats > 24:
        step = max(1, n_cats // 25)
        labels = [bar_labels[i] if i % step == 0 else "" for i in range(n_cats)]
        ax.set_xticklabels(labels, fontsize=max(4, 10 - n_cats // 20), rotation=90, ha="center")
    else:
        ax.set_xticklabels(bar_labels, rotation=45, ha="right")

    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_convergence_history(
    results: TrainingResults,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Plot the convergence history showing reclassifications per iteration.

    Creates a line plot showing how many samples were reclassified in each
    iteration, indicating when the network converged.

    Args:
        results: TrainingResults object from network training
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if len(results.iteration_history) == 0:
        ax.text(0.5, 0.5, "No iteration history available", ha="center", va="center", fontsize=14)
        return fig

    iterations, reclassifications = zip(*results.iteration_history)

    ax.plot(
        iterations,
        reclassifications,
        "o-",
        linewidth=2,
        markersize=6,
        color="#2E86AB",
        label="Reclassifications",
    )

    # mark convergence point if converged
    if results.converged:
        conv_iter = results.num_iterations
        ax.axvline(
            conv_iter,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Converged at iteration {conv_iter}",
        )

    ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Reclassifications", fontsize=12, fontweight="bold")
    ax.set_title("Training Convergence History", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_contours_by_category(
    contours: List[NDArray[np.float64]],
    categories: NDArray[np.float64],
    category_id: int,
    contour_names: Optional[List[str]] = None,
    reference_contour: Optional[NDArray[np.float64]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 300,
    max_contours: int = 20,
) -> Figure:
    """
    Plot all contours assigned to a specific category.

    Shows individual contours overlaid with the category reference contour
    for visual comparison.

    Args:
        contours: List of frequency contour arrays
        categories: Category assignments for each contour
        category_id: Category to plot
        contour_names: Optional list of contour names
        reference_contour: Optional reference contour to overlay
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure
        max_contours: Maximum number of contours to plot (for readability)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # contours in this category
    mask = categories == category_id
    category_indices = np.where(mask)[0]

    if len(category_indices) == 0:
        ax.text(
            0.5,
            0.5,
            f"No contours in category {category_id}",
            ha="center",
            va="center",
            fontsize=14,
        )
        return fig

    # limit contours for readability
    if len(category_indices) > max_contours:
        category_indices = category_indices[:max_contours]
        title_suffix = f" (showing {max_contours} of {len(category_indices)})"
    else:
        title_suffix = ""

    # plot individual contours
    cmap = colormaps.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.7, len(category_indices)))

    for idx, color in zip(category_indices, colors):
        contour = contours[idx]
        time_points = np.arange(len(contour))

        label = contour_names[idx] if contour_names else f"Contour {idx}"
        ax.plot(time_points, contour, alpha=0.5, linewidth=1.5, color=color, label=label)

    # ref contour if provided
    if reference_contour is not None:
        time_points_ref = np.arange(len(reference_contour))
        ax.plot(
            time_points_ref,
            reference_contour,
            "r-",
            linewidth=3,
            alpha=0.8,
            label="Reference Contour",
        )

    ax.set_xlabel("Time Point", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Contours in Category {category_id}{title_suffix}", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # legend for plotted contours (we already cap at max_contours, so size is bounded)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_match_distribution(
    results: TrainingResults,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
    bins: int = 30,
) -> Figure:
    """
    Plot the distribution of match values.

    Creates a histogram showing the distribution of match scores across
    all categorized samples.

    Args:
        results: TrainingResults object from network training
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure
        dpi: Resolution for saved figure
        bins: Number of histogram bins

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # get match values for categorized samples only (finite only to avoid empty-slice warnings)
    categorized_mask = ~np.isnan(results.categories)
    matches = results.matches[categorized_mask]
    matches = matches[np.isfinite(matches)]

    if len(matches) == 0:
        ax.text(0.5, 0.5, "No categorized samples", ha="center", va="center", fontsize=14)
        return fig

    # Create histogram
    n, bins_edges, patches = ax.hist(
        matches, bins=bins, alpha=0.7, color="#2E86AB", edgecolor="black"
    )

    # mean line
    mean_match = float(np.mean(matches))
    ax.axvline(
        mean_match, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_match:.1f}%"
    )

    # median line
    median_match = float(np.median(matches))
    ax.axvline(
        median_match,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_match:.1f}%",
    )

    ax.set_xlabel("Match Value (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("Distribution of Match Values", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def create_results_report(
    results: TrainingResults,
    contours: List[NDArray[np.float64]],
    contour_names: Optional[List[str]] = None,
    output_dir: str = "./artwarp_report",
    dpi: int = 300,
) -> Dict[str, str]:
    """
    Create a comprehensive visualization report with multiple figures.

    Generates and saves multiple visualization figures to a directory,
    providing a complete visual analysis of training results.

    Args:
        results: TrainingResults object from network training
        contours: List of frequency contour arrays
        contour_names: Optional list of contour names
        output_dir: Directory to save all figures
        dpi: Resolution for saved figures

    Returns:
        Dictionary mapping figure names to file paths

    Example:
        >>> files = create_results_report(results, contours, names)
        >>> print(f"Created {len(files)} figures in {output_dir}")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Training summary
    print("Creating training summary...")
    fig1 = plot_training_summary(results, contour_names)
    path1 = output_path / "training_summary.png"
    fig1.savefig(path1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    saved_files["training_summary"] = str(path1)

    # 2. Reference contours
    print("Creating reference contours plot...")
    fig2 = plot_reference_contours(results.weight_matrix)
    path2 = output_path / "reference_contours.png"
    fig2.savefig(path2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    saved_files["reference_contours"] = str(path2)

    # 3. Category distribution
    print("Creating category distribution...")
    fig3 = plot_category_distribution(results)
    path3 = output_path / "category_distribution.png"
    fig3.savefig(path3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    saved_files["category_distribution"] = str(path3)

    # 4. Convergence history
    print("Creating convergence history...")
    fig4 = plot_convergence_history(results)
    path4 = output_path / "convergence_history.png"
    fig4.savefig(path4, dpi=dpi, bbox_inches="tight")
    plt.close(fig4)
    saved_files["convergence_history"] = str(path4)

    # 5. Match distribution
    print("Creating match distribution...")
    fig5 = plot_match_distribution(results)
    path5 = output_path / "match_distribution.png"
    fig5.savefig(path5, dpi=dpi, bbox_inches="tight")
    plt.close(fig5)
    saved_files["match_distribution"] = str(path5)

    # 6. Contours by category (for each category)
    print("Creating per-category plots...")
    for cat_id in sorted(results.get_category_sizes().keys()):
        # ref contour for this category
        ref_contour_full = results.weight_matrix[:, cat_id]
        valid_mask = ~np.isnan(ref_contour_full)
        ref_contour = ref_contour_full[valid_mask]

        fig = plot_contours_by_category(
            contours, results.categories, cat_id, contour_names, ref_contour
        )
        path = output_path / f"category_{cat_id}_contours.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files[f"category_{cat_id}"] = str(path)

    # print(f"\nReport generated: {len(saved_files)} figures saved to {output_dir}")

    return saved_files


# Helper functions for training summary subplots


def _plot_reference_contours_axes(ax: Axes, weight_matrix: NDArray[np.float64]) -> None:
    """Plot reference contours on given axes."""
    num_categories = weight_matrix.shape[1]
    cmap = colormaps.get_cmap("tab10")
    colors = cmap(np.arange(num_categories) % 10)
    max_legend_cats = 25  # avoid huge legend

    for cat_idx in range(num_categories):
        contour = weight_matrix[:, cat_idx]
        valid_mask = ~np.isnan(contour)
        valid_contour = contour[valid_mask]
        label = f"Cat {cat_idx}" if num_categories <= max_legend_cats else None
        if len(valid_contour) > 0:
            time_points = np.arange(len(valid_contour))
            ax.plot(
                time_points,
                valid_contour,
                linewidth=2,
                color=colors[cat_idx],
                label=label,
                alpha=0.7,
            )

    ax.set_xlabel("Time Point", fontweight="bold")
    ax.set_ylabel("Frequency (Hz)", fontweight="bold")
    title = (
        f"Reference Contours ({num_categories} categories)"
        if num_categories > max_legend_cats
        else "Reference Contours"
    )
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if num_categories <= max_legend_cats:
        ax.legend(loc="best", fontsize=8, ncol=2)


def _plot_category_distribution_axes(ax: Axes, results: TrainingResults) -> None:
    """Plot category distribution on given axes."""
    sizes = results.get_category_sizes()
    categories = sorted(sizes.keys())
    counts = [sizes[c] for c in categories]
    n_cats = len(categories)

    ax.bar(range(n_cats), counts, alpha=0.7, color="skyblue", edgecolor="black")
    ax.set_xlabel("Category", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("Category Sizes", fontweight="bold")
    ax.set_xticks(range(n_cats))
    # many categories => smaller font, rotated, every k-th
    if n_cats > 24:
        step = max(1, n_cats // 25)
        labels = [str(c) if i % step == 0 else "" for i, c in enumerate(categories)]
        ax.set_xticklabels(labels, fontsize=max(4, 10 - n_cats // 20), rotation=90)
    else:
        ax.set_xticklabels([str(c) for c in categories])
    ax.grid(True, axis="y", alpha=0.3)


def _plot_match_distribution_axes(ax: Axes, results: TrainingResults) -> None:
    """Plot match distribution on given axes."""
    categorized = ~np.isnan(results.categories)
    matches = results.matches[categorized]
    matches = matches[np.isfinite(matches)]

    if len(matches) == 0:
        ax.text(0.5, 0.5, "No match data", ha="center", va="center", fontsize=12)
        ax.set_xlabel("Match (%)", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title("Match Distribution", fontweight="bold")
        return
    ax.hist(matches, bins=20, alpha=0.7, color="lightgreen", edgecolor="black")
    ax.axvline(float(np.mean(matches)), color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Match (%)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Match Distribution", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)


def _plot_convergence_history_axes(ax: Axes, results: TrainingResults) -> None:
    """Plot convergence history on given axes."""
    if len(results.iteration_history) > 0:
        iters, reclassifications = zip(*results.iteration_history)
        ax.plot(iters, reclassifications, "o-", linewidth=2, color="purple")
        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_ylabel("Reclassifications", fontweight="bold")
        ax.set_title("Convergence", fontweight="bold")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No history", ha="center", va="center")


def _plot_category_table_axes(
    ax: Axes, results: TrainingResults, contour_names: Optional[List[str]]
) -> None:
    """Plot category statistics table on given axes."""
    ax.axis("off")

    # summary stats
    stats_data = [
        ["Metric", "Value"],
        ["Total Samples", f"{len(results.categories)}"],
        ["Categories", f"{results.num_categories}"],
        ["Iterations", f"{results.num_iterations}"],
        ["Converged", "Yes" if results.converged else "No"],
        ["Training Time", f"{results.training_time:.2f}s"],
        ["Uncategorized", f"{results.get_uncategorized_count()}"],
    ]

    # mean/median match when we have finite values
    categorized = ~np.isnan(results.categories)
    if np.any(categorized):
        matches = results.matches[categorized]
        matches_finite = matches[np.isfinite(matches)]
        if len(matches_finite) > 0:
            mean_m = float(np.mean(matches_finite))
            median_m = float(np.median(matches_finite))
            stats_data.append(["Mean Match", f"{mean_m:.1f}%"])
            stats_data.append(["Median Match", f"{median_m:.1f}%"])

    table = ax.table(
        cellText=stats_data,
        cellLoc="left",
        loc="center",
        bbox=Bbox.from_bounds(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # style header row
    for i in range(2):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax.set_title("Training Statistics", fontweight="bold", pad=20)


def _plot_match_by_category_axes(ax: Axes, results: TrainingResults) -> None:
    """Plot match values by category as violin plot."""
    category_sizes = results.get_category_sizes()
    categories = sorted(category_sizes.keys())

    # only categories with at least one finite match (no mean-of-empty)
    data_by_category: List[NDArray[np.float64]] = []
    positions_cats: List[Tuple[int, Any]] = []
    for i, cat in enumerate(categories):
        mask = results.categories == cat
        cat_matches = results.matches[mask]
        cat_matches = cat_matches[np.isfinite(cat_matches)]
        if len(cat_matches) > 0:
            data_by_category.append(cat_matches)
            positions_cats.append((i, cat))

    if data_by_category:
        n_plot = len(data_by_category)
        parts = ax.violinplot(
            data_by_category, positions=range(n_plot), showmeans=True, showmedians=True
        )

        # color violins
        bodies = list(cast(Iterable[Any], parts["bodies"]))
        for pc in bodies:
            pc.set_facecolor("#2E86AB")
            pc.set_alpha(0.6)

        ax.set_xlabel("Category", fontweight="bold")
        ax.set_ylabel("Match Value (%)", fontweight="bold")
        ax.set_title("Match Values by Category", fontweight="bold")
        ax.set_xticks(range(n_plot))
        # many categories => smaller font, rotated, every k-th
        if n_plot > 24:
            step = max(1, n_plot // 25)
            labels = [
                str(cat) if i % step == 0 else ""
                for i, (_i, cat) in enumerate(list(positions_cats))
            ]
            ax.set_xticklabels(labels, fontsize=max(4, 10 - n_plot // 20), rotation=90)
        else:
            ax.set_xticklabels([str(cat) for _i, cat in list(positions_cats)])
        ax.grid(True, axis="y", alpha=0.3)
