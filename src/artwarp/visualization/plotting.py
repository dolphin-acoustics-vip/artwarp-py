"""
Professional visualization functions for ARTwarp results.

This module provides comprehensive plotting capabilities following scientific
visualization best practices. All functions support customization and export
to various formats.

@author: Pedro Gronda Garrigues
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox
from numpy.typing import NDArray

from artwarp.core.network import TrainingResults

# Optional: DTW and resampling for alignment/resampling plots
try:
    from artwarp.core.dtw import compute_similarity_matrix, dynamic_time_warp
except ImportError:
    dynamic_time_warp = None  # type: ignore[assignment]
    compute_similarity_matrix = None  # type: ignore[assignment]
try:
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    from scipy.cluster.hierarchy import linkage

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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

    ax.set_xlabel("Match Score (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Distribution of Match Scores", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_discovery_curve(
    results: TrainingResults,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Plot the category discovery curve (cumulative distinct categories vs sample order).

    For each sample index in training order, the curve shows how many distinct
    categories have been observed so far. This matches the logic of the
    original MATLAB DiscoveryCurves.m: as you move through the dataset in order,
    the curve rises when a new category is encountered and stays flat when
    the sample belongs to an already-seen category.

    Useful for assessing how quickly the dataset "reveals" its category
    diversity and for comparing discovery rates across species or conditions.

    Args:
        results: TrainingResults object from network training (categories in sample order).
        title: Optional title for the figure (e.g. species or dataset name).
        figsize: Figure size in inches (width, height).
        save_path: Optional path to save figure.
        dpi: Resolution for saved figure.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> fig = plot_discovery_curve(results, title="Tursiops truncatus")
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    _plot_discovery_curve_axes(ax, results, title=title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def _plot_discovery_curve_axes(
    ax: Axes,
    results: TrainingResults,
    title: Optional[str] = None,
) -> None:
    """
    Plot the discovery curve on the given axes (for use in composite figures).

    Args:
        ax: Matplotlib Axes to draw on.
        results: TrainingResults with categories in sample order.
        title: Optional title string for the axes.
    """
    categories = results.categories
    n_samples = len(categories)

    if n_samples == 0:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", fontsize=14)
        return

    # cumulative distinct categories -> for each prefix categories[0:i+1], count unique (finite) IDs
    seen: Set[int] = set()
    y_vals: List[int] = []
    for i in range(n_samples):
        c = categories[i]
        if np.isfinite(c):
            seen.add(int(c))
        y_vals.append(len(seen))

    x_vals = np.arange(1, n_samples + 1, dtype=np.int64)

    ax.plot(
        x_vals,
        y_vals,
        linewidth=2,
        color="#2E86AB",
        label="Cumulative categories",
    )

    ax.set_xlabel("Number of Samples (whistles / contours)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Categories", fontsize=12, fontweight="bold")
    ax.set_title(
        title if title else "Category Discovery Curve",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_results_report(
    results: TrainingResults,
    contours: List[NDArray[np.float64]],
    contour_names: Optional[List[str]] = None,
    output_dir: str = "./artwarp_report",
    dpi: int = 300,
    include_additional: bool = True,
    warp_factor_level: int = 3,
    tempres_list: Optional[List[Optional[float]]] = None,
) -> Dict[str, str]:
    """
    Create a visualization report with multiple figures.

    Generates and saves multiple visualization figures to a directory,
    providing a complete visual analysis of training results. If
    include_additional is True (default), also writes extra diagnostic
    and algorithm figures into output_dir/additional/.

    Args:
        results: TrainingResults object from network training
        contours: List of frequency contour arrays
        contour_names: Optional list of contour names
        output_dir: Directory to save all figures
        dpi: Resolution for saved figures
        include_additional: If True, generate extra figures in output_dir/additional/
        warp_factor_level: DTW warp factor used for alignment/similarity plots (default 3)
        tempres_list: Optional per-contour temporal resolution values (s/sample).
            When provided, the contour length distribution figure will include a
            temporal resolution histogram in its right panel.

    Returns:
        Dictionary mapping figure names to file paths (includes keys for additional/ if generated)

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

    # 6. Discovery curve
    print("Creating discovery curve...")
    fig6 = plot_discovery_curve(results)
    path6 = output_path / "discovery_curve.png"
    fig6.savefig(path6, dpi=dpi, bbox_inches="tight")
    plt.close(fig6)
    saved_files["discovery_curve"] = str(path6)

    # 7. Contours by category (for each category)
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

    # 8. Additional figures in output_dir/additional/ (algorithm, diagnostics, paper figure)
    if include_additional:
        add_path = output_path / "additional"
        add_path.mkdir(parents=True, exist_ok=True)
        print("Creating additional figures in additional/...")
        wfl = warp_factor_level

        try:
            fig = plot_art_schematic(figsize=(8, 5))
            p = add_path / "art_schematic.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files["additional/art_schematic"] = str(p)
        except Exception:  # noqa: BLE001
            pass

        try:
            # use first ref contour length for grid size (or defaults)
            n_valid = (
                int(np.sum(~np.isnan(results.weight_matrix[:, 0])))
                if results.weight_matrix.shape[1] > 0
                else 15
            )
            m, n = min(25, max(5, n_valid)), min(30, max(5, n_valid + 5))
            fig = plot_warp_constraint(warp_factor_level=wfl, m=m, n=n, figsize=(6, 6))
            p = add_path / "warp_constraint.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files["additional/warp_constraint"] = str(p)
        except Exception:  # noqa: BLE001
            pass

        try:
            fig = plot_per_category_match_quality(results, figsize=(10, 6))
            p = add_path / "per_category_match_quality.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files["additional/per_category_match_quality"] = str(p)
        except Exception:  # noqa: BLE001
            pass

        try:
            fig = plot_category_similarity_matrix(
                results.weight_matrix, warp_factor_level=wfl, figsize=(8, 7)
            )
            p = add_path / "category_similarity_matrix.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files["additional/category_similarity_matrix"] = str(p)
        except Exception:  # noqa: BLE001
            pass

        try:
            fig = plot_category_embedding(
                results.weight_matrix, warp_factor_level=wfl, figsize=(8, 6)
            )
            p = add_path / "category_embedding.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files["additional/category_embedding"] = str(p)
        except Exception:  # noqa: BLE001
            pass

        if len(contours) > 0:
            try:
                fig = plot_contour_length_distribution(
                    contours, tempres_list=tempres_list, figsize=(12, 5)
                )
                p = add_path / "contour_length_distribution.png"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                saved_files["additional/contour_length_distribution"] = str(p)
            except Exception:  # noqa: BLE001
                pass

        if len(contours) > 0 and results.num_categories > 0 and dynamic_time_warp is not None:
            try:
                # find a contour/prototype pair whose length ratio is within warp_factor_level
                # so that a valid warping path exists (try up to 50 contours).
                _dtw_fig: Optional[Figure] = None
                _max_ratio = float(wfl)
                for _ci, _ctr in enumerate(contours[:50]):
                    _cat = int(results.categories[_ci]) if _ci < len(results.categories) else 0
                    if _cat >= results.weight_matrix.shape[1]:
                        continue
                    _ref_col = results.weight_matrix[:, _cat]
                    _valid = ~np.isnan(_ref_col)
                    _ref = _ref_col[_valid]
                    if len(_ref) == 0 or len(_ctr) == 0:
                        continue
                    _ratio = max(len(_ctr), len(_ref)) / max(1, min(len(_ctr), len(_ref)))
                    if _ratio < _max_ratio:
                        _dtw_fig = plot_dtw_alignment(
                            _ctr, _ref.astype(np.float64), warp_factor_level=wfl, figsize=(8, 10)
                        )
                        break
                if _dtw_fig is not None:
                    p = add_path / "dtw_alignment.png"
                    _dtw_fig.savefig(p, dpi=dpi, bbox_inches="tight")
                    plt.close(_dtw_fig)
                    saved_files["additional/dtw_alignment"] = str(p)
            except Exception:  # noqa: BLE001
                pass

        if len(contours) > 0:
            try:
                fig = plot_resampling_before_after(
                    contours[0],
                    tempres=0.01,
                    sample_interval_sec=0.02,
                    title="First contour (demo)",
                )
                p = add_path / "resampling_before_after.png"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                saved_files["additional/resampling_before_after"] = str(p)
            except Exception:  # noqa: BLE001
                pass

        try:
            fig = create_paper_figure(
                results, contours, contour_names, title=None, figsize=(16, 12)
            )
            p = add_path / "paper_figure.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files["additional/paper_figure"] = str(p)
        except Exception:  # noqa: BLE001
            pass

        if SCIPY_AVAILABLE:
            try:
                fig = plot_category_dendrogram(
                    results.weight_matrix, warp_factor_level=wfl, figsize=(10, 6)
                )
                p = add_path / "category_dendrogram.png"
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                saved_files["additional/category_dendrogram"] = str(p)
            except Exception:  # noqa: BLE001
                pass

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
        ax.set_xlabel("Match Score (%)", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Match Distribution", fontweight="bold")
        return
    ax.hist(matches, bins=20, alpha=0.7, color="lightgreen", edgecolor="black")
    ax.axvline(float(np.mean(matches)), color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Match Score (%)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("Match Distribution", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)


def _plot_convergence_history_axes(ax: Axes, results: TrainingResults) -> None:
    """Plot convergence history on given axes."""
    if len(results.iteration_history) > 0:
        iters, reclassifications = zip(*results.iteration_history)
        ax.plot(iters, reclassifications, "o-", linewidth=2, color="purple")
        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_ylabel("Reclassifications", fontweight="bold")
        ax.set_title("Convergence History", fontweight="bold")
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
        ax.set_ylabel("Match Score (%)", fontweight="bold")
        ax.set_title("Match Score by Category", fontweight="bold")
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


# =============================================================================
# Additional visualizations (algorithm, diagnostics, reporting)
# =============================================================================


def plot_dtw_alignment(
    u1: NDArray[np.float64],
    u2: NDArray[np.float64],
    warp_factor_level: int = 3,
    figsize: Tuple[int, int] = (8, 10),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Plot DTW alignment between two frequency contours.

    Shows the optimal warping path on the alignment grid and the two contours
    overlaid for direct comparison. Standard visualization in time-series
    and bioacoustics.

    Args:
        u1: Reference contour (frequency values, Hz).
        u2: Comparison contour (frequency values, Hz).
        warp_factor_level: DTW warp factor used for alignment.
        figsize: Figure size in inches.
        save_path: Optional path to save figure.
        dpi: Resolution for saved figure.

    Returns:
        Matplotlib Figure.

    Raises:
        ImportError: If artwarp.core.dtw is unavailable.
        ValueError: If the length ratio exceeds warp_factor_level (no valid path).
    """
    if dynamic_time_warp is None:
        raise ImportError("plot_dtw_alignment requires artwarp.core.dtw")
    sim, warp_func = dynamic_time_warp(u1, u2, warp_factor_level)
    if len(warp_func) == 0:
        raise ValueError(
            f"DTW returned no valid path: length ratio between contours ({len(u1)}, {len(u2)}) "
            f"exceeds warp_factor_level={warp_factor_level}."
        )
    m, n = len(u1), len(u2)
    wfl = warp_factor_level
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # left -> both contours overlaid on the same time axis
    ax1.plot(np.arange(m), u1, color="#2E86AB", linewidth=2, label=f"Reference  (n = {m})")
    ax1.plot(
        np.arange(n),
        u2,
        color="#E84855",
        linewidth=2,
        linestyle="--",
        label=f"Comparison  (n = {n})",
    )
    ax1.set_xlabel("Time Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax1.set_title("Input Contours", fontsize=11, fontweight="bold")
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(-0.5, max(m, n) - 0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # right -> warping path on the alignment grid
    path_i = np.arange(m, dtype=float)
    path_j = warp_func.astype(float)
    # background -> allowed warping band (use same bounds as dtw.py DP loop)
    for i in range(m):
        j_lo = max(0, max(round((i + 1) / wfl) - 1, (i - m) * wfl + n))
        j_hi = min(n - 1, min(wfl * (i + 1), round((i - m) / wfl + n)) - 1)
        if j_hi >= j_lo:
            ax2.add_patch(
                Rectangle(
                    (j_lo - 0.5, i - 0.5),
                    j_hi - j_lo + 1.0,
                    1.0,
                    facecolor="#DAEAF5",
                    alpha=0.7,
                    linewidth=0,
                )
            )
    # diagonal -> no-warp reference
    diag_end = min(m, n)
    ax2.plot(
        np.linspace(0, diag_end - 1, diag_end),
        np.linspace(0, diag_end - 1, diag_end),
        color="#AAAAAA",
        linewidth=1,
        linestyle=":",
        label="No-Warp Diagonal",
    )
    # actual warping path
    ax2.plot(
        path_j, path_i, color="#1A1A2E", linewidth=2, label=f"Warping Path  (sim = {sim:.1f}%)"
    )
    ax2.set_xlabel("Comparison Contour Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Reference Contour Index", fontsize=11, fontweight="bold")
    ax2.set_title("DTW Warping Path", fontsize=11, fontweight="bold")
    ax2.set_xlim(-0.5, n - 0.5)
    ax2.set_ylim(m - 0.5, -0.5)
    ax2.set_aspect("auto")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Dynamic Time Warping Alignment", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_art_schematic(
    figsize: Tuple[int, int] = (13, 5),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """ARTwarp algorithm decision flow diagram."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    FLOW_FACE = "#D6E8F5"
    FLOW_EDGE = "#2E6DA4"
    COMMIT_FACE = "#D8EFD8"
    COMMIT_EDGE = "#3A7A3A"
    NEW_FACE = "#FFF0D0"
    NEW_EDGE = "#C87D00"
    ARROW_KW = dict(arrowstyle="-|>", color="#333333", lw=1.6, mutation_scale=14)
    BOX_H = 1.1

    def _box(
        cx: float,
        cy: float,
        w: float,
        label: str,
        facecolor: str,
        edgecolor: str,
        fontsize: float = 9.5,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - BOX_H / 2),
                w,
                BOX_H,
                boxstyle="round,pad=0.08",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.8,
                zorder=3,
            )
        )
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            zorder=4,
            linespacing=1.4,
        )

    def _arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(ARROW_KW), zorder=5)

    def _label(x: float, y: float, text: str, color: str = "#555555") -> None:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
            fontstyle="italic",
            color=color,
            zorder=6,
        )

    # ── Main flow boxes (top row) ─────────────────────────────────────────────
    _box(1.2, 3.0, 1.8, "Input\ncontour", FLOW_FACE, FLOW_EDGE)
    _box(3.7, 3.0, 2.2, "DTW to all\nprototypes", FLOW_FACE, FLOW_EDGE)
    _box(6.4, 3.0, 2.0, "Sort by\nactivation", FLOW_FACE, FLOW_EDGE)
    _box(9.1, 3.0, 2.0, "Vigilance\ntest  (ρ)", FLOW_FACE, FLOW_EDGE)

    # ── Outcome boxes (bottom row) ────────────────────────────────────────────
    _box(
        5.8, 1.0, 2.6, "Update prototype\nweights (commit)", COMMIT_FACE, COMMIT_EDGE, fontsize=9.0
    )
    _box(10.2, 1.0, 2.4, "Create new\ncategory", NEW_FACE, NEW_EDGE, fontsize=9.0)

    # ── Horizontal main-flow arrows ───────────────────────────────────────────
    _arrow(2.1, 3.0, 2.6, 3.0)
    _arrow(4.8, 3.0, 5.4, 3.0)
    _arrow(7.4, 3.0, 8.1, 3.0)

    # ── Decision routing from vigilance box ──────────────────────────────────
    # branch point -> bottom centre of vigilance box
    bx, by = 9.1, 3.0 - BOX_H / 2  # (9.1, 2.45)
    jy = 1.8  # horizontal junction y

    # trunk -> down from vigilance box to junction
    ax.plot([bx, bx], [by, jy], color="#333333", lw=1.6, zorder=2)
    # left branch -> junction → commit
    ax.plot([bx, 5.8], [jy, jy], color="#333333", lw=1.6, zorder=2)
    _arrow(5.8, jy, 5.8, 1.0 + BOX_H / 2)
    # right branch -> junction → new category
    ax.plot([bx, 10.2], [jy, jy], color="#333333", lw=1.6, zorder=2)
    _arrow(10.2, jy, 10.2, 1.0 + BOX_H / 2)

    # branch labels
    _label(5.0, jy + 0.22, "match ≥ ρ", COMMIT_EDGE)
    _label(10.9, jy + 0.22, "match < ρ", NEW_EDGE)

    # ── Step numbers ─────────────────────────────────────────────────────────
    for step, cx in enumerate([1.2, 3.7, 6.4, 9.1], 1):
        ax.text(
            cx,
            3.0 + BOX_H / 2 + 0.15,
            f"Step {step}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#777777",
        )

    ax.set_title(
        "ARTwarp Algorithm: Per-Sample Decision Flow", fontsize=12, fontweight="bold", pad=8
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_warp_constraint(
    warp_factor_level: int = 3,
    m: int = 20,
    n: int = 25,
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Itakura parallelogram constraint for DTW alignment.

    Shows the feasible cell region (shaded) in the m×n alignment grid for a
    given warp_factor_level (ρ). Cells outside the band are pruned during DP.

    Useful for explaining the temporal-flexibility constraint in papers.
    """
    fig, ax = plt.subplots(figsize=figsize)
    wfl = warp_factor_level

    # shaded feasible region -> use exact same bounds as dtw.py DP loop
    j_lo_all: NDArray[np.int_] = np.zeros(m, dtype=int)
    j_hi_all: NDArray[np.int_] = np.zeros(m, dtype=int)
    for i in range(m):
        j_lo_all[i] = max(0, max(round((i + 1) / wfl) - 1, (i - m) * wfl + n))
        j_hi_all[i] = min(n - 1, min(wfl * (i + 1), round((i - m) / wfl + n)) - 1)

    for i in range(m):
        if j_hi_all[i] >= j_lo_all[i]:
            ax.add_patch(
                Rectangle(
                    (j_lo_all[i] - 0.5, i - 0.5),
                    j_hi_all[i] - j_lo_all[i] + 1.0,
                    1.0,
                    facecolor="#2E86AB",
                    alpha=0.18,
                    linewidth=0,
                )
            )

    # band boundary -> connect the left and right edges of the feasible cells
    i_vals = np.arange(m)
    ax.plot(j_lo_all, i_vals, color="#2E86AB", lw=1.2, linestyle="--", label="Band Boundary")
    ax.plot(j_hi_all, i_vals, color="#2E86AB", lw=1.2, linestyle="--")

    # diagonal reference -> no warping
    diag = np.linspace(0, min(m, n) - 1, min(m, n))
    ax.plot(diag * n / m, diag, color="#E84855", lw=1.8, linestyle="-", label="No-warp diagonal")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    ax.set_xlabel("Comparison Contour Index ($j$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Reference Contour Index ($i$)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Itakura Warping Constraint  ($\\rho = {wfl}$)\n"
        f"Shaded: feasible DTW alignment cells  |  m = {m}, n = {n}",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_per_category_match_quality(
    results: TrainingResults,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Per-category match score distribution (violin plot).

    Shows how well samples in each category match their prototype via match
    score distributions.
    """
    sizes = results.get_category_sizes()
    categories = sorted(sizes.keys())
    data: List[NDArray[np.float64]] = []
    for cat in categories:
        mask = results.categories == cat
        m_arr = results.matches[mask]
        m_arr = m_arr[np.isfinite(m_arr)]
        data.append(m_arr if len(m_arr) > 0 else np.array([np.nan]))

    fig, ax = plt.subplots(figsize=figsize)
    n_cats = len(categories)
    if data and n_cats > 0:
        parts = ax.violinplot(data, positions=range(n_cats), showmeans=True, showmedians=True)
        bodies = list(cast(Iterable[Any], parts["bodies"]))
        for pc in bodies:
            pc.set_facecolor("#2E86AB")
            pc.set_alpha(0.6)

    ax.set_xticks(range(n_cats))
    if n_cats > 24:
        step = max(1, n_cats // 25)
        labels = [str(categories[i]) if i % step == 0 else "" for i in range(n_cats)]
        ax.set_xticklabels(labels, fontsize=max(4, 10 - n_cats // 20), rotation=90)
    else:
        ax.set_xticklabels([str(c) for c in categories], rotation=45 if n_cats > 12 else 0)

    ax.set_xlabel("Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Match Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Category Match Quality", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_category_similarity_matrix(
    weight_matrix: NDArray[np.float64],
    warp_factor_level: int = 3,
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Heatmap of pairwise DTW similarity between all category prototypes.

    Each cell (i, j) encodes the DTW similarity (%) between the learned prototype
    contours for categories i and j. The diagonal is 100 % by definition.

    High off-diagonal values indicate acoustically similar categories.
    """
    if dynamic_time_warp is None:
        raise ImportError("plot_category_similarity_matrix requires artwarp.core.dtw")
    num_cats = weight_matrix.shape[1]
    refs: List[NDArray[np.float64]] = []
    for c in range(num_cats):
        col = weight_matrix[:, c]
        valid = ~np.isnan(col)
        refs.append(col[valid].astype(np.float64) if np.any(valid) else np.array([0.0]))
    sim_matrix = np.zeros((num_cats, num_cats))
    for i in range(num_cats):
        for j in range(num_cats):
            if i == j:
                sim_matrix[i, j] = 100.0
            else:
                s, _ = dynamic_time_warp(refs[i], refs[j], warp_factor_level)
                sim_matrix[i, j] = s

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="DTW Similarity (%)", shrink=0.85)

    # tick thinning -> show at most ~20 labels per axis
    max_ticks = 20
    if num_cats > max_ticks:
        step = max(1, num_cats // max_ticks)
        ticks = list(range(0, num_cats, step))
        labels = [str(i) for i in ticks]
        tick_fs = max(5, 9 - num_cats // 15)
    else:
        ticks = list(range(num_cats))
        labels = [str(i) for i in ticks]
        tick_fs = 9

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=tick_fs, rotation=90 if num_cats > 12 else 0)
    ax.set_yticklabels(labels, fontsize=tick_fs)
    ax.set_xlabel("Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Category", fontsize=11, fontweight="bold")
    ax.set_title(
        "Pairwise DTW Similarity Between Category Prototypes (%)", fontsize=11, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def _classical_mds(D: NDArray[np.float64], n_components: int = 2) -> NDArray[np.float64]:
    """Classical MDS: embed distance matrix D into n_components dimensions."""
    n = D.shape[0]
    D2 = D**2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (H @ D2 @ H)
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:n_components]
    return v[:, idx] * np.sqrt(np.maximum(w[idx], 0))


def plot_category_embedding(
    weight_matrix: NDArray[np.float64],
    warp_factor_level: int = 3,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    2D acoustic space embedding of category prototypes via classical MDS.

    Pairwise DTW distances between all learned prototype contours are reduced to
    two dimensions using classical Multidimensional Scaling (MDS).

    Categories that are close on the scatter plot are acoustically similar (small DTW distance);
    distant points represent perceptually distinct call types.
    """
    if dynamic_time_warp is None:
        raise ImportError("plot_category_embedding requires artwarp.core.dtw")
    num_cats = weight_matrix.shape[1]
    refs: List[NDArray[np.float64]] = []
    for c in range(num_cats):
        col = weight_matrix[:, c]
        valid = ~np.isnan(col)
        refs.append(col[valid].astype(np.float64) if np.any(valid) else np.array([0.0]))
    dist_matrix = np.zeros((num_cats, num_cats))
    for i in range(num_cats):
        for j in range(num_cats):
            if i != j:
                s, _ = dynamic_time_warp(refs[i], refs[j], warp_factor_level)
                dist_matrix[i, j] = 100.0 - s  # distance = 100 - similarity
    embed = _classical_mds(dist_matrix, 2)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab20" if num_cats > 10 else "tab10")
    scatter = ax.scatter(
        embed[:, 0],
        embed[:, 1],
        c=np.arange(num_cats),
        cmap=cmap,
        s=90,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    # label points only when the plot is not too cluttered
    if num_cats <= 40:
        for i in range(num_cats):
            ax.annotate(
                str(i),
                (embed[i, 0], embed[i, 1]),
                fontsize=7.5,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )
    plt.colorbar(scatter, ax=ax, label="Category index", shrink=0.80)

    ax.set_xlabel(
        "MDS Dimension 1  (captures the most variation in pairwise DTW distances)",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylabel(
        "MDS Dimension 2",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_title(
        "Acoustic Space: Category Prototype Embedding\n"
        "(Classical MDS on pairwise DTW distances — closer = acoustically similar)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_resampling_before_after(
    contour: NDArray[np.float64],
    tempres: float,
    sample_interval_sec: float,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Plot one contour before and after resampling (original vs resampled).

    Demonstrates the effect of temporal normalization.
    """
    from artwarp.utils.resample import resample_contours

    resampled_list = resample_contours([contour], [tempres], sample_interval_sec)
    resampled = resampled_list[0]
    n_orig, n_res = len(contour), len(resampled)
    t_orig = np.arange(n_orig) * tempres
    t_res = np.arange(n_res) * sample_interval_sec
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharey=True)
    ax1.plot(t_orig, contour, "b-o", markersize=3, label=f"Original  (n = {n_orig})")
    ax1.set_ylabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax1.set_title("Before Resampling", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.plot(t_res, resampled, "g-o", markersize=3, label=f"Resampled  (n = {n_res})")
    ax2.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax2.set_title("After Resampling", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_contour_length_distribution(
    contours: List[NDArray[np.float64]],
    tempres_list: Optional[List[Optional[float]]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Histograms of contour lengths and (optionally) temporal resolution.

    The left panel shows the distribution of raw contour lengths in samples.
    The right panel shows the distribution of per-file temporal resolution
    (seconds per sample) when ``tempres_list`` is provided and contains valid
    values; otherwise a summary note is displayed!

    Useful for diagnosing mixed-resolution datasets before resampling :)
    """
    lengths = [len(c) for c in contours]
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # left -> contour length histogram
    axes[0].hist(
        lengths,
        bins=min(50, max(10, len(set(lengths)))),
        color="#2E86AB",
        edgecolor="white",
        alpha=0.85,
        linewidth=0.5,
    )
    axes[0].set_xlabel("Contour Length (samples)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=11, fontweight="bold")
    axes[0].set_title("Contour Length Distribution", fontsize=12, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    n_contours = len(contours)
    if n_contours > 0:
        axes[0].set_xlabel(
            f"Contour Length (samples)  —  n={n_contours}, "
            f"median={int(np.median(lengths))}, max={max(lengths)}",
            fontsize=9,
        )

    # right -> temporal resolution histogram or summary
    tr_vals: List[float] = []
    if tempres_list is not None:
        tr_vals = [t for t in tempres_list if t is not None and t > 0]

    if tr_vals:
        # IMPORTANT:
        # round to 6 significant decimal places before counting unique values so
        # that floating-point noise in identical resolutions doesn't inflate the
        # bin count and cause a "Too many bins for data range" error.
        #
        # Please do not change this :(
        tr_arr = np.array(tr_vals)
        n_unique_tr = len(np.unique(np.round(tr_arr, 6)))
        n_bins_tr = 1 if n_unique_tr <= 1 else min(30, max(3, n_unique_tr))
        axes[1].hist(
            tr_vals,
            bins=n_bins_tr,
            color="#E84855",
            edgecolor="white",
            alpha=0.85,
            linewidth=0.5,
        )
        axes[1].set_xlabel(
            f"Temporal Resolution (s/sample)  —  "
            f"median={np.median(tr_vals):.4f}s, range=[{min(tr_vals):.4f}, {max(tr_vals):.4f}]",
            fontsize=9,
        )
        axes[1].set_ylabel("Count", fontsize=11, fontweight="bold")
        axes[1].set_title("Temporal Resolution Distribution", fontsize=12, fontweight="bold")
    else:
        msg = (
            "Temporal resolution not available.\n" "Pass tempres_list to enable this panel."
            if tempres_list is None
            else "No valid temporal resolution values\nfound in the supplied list."
        )
        axes[1].text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=10,
            color="#777777",
            style="italic",
            multialignment="center",
        )
        axes[1].set_title("Temporal Resolution Distribution", fontsize=12, fontweight="bold")

    axes[1].grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_vigilance_sweep(
    sweep_results: List[Tuple[float, TrainingResults]],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Plot number of categories (and optional mean match) vs vigilance from a parameter sweep.

    sweep_results: list of (vigilance_value, TrainingResults).
    """
    vigs = [r[0] for r in sweep_results]
    n_cats = [r[1].num_categories for r in sweep_results]
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(vigs, n_cats, "bo-", linewidth=2, markersize=8, label="Number of categories")
    ax1.set_xlabel("Vigilance (ρ)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Categories", color="b", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    mean_matches = []
    for _, res in sweep_results:
        m = res.matches[np.isfinite(res.matches)]
        mean_matches.append(float(np.mean(m)) if len(m) > 0 else np.nan)
    ax2.plot(vigs, mean_matches, "r--", linewidth=1.5, markersize=6, label="Mean Match Score (%)")
    ax2.set_ylabel("Mean Match Score (%)", color="r", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.legend(loc="upper right")
    ax1.set_title("Vigilance Parameter Sweep", fontsize=13, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_run_stability(
    num_categories_per_run: List[int],
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Distribution of category count across multiple independent runs.

    Runs with different random seeds or data orderings are expected to produce
    slightly different numbers of categories. This histogram summarizes that
    variability, supporting reproducibility claims (somewhat).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        num_categories_per_run,
        bins=min(20, max(5, len(set(num_categories_per_run)))),
        color="#2E86AB",
        edgecolor="white",
        alpha=0.85,
        linewidth=0.5,
    )
    ax.set_xlabel("Number of Categories", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count (Runs)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Run Stability: Distribution of Category Count Across Runs", fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def create_paper_figure(
    results: TrainingResults,
    contours: List[NDArray[np.float64]],
    contour_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Single multi-panel figure: method summary + discovery curve + category distribution.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    ax1 = fig.add_subplot(gs[0, :])
    _plot_reference_contours_axes(ax1, results.weight_matrix)
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_discovery_curve_axes(ax2, results, title=None)
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_category_distribution_axes(ax3, results)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_category_dendrogram(
    weight_matrix: NDArray[np.float64],
    warp_factor_level: int = 3,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Dendrogram of category prototypes from hierarchical clustering on pairwise DTW distances.

    Shows which categories are acoustically similar.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("plot_category_dendrogram requires scipy")
    if dynamic_time_warp is None:
        raise ImportError("plot_category_dendrogram requires artwarp.core.dtw")
    num_cats = weight_matrix.shape[1]
    refs: List[NDArray[np.float64]] = []
    for c in range(num_cats):
        col = weight_matrix[:, c]
        valid = ~np.isnan(col)
        refs.append(col[valid].astype(np.float64) if np.any(valid) else np.array([0.0]))
    dist_list: List[float] = []
    for i in range(num_cats):
        for j in range(i + 1, num_cats):
            s, _ = dynamic_time_warp(refs[i], refs[j], warp_factor_level)
            dist_list.append(100.0 - s)

    # linkage expects condensed distance (upper triangle)
    Z = linkage(np.array(dist_list), method="average")
    fig, ax = plt.subplots(figsize=figsize)
    scipy_dendrogram(
        Z, ax=ax, labels=[str(i) for i in range(num_cats)], color_threshold=0.7 * max(Z[:, 2])
    )
    ax.set_title(
        "Category Prototype Dendrogram\n(Hierarchical Clustering DTW Distance — average linkage)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("DTW Distance", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    ground_truth_labels: List[Union[str, int]],
    categories: NDArray[np.float64],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """
    Confusion matrix: rows = ground truth, columns = ART category (or predicted).

    Use when ground-truth labels are available for evaluation.
    """
    gt = np.array(ground_truth_labels)
    pred = np.array(categories)
    valid = np.isfinite(pred)
    gt = gt[valid]
    pred = pred[valid].astype(int)
    unique_gt = list(dict.fromkeys(gt))
    unique_pred = list(dict.fromkeys(pred))
    n_gt, n_pred = len(unique_gt), len(unique_pred)
    cm = np.zeros((n_gt, n_pred))
    gt_idx = {v: i for i, v in enumerate(unique_gt)}
    pred_idx = {v: i for i, v in enumerate(unique_pred)}
    for g, p in zip(gt, pred):
        if g in gt_idx and p in pred_idx:
            cm[gt_idx[g], pred_idx[p]] += 1
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n_pred))
    ax.set_yticks(range(n_gt))
    ax.set_xticklabels([str(unique_pred[i]) for i in range(n_pred)])
    ax.set_yticklabels([str(unique_gt[i]) for i in range(n_gt)])
    ax.set_xlabel("ART category (predicted)")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix")
    for i in range(n_gt):
        for j in range(n_pred):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_label_vs_category(
    ground_truth_labels: List[Union[str, int]],
    categories: NDArray[np.float64],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Figure:
    """Stacked bar: per ART category, proportion of each ground-truth label."""
    gt = np.array(ground_truth_labels)
    pred = np.array(categories)
    valid = np.isfinite(pred)
    gt = gt[valid]
    pred = pred[valid].astype(int)
    unique_gt = list(dict.fromkeys(gt))
    unique_pred = sorted(dict.fromkeys(pred))
    n_pred = len(unique_pred)
    pred_idx = {v: i for i, v in enumerate(unique_pred)}
    counts = np.zeros((n_pred, len(unique_gt)))
    for g, p in zip(gt, pred):
        if p in pred_idx:
            j = unique_gt.index(g) if g in unique_gt else 0
            counts[pred_idx[p], j] += 1
    totals = counts.sum(axis=1, keepdims=True)
    proportions = np.where(totals > 0, counts / totals, 0)
    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(n_pred)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(unique_gt)))
    for j, label in enumerate(unique_gt):
        ax.bar(range(n_pred), proportions[:, j], bottom=left, label=str(label), color=colors[j])
        left += proportions[:, j]
    ax.set_xlabel("ART category")
    ax.set_ylabel("Proportion")
    ax.set_xticks(range(n_pred))
    ax.set_xticklabels([str(c) for c in unique_pred])
    ax.set_title("Label distribution per ART category")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# TODO: Colour different reference contour categories (according to naming)
