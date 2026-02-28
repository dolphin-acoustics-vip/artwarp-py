"""
ARTwarp-py: the complete self-contained visualization gallery.

Demonstrates every plotting function available in artwarp.visualization,
using a single set of synthetic contours so the script is self-contained.

Sections
--------
1.  Synthetic data + training
2.  Standard results plots   (6 functions)
3.  Algorithm diagrams       (3 functions)
4.  Diagnostic plots         (4 functions)
5.  Data-quality plots       (2 functions)
6.  Parameter-study plots    (2 functions)
7.  Publication figure       (1 function)
8.  Ground-truth plots       (2 functions — require labels)
9.  Full automated report    (create_results_report)

All figures are saved to ./viz_gallery/.

@author: Pedro Gronda Garrigues
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless — works on servers without a display
import matplotlib.pyplot as plt
import numpy as np

from artwarp import ARTwarp
from artwarp.visualization import (
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

OUT_DIR = Path("./viz_gallery")
DPI = 150          # lower DPI keeps the gallery quick; use 300 for publication


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


def make_synthetic_contours(
    seed: int = 42,
) -> tuple[list[np.ndarray], list[str], list[str], list[float]]:
    """Return (contours, names, labels, tempres_list) for 40 synthetic whistles."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 30)
    tempres = 1 / 30  # ~0.033 s per sample

    shapes = {
        "ascending":   lambda: 100 + 200 * t,
        "descending":  lambda: 300 - 200 * t,
        "u_shaped":    lambda: 200 - 100 * np.sin(np.pi * t),
        "arch":        lambda: 100 + 200 * np.sin(np.pi * t),
    }
    contours, names, labels = [], [], []
    for shape_name, fn in shapes.items():
        for i in range(10):
            c = fn().astype(np.float64) + rng.normal(0, 8, len(t))
            contours.append(np.abs(c))   # keep positive
            names.append(f"{shape_name}_{i:02d}")
            labels.append(shape_name)

    tempres_list: list[float] = [tempres] * len(contours)
    return contours, names, labels, tempres_list


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nARTwarp-py  ·  Visualization Gallery → {OUT_DIR.resolve()}\n")

    # ── 1. Synthetic data ────────────────────────────────────────────────────
    print("── 1. Generating synthetic data")
    contours, names, labels, tempres_list = make_synthetic_contours()
    print(f"     {len(contours)} contours  ·  4 ground-truth types")

    # ── 2. Train ─────────────────────────────────────────────────────────────
    print("\n── 2. Training ARTwarp network")
    network = ARTwarp(
        vigilance=85.0,
        learning_rate=0.1,
        bias=0.0,
        max_categories=20,
        max_iterations=50,
        warp_factor_level=3,
        random_seed=42,
        verbose=False,
    )
    results = network.fit(contours, contour_names=names)
    print(f"     Categories: {results.num_categories}  ·  "
          f"Iterations: {results.num_iterations}  ·  "
          f"Converged: {results.converged}")

    # ── 3. Standard results plots ────────────────────────────────────────────
    print("\n── 3. Standard results plots")

    _save(plot_training_summary(results, names, figsize=(16, 10)),
          "01_training_summary.png")

    _save(plot_reference_contours(results.weight_matrix),
          "02_reference_contours.png")

    _save(plot_category_distribution(results, figsize=(10, 5)),
          "03_category_distribution.png")

    _save(plot_convergence_history(results, figsize=(10, 5)),
          "04_convergence_history.png")

    _save(plot_match_distribution(results, figsize=(10, 5)),
          "05_match_distribution.png")

    _save(plot_discovery_curve(results, title="Synthetic dataset", figsize=(10, 5)),
          "06_discovery_curve.png")

    first_cat = sorted(results.get_category_sizes())[0]
    ref = results.weight_matrix[:, first_cat]
    ref_clean = ref[~np.isnan(ref)]
    _save(
        plot_contours_by_category(contours, results.categories, first_cat,
                                  names, ref_clean, figsize=(12, 7)),
        f"07_category_{first_cat}_contours.png",
    )

    # ── 4. Algorithm diagrams ────────────────────────────────────────────────
    print("\n── 4. Algorithm diagrams")

    _save(plot_art_schematic(figsize=(13, 5)),
          "08_art_schematic.png")

    _save(plot_warp_constraint(warp_factor_level=3, m=20, n=25, figsize=(6, 6)),
          "09_warp_constraint.png")

    # find a contour/prototype pair compatible with warp_factor_level=3
    wfl = 3
    for ci, ctr in enumerate(contours[:30]):
        cat = int(results.categories[ci])
        if cat >= results.weight_matrix.shape[1]:
            continue
        ref_col = results.weight_matrix[:, cat][~np.isnan(results.weight_matrix[:, cat])]
        if len(ref_col) == 0:
            continue
        ratio = max(len(ctr), len(ref_col)) / max(1, min(len(ctr), len(ref_col)))
        if ratio < wfl:
            _save(
                plot_dtw_alignment(ctr, ref_col.astype(np.float64),
                                   warp_factor_level=wfl, figsize=(11, 5)),
                "10_dtw_alignment.png",
            )
            break

    # ── 5. Diagnostic plots ─────────────────────────────────────────────────
    print("\n── 5. Diagnostic plots")

    _save(plot_per_category_match_quality(results, figsize=(11, 5)),
          "11_per_category_match_quality.png")

    _save(plot_category_similarity_matrix(results.weight_matrix,
                                          warp_factor_level=3, figsize=(8, 7)),
          "12_category_similarity_matrix.png")

    _save(plot_category_embedding(results.weight_matrix,
                                  warp_factor_level=3, figsize=(8, 6)),
          "13_category_embedding.png")

    try:
        _save(plot_category_dendrogram(results.weight_matrix,
                                       warp_factor_level=3, figsize=(10, 5)),
              "14_category_dendrogram.png")
    except ImportError:
        print("  (scipy not available — skipping dendrogram)")

    # ── 6. Data-quality plots ────────────────────────────────────────────────
    print("\n── 6. Data-quality plots")

    _save(plot_contour_length_distribution(contours, tempres_list=tempres_list,
                                           figsize=(12, 5)),
          "15_contour_length_distribution.png")

    _save(
        plot_resampling_before_after(
            contours[0], tempres=tempres_list[0],
            sample_interval_sec=tempres_list[0] * 2,
            title="Example contour — before vs after resampling",
            figsize=(11, 5),
        ),
        "16_resampling_before_after.png",
    )

    # ── 7. Parameter-study plots ─────────────────────────────────────────────
    print("\n── 7. Parameter-study plots")

    sweep = [
        (v, ARTwarp(vigilance=v, random_seed=42, verbose=False).fit(contours))
        for v in [70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
    ]
    _save(plot_vigilance_sweep(sweep, figsize=(10, 5)),
          "17_vigilance_sweep.png")

    n_cats_per_run = [
        ARTwarp(vigilance=85.0, random_seed=s, verbose=False).fit(contours).num_categories
        for s in range(10)
    ]
    _save(plot_run_stability(n_cats_per_run, figsize=(8, 5)),
          "18_run_stability.png")

    # ── 8. Publication figure ────────────────────────────────────────────────
    print("\n── 8. Publication figure")

    _save(
        create_paper_figure(results, contours, names,
                            title="ARTwarp — Synthetic Dataset", figsize=(16, 10)),
        "19_paper_figure.png",
    )

    # ── 9. Ground-truth plots (require external labels) ─────────────────────
    print("\n── 9. Ground-truth plots  (using known synthetic labels)")

    # encode the 4 string labels as integers for the confusion matrix
    unique_labels = sorted(set(labels))
    label_ids = np.array([unique_labels.index(l) for l in labels], dtype=int)

    _save(
        plot_confusion_matrix(label_ids, results.categories,
                              figsize=(8, 7)),
        "20_confusion_matrix.png",
    )

    _save(
        plot_label_vs_category(label_ids, results.categories,
                               figsize=(10, 5)),
        "21_label_vs_category.png",
    )

    # ── 10. Full automated report ────────────────────────────────────────────
    print("\n── 10. Full automated report")

    report_files = create_results_report(
        results,
        contours,
        contour_names=names,
        output_dir=str(OUT_DIR / "full_report"),
        dpi=DPI,
        include_additional=True,
        warp_factor_level=3,
        tempres_list=tempres_list,
    )
    print(f"     {len(report_files)} figures saved to {OUT_DIR / 'full_report'}")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_individual = 21
    print(f"\n{'─' * 60}")
    print(f"Gallery complete — {n_individual} individual plots + {len(report_files)} report figures")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
