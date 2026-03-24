#!/usr/bin/env python3
"""
Compare two ARTwarp categorization outputs for the same contour set.

Since category labels are arbitrary in unsupervised clustering, direct label
comparison is meaningless.  Instead, this script uses label-independent
clustering metrics that ask: "do the two systems agree on which contours
belong together?"

Metrics computed
----------------
Adjusted Rand Index (ARI)
    Measures pair-wise agreement between two clusterings, corrected for
    chance.  1.0 = perfect agreement, 0.0 = random, negative = worse than
    chance.

Normalized Mutual Information (NMI)
    Measures shared information between two label sets, normalised to [0, 1].
    1.0 = identical clustering structure.

Fowlkes-Mallows Score (FMI)
    Geometric mean of pairwise precision and recall.  1.0 = perfect.

Purity
    For each Python cluster, the fraction of items assigned to the most
    common MATLAB category.  Averaged across clusters weighted by size.

Homogeneity / Completeness / V-measure
    Homogeneity: each cluster contains only members of a single MATLAB label.
    Completeness: all members of a MATLAB label are in the same cluster.
    V-measure: harmonic mean of the two (similar to F1).

Best-match accuracy (Hungarian / linear-sum assignment)
    Optimally maps Python categories to MATLAB categories (1-to-1), then
    computes the fraction of contours with matching labels.  Gives an upper
    bound on label accuracy under the best possible relabelling.

Additionally prints:
  - category count comparison
  - average match score comparison
  - per-MATLAB-category purity breakdown (sorted by impurity)

Usage
-----
From artwarp-py root directory:

    python scripts/compare_categorizations.py \\
        scripts/ARTwarp96FINAL_assignments.csv \\
        scripts/category_assignments_large.csv

Or with explicit labels:
    python scripts/compare_categorizations.py <matlab_csv> <python_csv>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        fowlkes_mallows_score,
        homogeneity_completeness_v_measure,
    )
    from scipy.optimize import linear_sum_assignment

    _has_sklearn = True
except ImportError:  # pragma: no cover
    _has_sklearn = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strip_ext(name: str) -> str:
    """Remove a known contour file extension from a name."""
    for ext in (".ctr", ".csv", ".txt"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Weighted average purity of pred clusters relative to true labels."""
    total = len(labels_true)
    score = 0.0
    for cat in np.unique(labels_pred):
        mask = labels_pred == cat
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels_true[mask])
        score += counts.max()
    return score / total


def _best_match_accuracy(
    labels_true: np.ndarray, labels_pred: np.ndarray
) -> float:
    """
    Hungarian-algorithm accuracy: find the 1-to-1 mapping from pred categories
    to true categories that maximises accuracy, then return that accuracy.
    """
    true_cats = np.unique(labels_true)
    pred_cats = np.unique(labels_pred)
    # cost matrix: rows = pred categories, cols = true categories
    cost = np.zeros((len(pred_cats), len(true_cats)), dtype=int)
    true_idx = {c: i for i, c in enumerate(true_cats)}
    pred_idx = {c: i for i, c in enumerate(pred_cats)}
    for t, p in zip(labels_true, labels_pred):
        cost[pred_idx[p], true_idx[t]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)  # maximise
    matched = cost[row_ind, col_ind].sum()
    return matched / len(labels_true)


def _separator(char: str = "─", width: int = 70) -> str:
    return char * width


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/compare_categorizations.py "
            "<matlab_csv> <python_csv>"
        )
        sys.exit(1)

    matlab_path = Path(sys.argv[1])
    python_path = Path(sys.argv[2])

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    df_mat = pd.read_csv(matlab_path)
    df_py  = pd.read_csv(python_path)

    # Normalise contour names: strip extensions and whitespace
    df_mat["key"] = df_mat["contour_name"].str.strip().apply(_strip_ext)
    df_py["key"]  = df_py["contour_name"].str.strip().apply(_strip_ext)

    # Fix MATLAB match scores: if values are ~100× too large, divide by 100
    mat_match_max = df_mat["match"].max()
    if mat_match_max > 200:
        df_mat["match"] = df_mat["match"] / 100.0

    # ------------------------------------------------------------------
    # Align on shared contour names
    # ------------------------------------------------------------------
    merged = pd.merge(df_mat, df_py, on="key", suffixes=("_mat", "_py"))
    n_shared  = len(merged)
    n_mat_only = len(df_mat) - n_shared
    n_py_only  = len(df_py)  - n_shared

    print(_separator("═"))
    print("  ARTwarp Categorization Comparison")
    print(_separator("═"))
    print(f"  MATLAB file : {matlab_path.name}  ({len(df_mat)} contours)")
    print(f"  Python file : {python_path.name}  ({len(df_py)} contours)")
    print(f"  Shared      : {n_shared} contours")
    if n_mat_only:
        print(f"  MATLAB only : {n_mat_only} contours  (not compared)")
    if n_py_only:
        print(f"  Python only : {n_py_only} contours  (not compared)")

    if n_shared == 0:
        print("\n  ✗  No shared contour names found — check file paths / name format.")
        sys.exit(1)

    labels_mat = merged["category_mat"].to_numpy(dtype=int)
    labels_py  = merged["category_py"].to_numpy(dtype=int)

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------
    n_cats_mat = len(np.unique(labels_mat))
    n_cats_py  = len(np.unique(labels_py))
    avg_match_mat = merged["match_mat"].mean()
    avg_match_py  = merged["match_py"].mean()

    print()
    print(_separator())
    print("  OVERVIEW")
    print(_separator())
    print(f"  {'':30s}  {'MATLAB':>10s}  {'Python':>10s}")
    print(f"  {'Number of categories':30s}  {n_cats_mat:>10d}  {n_cats_py:>10d}")
    print(f"  {'Mean match score (%)':30s}  {avg_match_mat:>10.3f}  {avg_match_py:>10.3f}")
    avg_size_mat = n_shared / n_cats_mat
    avg_size_py  = n_shared / n_cats_py
    print(f"  {'Mean category size':30s}  {avg_size_mat:>10.1f}  {avg_size_py:>10.1f}")

    # Singleton counts (categories with exactly 1 member)
    _, mat_counts = np.unique(labels_mat, return_counts=True)
    _, py_counts  = np.unique(labels_py,  return_counts=True)
    sing_mat = int((mat_counts == 1).sum())
    sing_py  = int((py_counts  == 1).sum())
    print(f"  {'Singleton categories':30s}  {sing_mat:>10d}  {sing_py:>10d}")

    # ------------------------------------------------------------------
    # Cluster-agreement metrics
    # ------------------------------------------------------------------
    print()
    print(_separator())
    print("  CLUSTERING AGREEMENT  (label-permutation-invariant)")
    print(_separator())

    if _has_sklearn:
        ari  = adjusted_rand_score(labels_mat, labels_py)
        nmi  = normalized_mutual_info_score(labels_mat, labels_py, average_method="arithmetic")
        fmi  = fowlkes_mallows_score(labels_mat, labels_py)
        hom, com, vms = homogeneity_completeness_v_measure(labels_mat, labels_py)
        pur  = _purity(labels_mat, labels_py)
        bma  = _best_match_accuracy(labels_mat, labels_py)

        print(f"  {'Adjusted Rand Index (ARI)':40s}  {ari:+.4f}   [−1, 1]  higher=better")
        print(f"  {'Normalized Mutual Info (NMI)':40s}  {nmi:.4f}   [ 0, 1]  higher=better")
        print(f"  {'Fowlkes-Mallows Score (FMI)':40s}  {fmi:.4f}   [ 0, 1]  higher=better")
        print(f"  {'Purity (Python vs MATLAB)':40s}  {pur:.4f}   [ 0, 1]  higher=better")
        print(f"  {'Homogeneity':40s}  {hom:.4f}   [ 0, 1]  higher=better")
        print(f"  {'Completeness':40s}  {com:.4f}   [ 0, 1]  higher=better")
        print(f"  {'V-measure':40s}  {vms:.4f}   [ 0, 1]  higher=better")
        print(f"  {'Best-match accuracy (Hungarian)':40s}  {bma:.4f}   [ 0, 1]  higher=better")

        # Interpretation
        print()
        print("  Interpretation:")
        if ari >= 0.80:
            verdict = "Excellent agreement — the two clusterings are nearly identical."
        elif ari >= 0.60:
            verdict = "Good agreement — major structure is shared, minor differences exist."
        elif ari >= 0.40:
            verdict = "Moderate agreement — similar broad groupings but notable differences."
        elif ari >= 0.20:
            verdict = "Weak agreement — some shared structure but significant divergence."
        else:
            verdict = "Poor agreement — clusterings differ substantially."
        print(f"  ARI={ari:+.4f} → {verdict}")
    else:
        print("  ✗  scikit-learn not installed.  Run: pip install scikit-learn")
        print("     Falling back to purity (no sklearn required).")
        pur = _purity(labels_mat, labels_py)
        print(f"  {'Purity (Python vs MATLAB)':40s}  {pur:.4f}   [ 0, 1]")

    # ------------------------------------------------------------------
    # Per-MATLAB-category purity breakdown (top impure categories)
    # ------------------------------------------------------------------
    print()
    print(_separator())
    print("  PER-MATLAB-CATEGORY PURITY  (Python cluster agreement)")
    print(_separator())

    cat_purities = []
    for mat_cat in np.unique(labels_mat):
        mask = labels_mat == mat_cat
        n_in = int(mask.sum())
        py_in = labels_py[mask]
        py_counts_here = np.bincount(py_in - py_in.min() if py_in.min() < 0 else py_in)
        # count using pandas for simplicity
        series = pd.Series(py_in)
        dominant_py = int(series.mode()[0])
        dominant_n  = int((py_in == dominant_py).sum())
        purity_here = dominant_n / n_in
        n_py_cats   = int(series.nunique())
        cat_purities.append((mat_cat, n_in, purity_here, dominant_py, n_py_cats))

    # Sort by purity ascending (most impure first)
    cat_purities.sort(key=lambda x: x[2])

    print(f"  {'MATLAB cat':>10s}  {'size':>6s}  {'purity':>7s}  {'dominant Py cat':>16s}  {'# Py cats':>10s}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*16}  {'-'*10}")

    show_n = min(20, len(cat_purities))
    for mat_cat, n_in, pur_here, dom_py, n_py in cat_purities[:show_n]:
        bar = "▓" * int(pur_here * 10) + "░" * (10 - int(pur_here * 10))
        print(
            f"  {mat_cat:>10d}  {n_in:>6d}  {pur_here:>6.1%}  "
            f"{dom_py:>16d}  {n_py:>10d}  {bar}"
        )
    if len(cat_purities) > show_n:
        remaining_pure = [p[2] for p in cat_purities[show_n:]]
        print(
            f"  ... {len(cat_purities) - show_n} more MATLAB categories "
            f"(all purity ≥ {min(remaining_pure):.1%})"
        )

    # ------------------------------------------------------------------
    # Same-pair agreement: fraction of pairs that co-cluster in BOTH
    # ------------------------------------------------------------------
    print()
    print(_separator())
    print("  PAIR-LEVEL AGREEMENT")
    print(_separator())

    # Sample for speed if dataset is large
    n = len(labels_mat)
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 2000, replace=False)
        lm, lp = labels_mat[idx], labels_py[idx]
        sample_note = f" (sampled 2000/{n} contours for speed)"
    else:
        lm, lp = labels_mat, labels_py
        sample_note = ""

    same_mat = lm[:, None] == lm[None, :]   # pairs in same MATLAB category
    same_py  = lp[:, None] == lp[None, :]   # pairs in same Python category
    triu = np.triu_indices(len(lm), k=1)
    sm = same_mat[triu]
    sp = same_py[triu]

    both_same  = int((sm & sp).sum())
    both_diff  = int((~sm & ~sp).sum())
    mat_only   = int((sm & ~sp).sum())
    py_only    = int((~sm & sp).sum())
    total_pairs = len(sm)

    agreement_pct = 100.0 * (both_same + both_diff) / total_pairs

    print(f"  Contours compared{sample_note}: {len(lm)}")
    print(f"  Total contour pairs evaluated: {total_pairs:,}")
    print(f"  Both systems agree (same cat) : {both_same:,}  ({100*both_same/total_pairs:.1f}%)")
    print(f"  Both systems agree (diff cat) : {both_diff:,}  ({100*both_diff/total_pairs:.1f}%)")
    print(f"  MATLAB same / Python different: {mat_only:,}  ({100*mat_only/total_pairs:.1f}%)  ← MATLAB merged, Python split")
    print(f"  Python same / MATLAB different: {py_only:,}  ({100*py_only/total_pairs:.1f}%)  ← Python merged, MATLAB split")
    print(f"  Overall pair agreement        : {agreement_pct:.2f}%")

    print()
    print(_separator("═"))
    print("  Done.")
    print(_separator("═"))


if __name__ == "__main__":
    main()
