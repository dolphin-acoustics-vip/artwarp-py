#!/usr/bin/env python3
"""
MATLAB/artwarp-py parity analysis — deterministic (no-randomisation) mode.

Compares an artwarp-py run (``--no-rand``, vigilance=96, warp=3, 100 iterations)
against four per-iteration MATLAB snapshots obtained from a reference run in
which the randomised sample-order was disabled (contours processed in
alphabetical filename order on every iteration).

Algorithmic equivalence conditions under ``--no-rand``
-------------------------------------------------------
MATLAB ``ARTwarp_Run_Categorisation.m`` computes the permutation **once**
before the outer loop:

    [~, sortedRandom] = sort(randn(numSamples, 1));

When "no randomisation" is applied the effective order is ``1:numSamples``
(the order MATLAB's ``dir()`` returns files, which is alphabetical).
The **same** order is then used for every iteration.

artwarp-py with ``randomize_sample_order=False`` pre-computes:

    name_sorted_indices = np.argsort(contour_names, kind="stable")

and reuses it for every iteration — structurally identical to the MATLAB
no-randomisation mode.

Per-iteration snapshots
-----------------------
Training captures Python state at ``--snapshot-iters`` (default ``25,50,75,100``)
via ``fit(snapshot_iterations=...)``.  Each snapshot is written to
``artwarp_py_it{NNN}_categories.csv`` and compared against the matching MATLAB
``ARTwarp96it{NNN}.csv`` at the **same** iteration (not final-vs-intermediate).

Metrics
-------
All comparisons are label-permutation-invariant:

  - NMI   — normalized mutual information between Python and MATLAB label vectors
  - ARI   — adjusted Rand index
  - exact  — fraction of contours in the same category (after Hungarian matching)
  - match_corr — Pearson r between per-contour match scores (Python vs MATLAB)
  - cats  — number of categories

Ground truth (dolphin ID labels from ``scripts/sarasota/sarasota_ground_truth.csv``):
  - NMI, ARI, V-measure, purity for Python and MATLAB separately at each snapshot
    (same metrics as ``tune_sarasota_ground_truth.py``; ground truth = dolphin,
    prediction = ARTwarp category)

Usage (from artwarp-py/ root; use the ``sig-process`` conda env for Numba):

    conda activate sig-process
    python scripts/parity_matlab_norand.py \\
        --matlab-dir /path/to/sarasota-1800-results-NO-RANDOMISATION \\
        [--contour-dir data/contours-sarasota] \\
        [--output-dir scripts/sarasota/parity] \\
        [--sample-interval 0.01 --override-tempres 0.005] \\
        [--snapshot-iters 25,50,75,100] \\
        [--no-train]   # requires pickle trained with snapshots; re-run without this first time

Resampling must mirror the MATLAB run: ``sampleInterval=0.01`` s with
``tempres=0.005`` s/pt (Rocca ``.ctr`` metadata).  Pass ``--override-tempres
0.005`` because CSV-derived tempres differs slightly from the ``.ctr`` values
MATLAB used.

@author: Pedro Gronda Garrigues (with a bit of help from good ol' Claude)
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import (
        adjusted_rand_score,
        fowlkes_mallows_score,
        homogeneity_completeness_v_measure,
        normalized_mutual_info_score,
    )
except ImportError:
    print(
        "Error: scikit-learn and scipy are required.\n"
        "Install with: pip install scikit-learn scipy",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from artwarp import ARTwarp, load_contours  # noqa: E402
from artwarp.core.network import IterationSnapshot, TrainingResults  # noqa: E402
from artwarp.utils.resample import resample_contours  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sep(ch: str = "─", w: int = 60) -> str:
    return ch * w


def _safe_tempres(t: Optional[float], default: float = 0.01) -> float:
    if t is None:
        return default
    v = float(t)
    return default if v <= 0 else v


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_resampled(
    contour_dir: str,
    sample_interval: float = 0.01,
    default_tempres: float = 0.005,
    override_tempres: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load contours and resample to ``sample_interval`` seconds per point.

    When ``override_tempres`` is set (e.g. 0.005 for MATLAB ``.ctr`` metadata),
    that value is used for every contour instead of per-file timestamps from the
    CSV loader.  This matches the reference MATLAB no-randomisation run, which
    resampled from ``tempres=0.005`` to ``sampleInterval=0.01`` s.
    """
    contours, names, tempres_list = load_contours(
        contour_dir, file_format="csv", return_tempres=True
    )
    if override_tempres is not None:
        tr = [float(override_tempres)] * len(contours)
    else:
        tr = [_safe_tempres(t, default_tempres) for t in tempres_list]
    resampled = resample_contours(contours, tr, sample_interval)
    return resampled, names


def load_matlab_snapshot(csv_path: Path) -> pd.DataFrame:
    """
    Load a MATLAB per-iteration snapshot CSV.

    Expected columns: name, category, match  (1-based category index).
    Returns a DataFrame with ``stem`` (filename without .ctr), ``matlab_cat``
    (0-based), and ``matlab_match`` (percentage — MATLAB already stores in %).
    """
    df = pd.read_csv(csv_path)
    df["stem"] = df["name"].str.replace(r"\.(ctr|csv|txt)$", "", regex=True)
    df["matlab_cat"] = (df["category"] - 1).astype(int)  # 1→0-based
    df["matlab_match"] = df["match"].astype(float)
    return df[["stem", "matlab_cat", "matlab_match"]]


# ---------------------------------------------------------------------------
# Ground truth (dolphin ID) — aligned with tune_sarasota_ground_truth.py
# ---------------------------------------------------------------------------


def strip_extension(name: str) -> str:
    for ext in (".ctr", ".csv", ".txt"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    total = len(labels_true)
    score = 0.0
    for cat in np.unique(labels_pred):
        mask = labels_pred == cat
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels_true[mask])
        score += float(counts.max())
    return score / total


def supervised_clustering_metrics(y_true: np.ndarray, categories: np.ndarray) -> Dict[str, Any]:
    """Ground truth = dolphin index; prediction = ARTwarp category (may contain NaN)."""
    valid = np.isfinite(categories)
    yt = y_true[valid]
    yp = categories[valid].astype(np.int64)
    n_skip = int(np.sum(~valid))
    if len(yt) == 0:
        return {
            "n_evaluated": 0,
            "n_skipped_nan": n_skip,
            "nmi": float("nan"),
            "ari": float("nan"),
            "fmi": float("nan"),
            "homogeneity": float("nan"),
            "completeness": float("nan"),
            "v_measure": float("nan"),
            "purity": float("nan"),
        }
    nmi = normalized_mutual_info_score(yt, yp, average_method="arithmetic")
    ari = adjusted_rand_score(yt, yp)
    fmi = fowlkes_mallows_score(yt, yp)
    hom, comp, vm = homogeneity_completeness_v_measure(yt, yp)
    pur = _purity(yt, yp)
    return {
        "n_evaluated": int(len(yt)),
        "n_skipped_nan": n_skip,
        "nmi": float(nmi),
        "ari": float(ari),
        "fmi": float(fmi),
        "homogeneity": float(hom),
        "completeness": float(comp),
        "v_measure": float(vm),
        "purity": float(pur),
    }


def load_and_verify_ground_truth(
    gt_path: Path,
    contour_names: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load ``sarasota_ground_truth.csv`` and verify 1:1 stem match with ``contour_names``.

    Returns ``y_true`` aligned to ``contour_names`` order (dolphin_index per contour).
    """
    gt_df = pd.read_csv(gt_path)
    required = {"contour_name", "dolphin_id", "dolphin_index"}
    missing = required - set(gt_df.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV missing columns: {sorted(missing)}")

    gt_stems = [strip_extension(str(n)) for n in gt_df["contour_name"]]
    name_stems = [strip_extension(n) for n in contour_names]

    if len(gt_df) != len(contour_names):
        raise ValueError(
            f"Ground truth has {len(gt_df)} rows but dataset has {len(contour_names)} contours"
        )
    if len(set(gt_stems)) != len(gt_stems):
        raise ValueError("Ground-truth CSV contains duplicate contour_name entries")
    if len(set(name_stems)) != len(name_stems):
        raise ValueError("Dataset contains duplicate contour names")

    gt_set = set(gt_stems)
    name_set = set(name_stems)
    if gt_set != name_set:
        only_gt = sorted(gt_set - name_set)[:5]
        only_names = sorted(name_set - gt_set)[:5]
        raise ValueError(
            "Ground-truth stems differ from dataset stems "
            f"(only in GT e.g. {only_gt}; only in dataset e.g. {only_names})"
        )

    lookup = dict(zip(gt_stems, gt_df["dolphin_index"].astype(int)))
    y_true = np.array([lookup[s] for s in name_stems], dtype=np.int64)

    vc = gt_df["dolphin_id"].value_counts()
    n_dolphins = int(gt_df["dolphin_id"].nunique())
    print(
        f"  Ground truth: {len(gt_df)} contours, {n_dolphins} dolphins "
        f"({vc.iloc[0]} whistles each)"
    )
    return y_true, gt_df


def matlab_categories_for_names(matlab_df: pd.DataFrame, contour_names: List[str]) -> np.ndarray:
    """Build category vector (0-based, NaN allowed) in ``contour_names`` order."""
    stem_to_cat = dict(zip(matlab_df["stem"], matlab_df["matlab_cat"].astype(float)))
    out = np.array([stem_to_cat.get(strip_extension(n), np.nan) for n in contour_names])
    return out


def prefix_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Flatten supervised metrics with a key prefix (e.g. ``py_gt_``)."""
    keys = ("nmi", "ari", "fmi", "homogeneity", "completeness", "v_measure", "purity")
    return {f"{prefix}{k}": metrics[k] for k in keys}


# ---------------------------------------------------------------------------
# Category-assignment agreement (label-permutation-invariant)
# ---------------------------------------------------------------------------


def save_py_snapshot_csv(
    path: Path,
    names: List[str],
    snapshot: IterationSnapshot,
) -> None:
    """Write per-contour categories for one iteration (1-based category index)."""
    valid_cats = snapshot.categories.copy()
    # Export 1-based category index to mirror MATLAB CSV convention; NaN stays NaN
    export_cats = np.where(np.isnan(valid_cats), np.nan, valid_cats + 1)
    pd.DataFrame(
        {
            "contour_name": names,
            "category": export_cats,
            "match": snapshot.matches,
            "iteration": snapshot.iteration,
            "num_categories": snapshot.num_categories,
            "num_reclassifications": snapshot.num_reclassifications,
        }
    ).to_csv(path, index=False)


def parse_snapshot_iters(spec: str) -> List[int]:
    """Parse ``25,50,75,100`` into a sorted list of positive integers."""
    iters = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not iters or any(i < 1 for i in iters):
        raise ValueError("snapshot-iters must be a comma-separated list of positive integers")
    return sorted(set(iters))


def get_py_snapshot(results: TrainingResults, iteration: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Return (categories, matches, num_categories) for ``iteration``.

    Uses ``iteration_snapshots`` when available; falls back to final results only
    when the requested iteration equals ``results.num_iterations``.
    """
    snaps = getattr(results, "iteration_snapshots", None) or {}
    if iteration in snaps:
        snap = snaps[iteration]
        return snap.categories, snap.matches, snap.num_categories
    if iteration == results.num_iterations:
        return results.categories, results.matches, results.num_categories
    raise KeyError(
        f"No Python snapshot for iteration {iteration}. "
        f"Available: {sorted(snaps.keys())}. Re-run training with --snapshot-iters."
    )


def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of samples with matching labels after optimal label assignment
    (Hungarian method).  Both label vectors must use non-negative integers.
    """
    n_true = int(y_true.max()) + 1
    n_pred = int(y_pred.max()) + 1
    cost = np.zeros((n_true, n_pred), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cost[int(t), int(p)] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    return float(cost[row_ind, col_ind].sum()) / len(y_true)


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------


def compare_at_snapshot(
    py_categories: np.ndarray,
    py_matches: np.ndarray,
    py_names: List[str],
    matlab_df: pd.DataFrame,
    label: str,
) -> Dict[str, Any]:
    """
    Align Python and MATLAB results by contour name, compute agreement metrics.
    """
    py_stem = [n.replace(".csv", "").replace(".ctr", "").replace(".txt", "") for n in py_names]
    py_df = pd.DataFrame(
        {
            "stem": py_stem,
            "py_cat": py_categories.astype(float),
            "py_match": py_matches,
        }
    )

    merged = pd.merge(py_df, matlab_df, on="stem", how="inner")
    if len(merged) < len(py_stem):
        n_miss = len(py_stem) - len(merged)
        print(
            f"  Warning [{label}]: {n_miss} contour(s) could not be matched by name",
            file=sys.stderr,
        )

    # drop NaN assignments (unassigned contours)
    valid = merged[merged["py_cat"].notna() & merged["matlab_cat"].notna()].copy()
    n_valid = len(valid)
    n_nan_py = int(merged["py_cat"].isna().sum())
    n_nan_matlab = int(merged["matlab_cat"].isna().sum())

    py_labels = valid["py_cat"].astype(int).values
    matlab_labels = valid["matlab_cat"].values

    nmi = float(normalized_mutual_info_score(matlab_labels, py_labels, average_method="arithmetic"))
    ari = float(adjusted_rand_score(matlab_labels, py_labels))
    exact = hungarian_accuracy(matlab_labels, py_labels)

    # match score correlation
    match_corr = float(np.corrcoef(valid["matlab_match"].values, valid["py_match"].values)[0, 1])
    match_diff_mean = float(np.mean(valid["py_match"].values - valid["matlab_match"].values))
    match_diff_std = float(np.std(valid["py_match"].values - valid["matlab_match"].values))

    n_cats_py = int(valid["py_cat"].nunique())
    n_cats_matlab = int(valid["matlab_cat"].nunique())

    return {
        "label": label,
        "n_contours": n_valid,
        "n_nan_py": n_nan_py,
        "n_nan_matlab": n_nan_matlab,
        "parity_nmi": nmi,
        "parity_ari": ari,
        "hungarian_exact": exact,
        "match_corr": match_corr,
        "match_diff_mean": match_diff_mean,
        "match_diff_std": match_diff_std,
        "n_cats_py": n_cats_py,
        "n_cats_matlab": n_cats_matlab,
        # legacy aliases
        "nmi": nmi,
        "ari": ari,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_comparison_figures(
    results: List[Dict[str, Any]],
    matlab_snapshots: Dict[int, int],
    output_dir: Path,
    gt_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if not _HAS_MPL:
        return

    # ── Figure 1: parity metric progression across snapshots ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    iters = [int(r["label"].replace("it", "")) for r in results]
    nmis = [r["parity_nmi"] for r in results]
    aris = [r["parity_ari"] for r in results]
    exacts = [r["hungarian_exact"] for r in results]

    for ax, vals, title, colour in zip(
        axes,
        [nmis, aris, exacts],
        ["NMI (Python vs MATLAB)", "ARI (Python vs MATLAB)", "Hungarian Exact Match"],
        ["steelblue", "darkorange", "seagreen"],
    ):
        ax.plot(iters, vals, marker="o", color=colour, linewidth=2)
        ax.set_xlabel("MATLAB iteration snapshot")
        ax.set_ylabel(title.split(" ")[0])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        for i, v in zip(iters, vals):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(4, 4), fontsize=8)

    fig.suptitle(
        "artwarp-py vs MATLAB (no-randomisation) — matched iteration snapshots",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "parity_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: category count convergence ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ml_iters = sorted(matlab_snapshots.keys())
    ml_cats = [matlab_snapshots[i] for i in ml_iters]
    py_snap_cats = [r["n_cats_py"] for r in results]

    ax.plot(ml_iters, ml_cats, marker="s", color="tomato", linewidth=2, label="MATLAB")
    ax.plot(ml_iters, py_snap_cats, marker="o", color="steelblue", linewidth=2, label="artwarp-py")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of categories")
    ax.set_title("Category count at matched iteration snapshots")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "parity_category_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: ground truth NMI (Python vs MATLAB vs dolphin labels) ──
    if gt_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        giters = [int(r["iteration"]) for r in gt_results]
        ax.plot(
            giters,
            [r["py_gt_nmi"] for r in gt_results],
            marker="o",
            linewidth=2,
            color="steelblue",
            label="artwarp-py",
        )
        ax.plot(
            giters,
            [r["matlab_gt_nmi"] for r in gt_results],
            marker="s",
            linewidth=2,
            color="tomato",
            label="MATLAB",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("NMI vs dolphin ID")
        ax.set_title("Supervised clustering quality (ground truth = dolphin)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "parity_ground_truth_nmi.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 4: match score scatter summary (final snapshot) ──
    final = results[-1]
    fig, ax = plt.subplots(figsize=(5, 5))
    label_last = results[-1]["label"]
    # we need merged data; just show correlation as text
    ax.text(
        0.5,
        0.5,
        f"Match score correlation\n(artwarp-py vs MATLAB at {label_last})\n\n"
        f"r = {final['match_corr']:.4f}\n"
        f"mean diff = {final['match_diff_mean']:+.3f} pp\n"
        f"std  diff = {final['match_diff_std']:.3f} pp",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.9),
    )
    ax.axis("off")
    ax.set_title("Match score agreement")
    fig.tight_layout()
    fig.savefig(output_dir / "parity_match_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    p = argparse.ArgumentParser(
        description=(
            "MATLAB/artwarp-py parity analysis under no-randomisation mode.\n"
            "Trains artwarp-py (or loads an existing pickle) and compares against\n"
            "four MATLAB per-iteration snapshot CSVs."
        )
    )
    p.add_argument(
        "--matlab-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing MATLAB snapshot CSV files:\n"
            "  ARTwarp96it025.csv, ARTwarp96it050.csv, "
            "ARTwarp96it075.csv, ARTwarp96it100.csv"
        ),
    )
    p.add_argument(
        "--contour-dir",
        type=Path,
        default=root / "data" / "contours-sarasota",
        help="Directory of contour CSV files (default: data/contours-sarasota)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "scripts" / "sarasota" / "parity",
        help="Output directory for results and figures",
    )
    p.add_argument("--vigilance", type=float, default=96.0, help="Vigilance (default: 96)")
    p.add_argument("--warp", type=int, default=3, help="Warp factor (default: 3)")
    p.add_argument("--max-iter", type=int, default=100, help="Max iterations (default: 100)")
    p.add_argument(
        "--max-categories", type=int, default=1800, help="Max categories (default: 1800)"
    )
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--bias", type=float, default=1e-6)
    p.add_argument(
        "--sample-interval",
        type=float,
        default=0.01,
        help="Resample interval in seconds (default: 0.01, matching MATLAB GUI 10 ms)",
    )
    p.add_argument(
        "--tempres",
        type=float,
        default=0.005,
        help="Fallback tempres when a file has none (default: 0.005, Rocca .ctr)",
    )
    p.add_argument(
        "--override-tempres",
        type=float,
        default=0.005,
        help=(
            "Force this tempres (s/pt) for every contour when resampling, "
            "instead of CSV-derived values. Default 0.005 matches MATLAB .ctr "
            "metadata. Pass --override-tempres 0 to use per-file tempres."
        ),
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training; load existing pickle from output-dir instead",
    )
    p.add_argument(
        "--pickle",
        type=Path,
        default=None,
        help="Path to existing artwarp-py pickle (overrides --no-train lookup)",
    )
    p.add_argument("--verbose", action="store_true", help="Print full artwarp-py training log")
    p.add_argument(
        "--snapshot-iters",
        type=str,
        default="25,50,75,100",
        help=(
            "Comma-separated training iterations at which to capture Python state "
            "for comparison with MATLAB snapshots (default: 25,50,75,100)"
        ),
    )
    p.add_argument(
        "--matlab-vigilance",
        type=float,
        default=96.0,
        help="Vigilance encoded in MATLAB snapshot filenames (default: 96)",
    )
    p.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=None,
        help=("Dolphin ground-truth labels (default: scripts/sarasota/sarasota_ground_truth.csv)"),
    )

    args = p.parse_args()
    snapshot_iters = parse_snapshot_iters(args.snapshot_iters)

    # ── Resolve paths ────────────────────────────────────────────────────────
    matlab_dir = args.matlab_dir.resolve()
    contour_dir = args.contour_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snap_files = {
        it: matlab_dir / f"ARTwarp{int(args.matlab_vigilance):02d}it{it:03d}.csv"
        for it in snapshot_iters
    }
    for it, fp in snap_files.items():
        if not fp.exists():
            print(f"Error: MATLAB snapshot not found: {fp}", file=sys.stderr)
            sys.exit(1)

    # ── Load MATLAB snapshots ────────────────────────────────────────────────
    print(_sep("═"))
    print("  MATLAB snapshots")
    print(_sep("─"))
    matlab_dfs: Dict[int, pd.DataFrame] = {}
    matlab_snapshots: Dict[int, int] = {}
    for it, fp in snap_files.items():
        df = load_matlab_snapshot(fp)
        matlab_dfs[it] = df
        matlab_snapshots[it] = int(df["matlab_cat"].nunique())
        print(f"  it{it:03d}: {len(df)} contours, {matlab_snapshots[it]} categories")

    # ── Load & resample contours ─────────────────────────────────────────────
    print()
    print(_sep("═"))
    print(f"  Loading contours from: {contour_dir}")
    override_tr = args.override_tempres if args.override_tempres > 0 else None
    contours, names = load_resampled(
        str(contour_dir),
        sample_interval=args.sample_interval,
        default_tempres=args.tempres,
        override_tempres=override_tr,
    )
    tr_note = (
        f"tempres={override_tr}s/pt (forced)"
        if override_tr is not None
        else "tempres=per-file from CSV"
    )
    print(
        f"  Loaded and resampled {len(contours)} contours "
        f"(interval={args.sample_interval}s, {tr_note})"
    )
    if override_tr is not None:
        lens = [len(c) for c in contours]
        print(f"  Resampled length: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")

    # ── Ground truth ─────────────────────────────────────────────────────────
    gt_path = args.ground_truth_csv or (
        Path(__file__).resolve().parent / "sarasota" / "sarasota_ground_truth.csv"
    )
    print()
    print(_sep("═"))
    print(f"  Ground truth: {gt_path}")
    y_true, _gt_df = load_and_verify_ground_truth(gt_path.resolve(), names)

    # ── Train / load artwarp-py ──────────────────────────────────────────────
    pkl_path = args.pickle or (output_dir / f"parity_v{args.vigilance:g}_w{args.warp}.pkl")

    if args.no_train and pkl_path.exists():
        print()
        print(_sep("═"))
        print(f"  Loading existing pickle: {pkl_path}")
        with open(pkl_path, "rb") as fh:
            results_py = pickle.load(fh)
        if results_py.iteration_snapshots:
            for it, snap in sorted(results_py.iteration_snapshots.items()):
                csv_path = output_dir / f"artwarp_py_it{it:03d}_categories.csv"
                if not csv_path.exists():
                    save_py_snapshot_csv(csv_path, names, snap)
                    print(f"  Exported snapshot it{it:03d} → {csv_path.name}")
    else:
        print()
        print(_sep("═"))
        print(
            f"  Training artwarp-py  v={args.vigilance}  w={args.warp}  "
            f"max_iter={args.max_iter}  --no-rand"
        )
        net = ARTwarp(
            vigilance=args.vigilance,
            learning_rate=args.learning_rate,
            bias=args.bias,
            max_categories=args.max_categories,
            max_iterations=args.max_iter,
            warp_factor_level=args.warp,
            randomize_sample_order=False,  # ← the key flag :D
            verbose=args.verbose,
        )
        t0 = time.perf_counter()
        results_py = net.fit(
            contours,
            contour_names=names,
            snapshot_iterations=snapshot_iters,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"  Done in {elapsed:.1f}s  |  "
            f"categories={results_py.num_categories}  "
            f"iterations={results_py.num_iterations}  "
            f"converged={results_py.converged}"
        )
        with open(pkl_path, "wb") as fh:
            pickle.dump(results_py, fh)
        print(f"  Saved pickle → {pkl_path}")
        for it in sorted(results_py.iteration_snapshots.keys()):
            snap = results_py.iteration_snapshots[it]
            csv_path = output_dir / f"artwarp_py_it{it:03d}_categories.csv"
            save_py_snapshot_csv(csv_path, names, snap)
            print(
                f"  Saved Python snapshot it{it:03d} → {csv_path.name}  "
                f"({snap.num_categories} categories, "
                f"{snap.num_reclassifications} reclassifications)"
            )

    # ── Print artwarp-py convergence summary ─────────────────────────────────
    py_history: List[Tuple[int, int]] = results_py.iteration_history
    print()
    print(_sep("─"))
    print("  artwarp-py convergence history (first 10 / last 5 iterations):")
    shown = py_history[:10] + (py_history[-5:] if len(py_history) > 10 else [])
    for it, nreclass in shown:
        print(f"    iter {it:3d}  reclassifications={nreclass}")
    print(
        f"  Final: {results_py.num_categories} categories, "
        f"{results_py.num_iterations} iterations"
    )

    # ── Compare at each snapshot (Python state at iteration N vs MATLAB itN) ──
    print()
    print(_sep("═"))
    print("  Parity comparison at each iteration (matched snapshots)")
    print(_sep("─"))
    comparison_rows: List[Dict] = []
    gt_rows: List[Dict] = []
    for it in sorted(snap_files.keys()):
        try:
            py_cats, py_matches, py_n_cats = get_py_snapshot(results_py, it)
        except KeyError as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            sys.exit(1)
        row = compare_at_snapshot(
            py_categories=py_cats,
            py_matches=py_matches,
            py_names=names,
            matlab_df=matlab_dfs[it],
            label=f"it{it:03d}",
        )
        snap = results_py.iteration_snapshots.get(it)
        row["py_iteration"] = it
        row["py_reclassifications"] = int(snap.num_reclassifications) if snap is not None else None

        matlab_cats = matlab_categories_for_names(matlab_dfs[it], names)
        py_gt = supervised_clustering_metrics(y_true, py_cats)
        matlab_gt = supervised_clustering_metrics(y_true, matlab_cats)
        row.update(prefix_metrics(py_gt, "py_gt_"))
        row.update(prefix_metrics(matlab_gt, "matlab_gt_"))

        gt_row = {
            "iteration": it,
            "py_gt_nmi": py_gt["nmi"],
            "py_gt_ari": py_gt["ari"],
            "py_gt_v_measure": py_gt["v_measure"],
            "py_gt_purity": py_gt["purity"],
            "py_num_categories": py_n_cats,
            "matlab_gt_nmi": matlab_gt["nmi"],
            "matlab_gt_ari": matlab_gt["ari"],
            "matlab_gt_v_measure": matlab_gt["v_measure"],
            "matlab_gt_purity": matlab_gt["purity"],
            "matlab_num_categories": row["n_cats_matlab"],
        }
        gt_rows.append(gt_row)

        comparison_rows.append(row)
        print(
            f"  iter {it:3d}  parity NMI={row['parity_nmi']:.4f}  "
            f"GT NMI py/matlab={py_gt['nmi']:.4f}/{matlab_gt['nmi']:.4f}  "
            f"purity py/matlab={py_gt['purity']:.4f}/{matlab_gt['purity']:.4f}  "
            f"cats py/matlab={py_n_cats}/{row['n_cats_matlab']}"
        )

    # ── Save CSV summary ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(comparison_rows)
    summary_csv = output_dir / "parity_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    gt_df_out = pd.DataFrame(gt_rows)
    gt_csv = output_dir / "parity_ground_truth_summary.csv"
    gt_df_out.to_csv(gt_csv, index=False)
    print()
    print(f"  Saved comparison table      → {summary_csv}")
    print(f"  Saved ground-truth table   → {gt_csv}")

    # ── Figures ──────────────────────────────────────────────────────────────
    make_comparison_figures(comparison_rows, matlab_snapshots, output_dir, gt_results=gt_rows)

    # ── Final printout ───────────────────────────────────────────────────────
    print()
    print(_sep("═", 70))
    final = comparison_rows[-1]
    print("  FINAL PARITY SUMMARY  (matched iteration 100)")
    print(_sep("─", 70))
    print(
        f"  Parity (py vs matlab):  NMI={final['parity_nmi']:.6f}  "
        f"exact={final['hungarian_exact']:.6f}"
    )
    print(
        f"  Ground truth (py):      NMI={final['py_gt_nmi']:.6f}  "
        f"ARI={final['py_gt_ari']:.6f}  purity={final['py_gt_purity']:.6f}"
    )
    print(
        f"  Ground truth (matlab):  NMI={final['matlab_gt_nmi']:.6f}  "
        f"ARI={final['matlab_gt_ari']:.6f}  purity={final['matlab_gt_purity']:.6f}"
    )
    print(f"  Categories:             py={final['n_cats_py']}  matlab={final['n_cats_matlab']}")
    print(f"  Match corr (r):         {final['match_corr']:.6f}")
    print(_sep("═", 70))


if __name__ == "__main__":
    main()

# This script took an obscene amount of time to write.
# If you change it and it breaks, I will be sad and will not forgive you. :'(
