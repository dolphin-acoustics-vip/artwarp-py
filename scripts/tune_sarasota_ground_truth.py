#!/usr/bin/env python3
"""
Tune vigilance and DTW warp-factor against Sarasota-style ground truth (dolphin ID).

Sarasota naming convention (1800 contours = 60 dolphins × 30 whistles each):
    F109-2001-SW-IND63104_ROCCA
The **dolphin** label is the first hyphen-separated token (here ``F109``).  There are
60 unique such tokens; each appears exactly 30 times.

This script:
  1. Optionally writes ``sarasota_ground_truth.csv`` (contour_name, dolphin_id,
     dolphin_index) for reuse / inspection.
  2. Loads contours, resamples to a fixed interval (default 0.01 s), matching
     typical ``artwarp-py train --resample`` behaviour.
  3. Runs **coordinate-ascent** search starting at vigilance **95** and warp **3**:
     evaluates the current point and axis neighbours (vigilance ± step, warp ± 1),
     moves to the neighbour with highest **NMI** (primary objective), and repeats
     until no improvement or the evaluation budget is exhausted.  The vigilance
     step halves after a stagnant iteration (down to a minimum of 0.25).

Metrics (same family as ``compare_categorizations.py``; **ground truth** = dolphin,
**prediction** = ARTwarp category index):
  - **NMI** (primary) — ``normalized_mutual_info_score`` (arithmetic mean)
  - ARI, Fowlkes–Mallows, homogeneity, completeness, V-measure
  - Purity of predicted clusters w.r.t. dolphins (same definition as compare script)

Requires **scikit-learn** (``pip install scikit-learn``).

Usage (from ``artwarp-py/`` root, with conda/venv that has artwarp + sklearn):

    # default contour dir: ../contours-sarasota (sibling of artwarp-py under repo root)
    python scripts/tune_sarasota_ground_truth.py

    python scripts/tune_sarasota_ground_truth.py \\
        --contour-dir /path/to/contours-sarasota \\
        --output-csv scripts/sarasota/tuning_runs.csv \\
        --ground-truth-csv scripts/sarasota/sarasota_ground_truth.csv \\
        --eval-budget 25

    # only emit ground-truth CSV (no training)
    python scripts/tune_sarasota_ground_truth.py --ground-truth-only \\
        --ground-truth-csv scripts/sarasota/sarasota_ground_truth.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from sklearn.metrics import (
        adjusted_rand_score,
        fowlkes_mallows_score,
        homogeneity_completeness_v_measure,
        normalized_mutual_info_score,
    )
except ImportError:  # pragma: no cover
    print(
        "Error: scikit-learn is required. Install with: pip install scikit-learn",
        file=sys.stderr,
    )
    sys.exit(1)

from artwarp import ARTwarp, load_contours
from artwarp.utils.resample import resample_contours


# ---------------------------------------------------------------------------
# Ground truth from filenames
# ---------------------------------------------------------------------------


def strip_extension(name: str) -> str:
    for ext in (".ctr", ".csv", ".txt"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def dolphin_id_from_name(contour_name: str) -> str:
    """
    First token before '-' (e.g. F109 from F109-2001-SW-IND....).
    """
    base = strip_extension(contour_name.strip())
    base = base.replace("_ROCCA", "")
    token = base.split("-", 1)[0]
    return token


def build_ground_truth(names: List[str]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Returns:
        y_true: int array shape (n,), values 0 .. n_dolphins-1
        dolphin_ids: sorted list of dolphin id strings (index = label)
        dataframe for CSV export
    """
    ids = [dolphin_id_from_name(n) for n in names]
    unique_sorted = sorted(set(ids))
    id_to_idx = {d: i for i, d in enumerate(unique_sorted)}
    y = np.array([id_to_idx[d] for d in ids], dtype=np.int64)
    df = pd.DataFrame(
        {
            "contour_name": names,
            "dolphin_id": ids,
            "dolphin_index": y,
        }
    )
    return y, unique_sorted, df


# ---------------------------------------------------------------------------
# Metrics (aligned with compare_categorizations philosophy)
# ---------------------------------------------------------------------------


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


def supervised_clustering_metrics(
    y_true: np.ndarray, categories: np.ndarray
) -> Dict[str, Any]:
    """
    categories: float array from TrainingResults; may contain NaN.
    """
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


def _safe_tempres(t: Optional[float], default: float) -> float:
    if t is None:
        return default
    v = float(t)
    return default if v <= 0 else v


def load_resampled_contours(
    contour_dir: str,
    file_format: str,
    sample_interval: float,
    default_tempres: float,
) -> Tuple[List[np.ndarray], List[str], List[float]]:
    contours, names, tempres_list = load_contours(
        contour_dir,
        file_format=file_format,
        return_tempres=True,
    )
    tempres_floats = [_safe_tempres(t, default_tempres) for t in tempres_list]
    resampled = resample_contours(contours, tempres_floats, sample_interval)
    return resampled, names, tempres_floats


def train_once(
    contours: List[np.ndarray],
    names: List[str],
    vigilance: float,
    warp_factor: int,
    max_categories: int,
    max_iterations: int,
    learning_rate: float,
    bias: float,
    seed: int,
    verbose: bool,
) -> Tuple[Any, float]:
    """Returns (TrainingResults, elapsed_seconds)."""
    np.random.seed(seed)
    net = ARTwarp(
        vigilance=vigilance,
        learning_rate=learning_rate,
        bias=bias,
        max_categories=max_categories,
        max_iterations=max_iterations,
        warp_factor_level=warp_factor,
        random_seed=seed,
        verbose=verbose,
    )
    t0 = time.perf_counter()
    results = net.fit(contours, contour_names=names)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def clamp_vigilance(v: float) -> float:
    return float(min(99.0, max(1.0, v)))


def clamp_warp(w: int, w_min: int, w_max: int) -> int:
    return int(min(w_max, max(w_min, w)))


def coordinate_ascent_tune(
    contours: List[np.ndarray],
    names: List[str],
    y_true: np.ndarray,
    start_v: float,
    start_w: int,
    v_step_init: float,
    v_step_min: float,
    w_min: int,
    w_max: int,
    eval_budget: int,
    max_categories: int,
    max_iterations: int,
    learning_rate: float,
    bias: float,
    seed: int,
    verbose_train: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Greedy hill-climb on (vigilance, warp) maximising NMI.
    """
    cache_d: Dict[Tuple[float, int], Dict[str, Any]] = {}

    rows: List[Dict[str, Any]] = []
    eval_count = 0

    def evaluate(v: float, w: int) -> Dict[str, Any]:
        nonlocal eval_count
        v_key = round(v, 4)
        key = (v_key, w)
        if key in cache_d:
            return cache_d[key]
        if eval_count >= eval_budget:
            return {}
        eval_count += 1
        results, elapsed = train_once(
            contours,
            names,
            vigilance=v,
            warp_factor=w,
            max_categories=max_categories,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            bias=bias,
            seed=seed,
            verbose=verbose_train,
        )
        m = supervised_clustering_metrics(y_true, results.categories)
        row = {
            "run_index": eval_count,
            "vigilance": v_key,
            "warp_factor": w,
            "num_categories": results.num_categories,
            "train_seconds": round(elapsed, 3),
            **{k: m[k] for k in m},
        }
        rows.append(row)
        cache_d[key] = row
        print(
            f"  [{eval_count}/{eval_budget}]  v={v_key:g}  w={w}  "
            f"NMI={m['nmi']:.4f}  ARI={m['ari']:.4f}  "
            f"cats={results.num_categories}  time={elapsed:.1f}s"
        )
        return row

    current_v = clamp_vigilance(start_v)
    current_w = clamp_warp(start_w, w_min, w_max)
    v_step = v_step_init

    best_row = evaluate(current_v, current_w)
    if not best_row:
        return pd.DataFrame(rows), {}

    stagnant = 0
    while eval_count < eval_budget:
        candidates: List[Tuple[float, int]] = []
        # vigilance neighbours
        for dv in (-v_step, v_step):
            nv = clamp_vigilance(current_v + dv)
            if abs(nv - current_v) > 1e-9:
                candidates.append((nv, current_w))
        # warp neighbours
        for dw in (-1, 1):
            nw = clamp_warp(current_w + dw, w_min, w_max)
            if nw != current_w:
                candidates.append((current_v, nw))

        best_nmi = float(best_row["nmi"])
        best_candidate: Optional[Dict[str, Any]] = None

        for nv, nw in candidates:
            if eval_count >= eval_budget:
                break
            r = evaluate(nv, nw)
            if not r:
                break
            if r["nmi"] > best_nmi + 1e-12:
                best_nmi = r["nmi"]
                best_candidate = r

        if best_candidate is not None:
            current_v = float(best_candidate["vigilance"])
            current_w = int(best_candidate["warp_factor"])
            best_row = best_candidate
            stagnant = 0
            continue

        # no neighbour improved — shrink vigilance step or stop
        stagnant += 1
        if v_step > v_step_min + 1e-9:
            v_step = max(v_step_min, v_step / 2.0)
            print(f"  (no improvement; reducing vigilance step to {v_step:g})")
            stagnant = 0
            continue
        print("  Local optimum (NMI) reached at minimum vigilance step.")
        break

    summary = {
        "best_vigilance": float(best_row["vigilance"]),
        "best_warp_factor": int(best_row["warp_factor"]),
        "best_nmi": float(best_row["nmi"]),
        "best_ari": float(best_row["ari"]),
        "best_v_measure": float(best_row["v_measure"]),
        "best_num_categories": int(best_row["num_categories"]),
        "total_evaluations": eval_count,
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    default_contours = root.parent / "contours-sarasota"

    p = argparse.ArgumentParser(
        description="Tune vigilance & warp factor against Sarasota dolphin ground truth."
    )
    p.add_argument(
        "--contour-dir",
        type=Path,
        default=default_contours,
        help="Directory of contour files (default: ../contours-sarasota from artwarp-py)",
    )
    p.add_argument(
        "--format",
        default="csv",
        choices=("csv", "ctr", "txt"),
        help="Contour file format (default: csv)",
    )
    p.add_argument("--sample-interval", type=float, default=0.01, help="Resample interval (s)")
    p.add_argument(
        "--tempres",
        type=float,
        default=0.01,
        help="Fallback temporal resolution when missing from file (default: 0.01)",
    )
    p.add_argument("--max-categories", type=int, default=1800)
    p.add_argument("--max-iterations", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--bias", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-vigilance", type=float, default=95.0)
    p.add_argument("--start-warp", type=int, default=3)
    p.add_argument("--warp-min", type=int, default=2)
    p.add_argument("--warp-max", type=int, default=6)
    p.add_argument("--vigilance-step", type=float, default=1.0, help="Initial ± step for vigilance")
    p.add_argument("--vigilance-step-min", type=float, default=0.25)
    p.add_argument("--eval-budget", type=int, default=20, help="Max training runs")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Append/write all evaluation rows (default: scripts/sarasota/tuning_runs.csv)",
    )
    p.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=None,
        help="Write ground-truth table (default: scripts/sarasota/sarasota_ground_truth.csv)",
    )
    p.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Only write ground-truth CSV and exit (no training)",
    )
    p.add_argument(
        "--verbose-train",
        action="store_true",
        help="Print full ARTwarp training logs for each run",
    )
    args = p.parse_args()

    contour_dir = args.contour_dir.resolve()
    out_default_dir = Path(__file__).resolve().parent / "sarasota"
    out_default_dir.mkdir(parents=True, exist_ok=True)
    gt_path = (
        args.ground_truth_csv
        if args.ground_truth_csv is not None
        else out_default_dir / "sarasota_ground_truth.csv"
    )
    runs_path = (
        args.output_csv
        if args.output_csv is not None
        else out_default_dir / "tuning_runs.csv"
    )

    if not contour_dir.is_dir():
        print(f"Error: contour directory not found: {contour_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading contours from: {contour_dir}")
    contours, names, _ = load_resampled_contours(
        str(contour_dir),
        file_format=args.format,
        sample_interval=args.sample_interval,
        default_tempres=args.tempres,
    )
    n = len(contours)
    print(f"Loaded and resampled {n} contours (interval={args.sample_interval}s)")

    y_true, dolphin_ids, gt_df = build_ground_truth(names)
    gt_df.to_csv(gt_path, index=False)
    print(f"Wrote ground truth ({len(dolphin_ids)} dolphins) -> {gt_path}")

    vc = gt_df["dolphin_id"].value_counts()
    if vc.min() != vc.max():
        print(
            "Warning: whistles per dolphin are not uniform:",
            f"min={vc.min()}, max={vc.max()}",
            file=sys.stderr,
        )
    elif len(dolphin_ids) != 60 or vc.iloc[0] != 30:
        print(
            f"Note: expected 60×30 layout; got {len(dolphin_ids)} dolphins, "
            f"{vc.iloc[0]} per dolphin (first count).",
        )

    if args.ground_truth_only:
        print("Ground-truth-only mode; done.")
        return

    print()
    print("=== Coordinate-ascent tuning (maximise NMI vs dolphin labels) ===")
    print(
        f"  start: vigilance={args.start_vigilance}, warp={args.start_warp}, "
        f"budget={args.eval_budget}"
    )
    print()

    df_runs, summary = coordinate_ascent_tune(
        contours,
        names,
        y_true,
        start_v=args.start_vigilance,
        start_w=args.start_warp,
        v_step_init=args.vigilance_step,
        v_step_min=args.vigilance_step_min,
        w_min=args.warp_min,
        w_max=args.warp_max,
        eval_budget=args.eval_budget,
        max_categories=args.max_categories,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        bias=args.bias,
        seed=args.seed,
        verbose_train=args.verbose_train,
    )

    df_runs.to_csv(runs_path, index=False)
    print()
    print(f"Saved run log -> {runs_path}")
    print()
    print("=== Best (by NMI) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
