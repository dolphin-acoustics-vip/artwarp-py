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
     Neighbour candidates within each iteration are evaluated **in parallel**
     (``--n-jobs -1`` = all CPU cores, default).

Per-run outputs are saved to ``--runs-dir`` (default: ``scripts/sarasota/runs/``):
  - ``run{N:03d}_v{v}_w{w}.pkl``             — ARTwarp TrainingResults (pickle)
  - ``run{N:03d}_v{v}_w{w}_categories.csv``  — per-contour category assignment
  - ``run{N:03d}_v{v}_w{w}.png``             — cluster-size + metrics figure
A ``tuning_summary.png`` convergence/parameter-space overview is written to the
same directory after all runs complete.

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
        --eval-budget 25 \\
        --n-jobs 4

    # serial execution (no multiprocessing)
    python scripts/tune_sarasota_ground_truth.py --n-jobs 1

    # only emit ground-truth CSV (no training)
    python scripts/tune_sarasota_ground_truth.py --ground-truth-only \\
        --ground-truth-csv scripts/sarasota/sarasota_ground_truth.csv

@author: Pedro Gronda Garrigues
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

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
    """First token before '-' (e.g. F109 from F109-2001-SW-IND....)."""
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
    """categories: float array from TrainingResults; may contain NaN."""
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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


def _train_worker(args: tuple) -> Tuple[float, int, Any, float]:
    """
    Top-level worker function for ProcessPoolExecutor (must be module-level to be
    picklable by the multiprocessing spawner).

    args: (contours, names, vigilance, warp_factor, max_categories,
           max_iterations, learning_rate, bias, seed, verbose)
    Returns: (vigilance, warp_factor, TrainingResults, elapsed_seconds)
    """
    contours, names, v, w, max_categories, max_iterations, lr, bias, seed, verbose = args
    results, elapsed = train_once(
        contours, names, v, w, max_categories, max_iterations, lr, bias, seed, verbose
    )
    return v, w, results, elapsed


# ---------------------------------------------------------------------------
# Per-run artifact saving
# ---------------------------------------------------------------------------


def _run_stem(run_idx: int, v: float, w: int) -> str:
    """Canonical file-stem for run artifacts: run001_v95_w3"""
    return f"run{run_idx:03d}_v{v:g}_w{w}"


def _save_run_figure(
    path: Path,
    results: Any,
    metrics: Dict[str, Any],
    v: float,
    w: int,
) -> None:
    """Two-panel figure: sorted cluster-size distribution + metrics summary."""
    fig, (ax_sizes, ax_metrics) = plt.subplots(1, 2, figsize=(13, 5))

    sizes = sorted(results.get_category_sizes().values(), reverse=True)
    top_n = min(40, len(sizes))
    ax_sizes.bar(
        range(top_n), sizes[:top_n], color="steelblue", edgecolor="none", width=0.8
    )
    ax_sizes.set_xlabel("Category rank (by size, descending)")
    ax_sizes.set_ylabel("Contour count")
    ax_sizes.set_title(f"Cluster sizes  ({results.num_categories} total categories)")
    ax_sizes.set_xlim(-0.5, top_n - 0.5)

    ax_metrics.axis("off")
    lines = [
        f"vigilance   = {v:g}",
        f"warp factor = {w}",
        "",
        f"NMI         = {metrics['nmi']:.4f}",
        f"ARI         = {metrics['ari']:.4f}",
        f"FMI         = {metrics['fmi']:.4f}",
        f"homogeneity = {metrics['homogeneity']:.4f}",
        f"completeness= {metrics['completeness']:.4f}",
        f"V-measure   = {metrics['v_measure']:.4f}",
        f"purity      = {metrics['purity']:.4f}",
        "",
        f"n_categories= {results.num_categories}",
        f"n_evaluated = {metrics['n_evaluated']}",
        f"n_skipped   = {metrics['n_skipped_nan']}",
        f"iterations  = {results.num_iterations}",
        f"converged   = {results.converged}",
    ]
    ax_metrics.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax_metrics.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )
    ax_metrics.set_title("Metrics")

    fig.suptitle(
        f"ARTwarp tuning run  —  v={v:g}, w={w}", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_run_artifacts(
    runs_dir: Path,
    run_idx: int,
    v: float,
    w: int,
    results: Any,
    names: List[str],
    y_true: np.ndarray,
    metrics: Dict[str, Any],
) -> None:
    """Save pickle, categories CSV, and (if matplotlib available) figure."""
    stem = _run_stem(run_idx, v, w)

    with open(runs_dir / f"{stem}.pkl", "wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)

    cat_df = pd.DataFrame(
        {
            "contour_name": names,
            "dolphin_id": [dolphin_id_from_name(n) for n in names],
            "dolphin_index": y_true,
            "category": results.categories,
            "match": results.matches,
        }
    )
    cat_df.to_csv(runs_dir / f"{stem}_categories.csv", index=False)

    if _HAS_MPL:
        _save_run_figure(runs_dir / f"{stem}.png", results, metrics, v, w)


# ---------------------------------------------------------------------------
# Summary figure (all runs)
# ---------------------------------------------------------------------------


def _save_summary_figure(path: Path, df_runs: pd.DataFrame) -> None:
    """Convergence trace and parameter-space overview for all evaluated runs."""
    if df_runs.empty:
        return

    warp_values = sorted(df_runs["warp_factor"].unique())
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    warp_color = {w: palette[i % len(palette)] for i, w in enumerate(warp_values)}

    fig, (ax_conv, ax_space) = plt.subplots(1, 2, figsize=(14, 5))

    # NMI convergence trace coloured by warp factor!
    for warp in warp_values:
        sub = df_runs[df_runs["warp_factor"] == warp].sort_values("run_index")
        ax_conv.plot(
            sub["run_index"],
            sub["nmi"],
            marker="o",
            linewidth=1.5,
            color=warp_color[warp],
            label=f"warp={warp}",
        )
    best_idx = df_runs["nmi"].idxmax()
    best = df_runs.loc[best_idx]
    ax_conv.axvline(
        best["run_index"], linestyle="--", color="crimson", linewidth=1, alpha=0.7
    )
    ax_conv.annotate(
        f"best  NMI={best['nmi']:.4f}\nv={best['vigilance']:g}, w={int(best['warp_factor'])}",
        xy=(best["run_index"], best["nmi"]),
        xytext=(12, -28),
        textcoords="offset points",
        fontsize=9,
        color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson"),
    )
    ax_conv.set_xlabel("Run index")
    ax_conv.set_ylabel("NMI")
    ax_conv.set_title("NMI convergence trace")
    ax_conv.legend(fontsize=9)
    ax_conv.grid(True, alpha=0.3)

    # vigilance vs NMI scatter; bubble size ∝ num_categories
    max_cats = df_runs["num_categories"].max() or 1
    for warp in warp_values:
        sub = df_runs[df_runs["warp_factor"] == warp]
        ax_space.scatter(
            sub["vigilance"],
            sub["nmi"],
            s=sub["num_categories"] / max_cats * 300 + 30,
            c=[warp_color[warp]] * len(sub),
            alpha=0.78,
            edgecolors="none",
            label=f"warp={warp}",
        )
    ax_space.set_xlabel("Vigilance")
    ax_space.set_ylabel("NMI")
    ax_space.set_title("Parameter space  (bubble size ∝ num_categories)")
    ax_space.legend(fontsize=9)
    ax_space.grid(True, alpha=0.3)

    fig.suptitle("Coordinate-ascent tuning summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exhaustive vigilance sweep
# ---------------------------------------------------------------------------


def sweep_tune(
    contours: List[np.ndarray],
    names: List[str],
    y_true: np.ndarray,
    indexed_pairs: List[Tuple[int, float, int]],
    n_total: int,
    max_categories: int,
    max_iterations: int,
    learning_rate: float,
    bias: float,
    seed: int,
    verbose_train: bool,
    runs_dir: Optional[Path] = None,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train each (run_index, vigilance, warp) triple in ``indexed_pairs``.

    ``n_total`` is the full sweep size (used for display only; may be larger
    than ``len(indexed_pairs)`` when resuming a partial run).

    All combinations are submitted to the process pool at once; artifacts are
    saved to ``runs_dir`` as each worker finishes.  The returned DataFrame is
    ordered by run index.
    """
    if not indexed_pairs:
        return pd.DataFrame(), {}

    if runs_dir is not None:
        runs_dir.mkdir(parents=True, exist_ok=True)

    n_new = len(indexed_pairs)
    n_workers = 1 if n_jobs == 1 else min(
        (os.cpu_count() or 1) if n_jobs < 0 else n_jobs, n_new
    )

    print(f"  {n_new} run(s) to evaluate  |  {n_workers} parallel worker(s)")
    print("  Queue:")
    for idx, v, w in indexed_pairs:
        print(f"    [{idx:03d}/{n_total}]  v={v:g}  w={w}")
    print(flush=True)

    worker_args: Dict[int, tuple] = {
        idx: (
            contours, names, v, w,
            max_categories, max_iterations, learning_rate, bias, seed, verbose_train,
        )
        for idx, v, w in indexed_pairs
    }

    rows: List[Dict[str, Any]] = []

    def _process(run_idx: int, v: float, w: int, results: Any, elapsed: float) -> None:
        v_key = round(v, 4)
        m = supervised_clustering_metrics(y_true, results.categories)
        row: Dict[str, Any] = {
            "run_index": run_idx,
            "vigilance": v_key,
            "warp_factor": w,
            "num_categories": results.num_categories,
            "train_seconds": round(elapsed, 3),
            **m,
        }
        rows.append(row)
        print(
            f"  ✓ [{run_idx:03d}/{n_total}]  v={v_key:g}  w={w}  "
            f"NMI={m['nmi']:.4f}  ARI={m['ari']:.4f}  "
            f"cats={results.num_categories}  time={elapsed:.1f}s",
            flush=True,
        )
        if runs_dir is not None:
            save_run_artifacts(runs_dir, run_idx, v_key, w, results, names, y_true, m)

    if n_jobs == 1:
        for idx, v, w in indexed_pairs:
            print(f"\n  → [{idx:03d}/{n_total}]  starting v={v:g}  w={w} ...", flush=True)
            rv, rw, res, elapsed = _train_worker(worker_args[idx])
            _process(idx, rv, rw, res, elapsed)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(_train_worker, args): idx
                for idx, args in worker_args.items()
            }
            print(f"\n  All {n_new} job(s) queued — waiting for results ...\n", flush=True)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                v = next(v for i, v, _ in indexed_pairs if i == idx)
                w = next(w for i, _, w in indexed_pairs if i == idx)
                try:
                    rv, rw, res, elapsed = future.result()
                    _process(idx, rv, rw, res, elapsed)
                except Exception as exc:
                    print(
                        f"  Warning: worker v={v:g}, w={w} raised: {exc}",
                        file=sys.stderr,
                    )

    return pd.DataFrame(rows).sort_values("run_index").reset_index(drop=True), {}


# ---------------------------------------------------------------------------
# Hill-climb coordinate-ascent tuner
# ---------------------------------------------------------------------------


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
    runs_dir: Optional[Path] = None,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Greedy hill-climb on (vigilance, warp) maximising NMI.

    Neighbour candidates within each iteration are evaluated in parallel when
    ``n_jobs != 1``.  ``n_jobs=-1`` uses all available CPU cores (default).
    """
    cache_d: Dict[Tuple[float, int], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    eval_count = 0

    if runs_dir is not None:
        runs_dir.mkdir(parents=True, exist_ok=True)

    def _record(v: float, w: int, results: Any, elapsed: float) -> Dict[str, Any]:
        """Compute metrics, persist artifacts, return the row dict."""
        nonlocal eval_count
        eval_count += 1
        v_key = round(v, 4)
        m = supervised_clustering_metrics(y_true, results.categories)
        row: Dict[str, Any] = {
            "run_index": eval_count,
            "vigilance": v_key,
            "warp_factor": w,
            "num_categories": results.num_categories,
            "train_seconds": round(elapsed, 3),
            **m,
        }
        rows.append(row)
        cache_d[(v_key, w)] = row
        print(
            f"  └─ [{eval_count}/{eval_budget}]  v={v_key:g}  w={w}  "
            f"NMI={m['nmi']:.4f}  ARI={m['ari']:.4f}  "
            f"cats={results.num_categories}  time={elapsed:.1f}s",
            flush=True,
        )
        if runs_dir is not None:
            save_run_artifacts(runs_dir, eval_count, v_key, w, results, names, y_true, m)
        return row

    def _eval_batch(pairs: List[Tuple[float, int]]) -> None:
        """
        Train all (v, w) pairs and record results.  Uses ProcessPoolExecutor
        when n_jobs != 1 and there are multiple pairs to evaluate.
        """
        if not pairs:
            return

        label = "  ".join(f"v={v:g}/w={w}" for v, w in pairs)
        n_par = 1 if (n_jobs == 1 or len(pairs) == 1) else min(
            (os.cpu_count() or 1) if n_jobs < 0 else n_jobs, len(pairs)
        )
        print(
            f"\n  ┌─ launching {len(pairs)} run(s) on {n_par} core(s): {label}",
            flush=True,
        )

        worker_args = [
            (
                contours,
                names,
                v,
                w,
                max_categories,
                max_iterations,
                learning_rate,
                bias,
                seed,
                verbose_train,
            )
            for v, w in pairs
        ]

        if n_jobs == 1 or len(pairs) == 1:
            for args, (v, w) in zip(worker_args, pairs):
                print(f"  │  starting  v={v:g}  w={w} ...", flush=True)
                rv, rw, res, elapsed = _train_worker(args)
                _record(rv, rw, res, elapsed)
            return

        n_workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
        n_workers = min(n_workers, len(pairs))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_pair = {}
            for a, (v, w) in zip(worker_args, pairs):
                print(f"  │  queued    v={v:g}  w={w}", flush=True)
                future_to_pair[pool.submit(_train_worker, a)] = (v, w)
            print(f"  │  (waiting for results ...)", flush=True)
            for future in as_completed(future_to_pair):
                try:
                    rv, rw, res, elapsed = future.result()
                    _record(rv, rw, res, elapsed)
                except Exception as exc:
                    fv, fw = future_to_pair[future]
                    print(
                        f"  Warning: worker v={fv}, w={fw} raised: {exc}",
                        file=sys.stderr,
                    )

    # ---- Starting point ----
    current_v = clamp_vigilance(start_v)
    current_w = clamp_warp(start_w, w_min, w_max)
    print(f"  Evaluating starting point: v={current_v:g}, w={current_w}", flush=True)
    _eval_batch([(current_v, current_w)])
    if (round(current_v, 4), current_w) not in cache_d:
        return pd.DataFrame(rows), {}
    best_row = cache_d[(round(current_v, 4), current_w)]

    v_step = v_step_init
    while eval_count < eval_budget:
        # build axis-neighbour candidate list
        candidates: List[Tuple[float, int]] = []
        for dv in (-v_step, v_step):
            nv = clamp_vigilance(current_v + dv)
            if abs(nv - current_v) > 1e-9:
                candidates.append((nv, current_w))
        for dw in (-1, 1):
            nw = clamp_warp(current_w + dw, w_min, w_max)
            if nw != current_w:
                candidates.append((current_v, nw))

        # eval only uncached candidates, respecting budget
        uncached = [
            (nv, nw)
            for nv, nw in candidates
            if (round(nv, 4), nw) not in cache_d
        ]
        _eval_batch(uncached[: eval_budget - eval_count])

        # select best neighbour (including any previously cached)
        best_nmi = float(best_row["nmi"])
        best_candidate: Optional[Dict[str, Any]] = None
        for nv, nw in candidates:
            r = cache_d.get((round(nv, 4), nw))
            if r is not None and r["nmi"] > best_nmi + 1e-12:
                best_nmi = r["nmi"]
                best_candidate = r

        if best_candidate is not None:
            current_v = float(best_candidate["vigilance"])
            current_w = int(best_candidate["warp_factor"])
            best_row = best_candidate
            continue

        # no improvement — halve vigilance step or stop
        if v_step > v_step_min + 1e-9:
            v_step = max(v_step_min, v_step / 2.0)
            print(f"  (no improvement; reducing vigilance step to {v_step:g})", flush=True)
            continue
        print("  Local optimum (NMI) reached at minimum vigilance step.", flush=True)
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


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
    p.add_argument("--start-vigilance", type=float, default=93.0)
    p.add_argument("--start-warp", type=int, default=3)
    p.add_argument("--warp-min", type=int, default=2)
    p.add_argument("--warp-max", type=int, default=3)
    p.add_argument("--vigilance-step", type=float, default=0.25, help="Initial ± step for vigilance")
    p.add_argument("--vigilance-step-min", type=float, default=0.25)
    p.add_argument("--eval-budget", type=int, default=20, help="Max training runs")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Worker processes for parallel neighbour evaluation. "
            "-1 = all CPU cores (default), 1 = serial, N = exactly N workers."
        ),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write all evaluation rows (default: scripts/sarasota/tuning_runs.csv)",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-run pickle, categories CSV, and figure "
            "(default: scripts/sarasota/runs)"
        ),
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

    # Sweep mode
    p.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Exhaustive grid sweep instead of coordinate ascent. "
            "Evaluates every vigilance in [--sweep-v-min, --sweep-v-max] at "
            "--vigilance-step increments for each warp in [--warp-min, --warp-max]."
        ),
    )
    p.add_argument(
        "--sweep-v-min",
        type=float,
        default=92.0,
        help="Sweep: lowest vigilance to evaluate (default: 92.0)",
    )
    p.add_argument(
        "--sweep-v-max",
        type=float,
        default=96.0,
        help="Sweep: highest vigilance to evaluate (default: 96.0)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Sweep only: skip combinations whose .pkl already exists in --runs-dir, "
            "reload their metrics from disk, then combine old + new results into the "
            "final tuning_runs.csv and summary figure."
        ),
    )

    args = p.parse_args()

    contour_dir = args.contour_dir.resolve()
    sarasota_dir = Path(__file__).resolve().parent / "sarasota"
    sarasota_dir.mkdir(parents=True, exist_ok=True)

    gt_path = args.ground_truth_csv or sarasota_dir / "sarasota_ground_truth.csv"
    runs_path = args.output_csv or sarasota_dir / "tuning_runs.csv"
    runs_dir = args.runs_dir or sarasota_dir / "runs"

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

    n_cpu = os.cpu_count() or 1
    effective_jobs = n_cpu if args.n_jobs < 0 else args.n_jobs
    print()

    if args.sweep:
        # Build the full ordered (run_index, v, w) list for the sweep
        v_arr = np.arange(
            args.sweep_v_min,
            args.sweep_v_max + args.vigilance_step / 2,
            args.vigilance_step,
        )
        v_values = [round(float(v), 4) for v in v_arr]
        warp_values = list(range(args.warp_min, args.warp_max + 1))
        all_indexed: List[Tuple[int, float, int]] = [
            (idx, v, w)
            for idx, (w, v) in enumerate(
                [(w, v) for w in sorted(warp_values) for v in v_values], 1
            )
        ]
        n_total = len(all_indexed)
        v_step_display = round(v_values[1] - v_values[0], 6) if len(v_values) > 1 else 0.0

        mode_label = "RESUMING" if args.resume else "exhaustive grid"
        print(f"=== Vigilance sweep ({mode_label}) ===")
        print(
            f"  vigilance : {v_values[0]:g} → {v_values[-1]:g}  "
            f"step={v_step_display:g}  ({n_total} combinations)"
        )
        print(f"  warp      : {sorted(warp_values)}")
        print(f"  jobs      : {effective_jobs} / {n_cpu} available cores")
        print(f"  output    : {runs_dir}/")
        print()

        # --- Resume: reload completed runs from their pkl files ---
        cached_rows: List[Dict[str, Any]] = []
        to_run: List[Tuple[int, float, int]] = []

        if args.resume:
            runs_dir.mkdir(parents=True, exist_ok=True)
            print("  Scanning for completed runs ...", flush=True)
            for idx, v, w in all_indexed:
                pkl_path = runs_dir / f"{_run_stem(idx, v, w)}.pkl"
                if pkl_path.exists():
                    try:
                        with open(pkl_path, "rb") as fh:
                            results = pickle.load(fh)
                        m = supervised_clustering_metrics(y_true, results.categories)
                        cached_rows.append(
                            {
                                "run_index": idx,
                                "vigilance": v,
                                "warp_factor": w,
                                "num_categories": results.num_categories,
                                "train_seconds": round(results.training_time, 3),
                                **m,
                            }
                        )
                        print(
                            f"  ✓ (cached) [{idx:03d}/{n_total}]  v={v:g}  w={w}  "
                            f"NMI={m['nmi']:.4f}  cats={results.num_categories}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"  Warning: could not load {pkl_path.name}: {exc} — will re-run",
                            file=sys.stderr,
                        )
                        to_run.append((idx, v, w))
                else:
                    to_run.append((idx, v, w))

            n_cached = len(cached_rows)
            print(
                f"\n  {n_cached} cached  |  {len(to_run)} to run\n",
                flush=True,
            )
        else:
            to_run = all_indexed

        # --- Run missing combinations ---
        new_df, _ = sweep_tune(
            contours,
            names,
            y_true,
            indexed_pairs=to_run,
            n_total=n_total,
            max_categories=args.max_categories,
            max_iterations=args.max_iterations,
            learning_rate=args.learning_rate,
            bias=args.bias,
            seed=args.seed,
            verbose_train=args.verbose_train,
            runs_dir=runs_dir,
            n_jobs=args.n_jobs,
        )

        # --- Combine cached + new, sort by run_index ---
        cached_df = pd.DataFrame(cached_rows)
        df_runs = (
            pd.concat([cached_df, new_df], ignore_index=True)
            .sort_values("run_index")
            .reset_index(drop=True)
        )

        # Best summary across all runs
        if df_runs.empty:
            summary: Dict[str, Any] = {}
        else:
            best = df_runs.loc[df_runs["nmi"].idxmax()]
            summary = {
                "best_vigilance": float(best["vigilance"]),
                "best_warp_factor": int(best["warp_factor"]),
                "best_nmi": float(best["nmi"]),
                "best_ari": float(best["ari"]),
                "best_v_measure": float(best["v_measure"]),
                "best_num_categories": int(best["num_categories"]),
                "total_evaluations": len(df_runs),
            }
    else:
        print("=== Coordinate-ascent tuning (maximise NMI vs dolphin labels) ===")
        print(
            f"  start : vigilance={args.start_vigilance}, warp={args.start_warp}, "
            f"budget={args.eval_budget}"
        )
        print(f"  jobs  : {effective_jobs} / {n_cpu} available cores")
        print(f"  output: {runs_dir}/")
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
            runs_dir=runs_dir,
            n_jobs=args.n_jobs,
        )

    df_runs.to_csv(runs_path, index=False)
    print()
    print(f"Saved run log        -> {runs_path}")

    if _HAS_MPL and not df_runs.empty:
        summary_fig = runs_dir / "tuning_summary.png"
        _save_summary_figure(summary_fig, df_runs)
        print(f"Saved summary figure -> {summary_fig}")

    print()
    print("=== Best (by NMI) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
