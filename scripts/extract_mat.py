#!/usr/bin/env python3
"""
Extract category assignments and weight matrix from an ARTwarp MATLAB .mat file.

Outputs two CSV files next to the input .mat file (or in a specified output
directory):

    <stem>_assignments.csv   contour_name, category, match
    <stem>_weights.csv       one column per category prototype (cat_1 … cat_N)

The assignments CSV is format-compatible with artwarp-py's
--export-categories output (e.g. category_assignments_sarasota.csv).

MATLAB match scores are stored as fractions in [0, 1]; this script converts
them to percentages (×100) to match the artwarp-py convention.  If the values
are already in percentage range (>2.0), no conversion is applied.

Usage
-----
From the artwarp-py root directory:

    python scripts/extract_mat.py <path/to/file.mat> [--output-dir <dir>]

Examples
--------
    # outputs alongside the .mat file
    python scripts/extract_mat.py scripts/ARTwarp96FINAL.mat

    # outputs into a specific directory
    python scripts/extract_mat.py scripts/ARTwarp96FINAL.mat --output-dir scripts/sarasota/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from artwarp.io.loaders import load_mat_categorization


def _separator(char: str = "─", width: int = 60) -> str:
    return char * width


def extract(mat_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = mat_path.stem

    print(_separator("═"))
    print(f"  extract_mat  —  {mat_path.name}")
    print(_separator("═"))

    d = load_mat_categorization(str(mat_path))

    # ------------------------------------------------------------------
    # Network parameters
    # ------------------------------------------------------------------
    print()
    print("  NET (network parameters)")
    print(_separator())
    print(f"  {'num_categories':<22}: {d['num_categories']}")
    print(f"  {'vigilance':<22}: {d['vigilance']}")
    print(f"  {'learning_rate':<22}: {d['learning_rate']}")
    print(f"  {'bias':<22}: {d['bias']}")
    print(f"  {'max_num_categories':<22}: {d.get('max_num_categories', 'n/a')}")
    print(f"  {'max_num_iterations':<22}: {d.get('max_num_iterations', 'n/a')}")
    print(f"  {'weight_matrix shape':<22}: {d['weight_matrix'].shape}")

    # ------------------------------------------------------------------
    # Category assignments CSV
    # ------------------------------------------------------------------
    print()
    assignments_path = output_dir / f"{stem}_assignments.csv"
    if d.get("categories") is not None:
        names = d.get(
            "contour_names",
            [f"contour_{i}" for i in range(len(d["categories"]))],
        )
        # MATLAB categories are 0-based after load_mat_categorization; convert
        # back to 1-based to match MATLAB convention used in comparison scripts
        cats = np.asarray(d["categories"]).ravel() + 1

        if d.get("matches") is not None:
            raw = np.asarray(d["matches"]).ravel()
            # convert fraction → percentage if stored as [0, 1]
            matches = np.round(raw * 100 if raw.max() <= 2.0 else raw, 3)
        else:
            matches = [None] * len(cats)

        df = pd.DataFrame(
            {
                "contour_name": names,
                "category": cats.astype(int),
                "match": matches,
            }
        )
        df.to_csv(assignments_path, index=False)
        print(f"  Saved assignments  → {assignments_path}  ({len(df)} rows)")
        print(f"  Categories range   : {int(cats.min())} – {int(cats.max())}")
        if d.get("matches") is not None:
            print(f"  Match score range  : {float(np.min(matches)):.3f} – {float(np.max(matches)):.3f} %")
    else:
        print("  No DATA block found — assignments CSV not written.")

    # ------------------------------------------------------------------
    # Weight matrix CSV
    # ------------------------------------------------------------------
    wm = d["weight_matrix"]
    weights_path = output_dir / f"{stem}_weights.csv"
    wm_df = pd.DataFrame(
        wm, columns=[f"cat_{i + 1}" for i in range(wm.shape[1])]
    )
    wm_df.to_csv(weights_path, index=False)
    print(f"  Saved weight matrix → {weights_path}  (shape {wm.shape})")

    print()
    print(_separator("═"))
    print("  Done.")
    print(_separator("═"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract assignments and weights from an ARTwarp .mat file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "mat_file",
        type=Path,
        help="Path to the ARTwarp .mat file (e.g. scripts/ARTwarp96FINAL.mat)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where CSV files are written.  "
            "Defaults to the same directory as the .mat file."
        ),
    )
    args = parser.parse_args()

    mat_path: Path = args.mat_file.resolve()
    if not mat_path.exists():
        print(f"Error: file not found: {mat_path}", file=sys.stderr)
        sys.exit(1)

    output_dir: Path = args.output_dir.resolve() if args.output_dir else mat_path.parent
    extract(mat_path, output_dir)


if __name__ == "__main__":
    main()
