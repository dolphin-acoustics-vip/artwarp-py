"""
Export utilities for ARTwarp results.

Supports exporting:
- Training results (pickle format)
- Reference contours (CSV format)
- Reference contour provenance metadata (CSV format)
- Category assignments (CSV format)

@author: Pedro Gronda Garrigues
"""

import csv
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
from numpy.typing import NDArray

from artwarp.core.network import TrainingResults


def export_results(
    results: TrainingResults,
    filepath: str,
    include_names: bool = True,
    contour_names: Optional[List[str]] = None,
) -> None:
    """
    Export training results to a pickle file.

    Args:
        results: TrainingResults object from network training
        filepath: Output file path (.pkl extension recommended)
        include_names: Whether to include contour names in export
        contour_names: List of contour names (if include_names=True)

    Note:
        The pickle file contains all training results including the weight matrix,
        making it suitable for later prediction or analysis.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "categories": results.categories,
        "matches": results.matches,
        "weight_matrix": results.weight_matrix,
        "num_categories": results.num_categories,
        "num_iterations": results.num_iterations,
        "converged": results.converged,
        "iteration_history": results.iteration_history,
        "training_time": results.training_time,
        "category_parent_names": results.category_parent_names,
    }

    if include_names and contour_names is not None:
        export_data["contour_names"] = contour_names

    with open(output_path, "wb") as f:
        pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def export_reference_contours(
    weight_matrix: NDArray[np.float64],
    output_dir: str,
    prefix: str = "refContour",
    one_based_filenames: bool = True,
) -> None:
    """
    Export reference contours (category prototypes) to individual CSV files.

    Matches MATLAB SaveRefContours.m: one file per category (refContour_1.csv, ...),
    one frequency value per line (%7.1f style).

    Args:
        weight_matrix: Weight matrix from trained network
        output_dir: Output directory for CSV files
        prefix: Filename prefix (default "refContour" for MATLAB compatibility)
        one_based_filenames: If True (default), use 1-based numbering in filenames
            (e.g. refContour_1.csv) to match MATLAB SaveRefContours.m. If False, use 0-based.

    Creates one CSV file per category: {prefix}_{category_num}.csv
    Each file contains one frequency value per line.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_categories = weight_matrix.shape[1]

    for cat_idx in range(num_categories):
        # extract contour (remove NaN padding; MATLAB: oneContour(isnan(oneContour))=[])
        contour = weight_matrix[:, cat_idx]
        valid_mask = ~np.isnan(contour)
        valid_contour = contour[valid_mask]

        if len(valid_contour) == 0:
            continue

        num = (cat_idx + 1) if one_based_filenames else cat_idx
        filename = output_path / f"{prefix}_{num}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            for freq_value in valid_contour:
                writer.writerow([f"{freq_value:7.1f}"])  # MATLAB SaveRefContours: %7.1f


def export_reference_contour_metadata(
    category_parent_names: Dict[int, List[str]],
    filepath: str,
    one_based_categories: bool = False,
) -> Dict[int, str]:
    """
    Export reference contour provenance metadata to a CSV file.

    For each category prototype, assigns a stable UUID (ref_contour_id) and
    records every input contour name that was assigned to that category during
    training (parent_contour_name). The UUIDs are generated fresh each call,
    consistent with MATLAB ARTwarp which regenerates them on every run.

    Mirrors REFCONTOURS.id and REFCONTOURS.parent_ids from ARTwarp (MATLAB)
    stable-1 (post-Feb 2026), stored as CSV instead of .ctr.

    Args:
        category_parent_names: Mapping from 0-based category index to the list of
            contour names assigned to that category.
            Comes directly from TrainingResults.category_parent_names.
        filepath: Output CSV file path.
        one_based_categories: If True, export category column as 1-based to match
            the MATLAB convention.  Default False keeps Python 0-based indices.

    Returns:
        Dict mapping 0-based category index to its generated UUID ref_contour_id,
        allowing callers to cross-reference IDs with other exports.

    Creates a CSV with columns: category, ref_contour_id, parent_contour_name.
    Each row corresponds to one (category, parent contour) pair.
    Categories with no assigned contours are omitted.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    offset = 1 if one_based_categories else 0

    ref_ids: Dict[int, str] = {cat_idx: str(uuid.uuid4()) for cat_idx in category_parent_names}

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "ref_contour_id", "parent_contour_name"])
        for cat_idx in sorted(category_parent_names.keys()):
            ref_id = ref_ids[cat_idx]
            cat_out = cat_idx + offset
            for parent_name in category_parent_names[cat_idx]:
                writer.writerow([cat_out, ref_id, parent_name])

    return ref_ids


def export_category_assignments(
    categories: NDArray[np.float64],
    matches: NDArray[np.float64],
    contour_names: list,
    filepath: str,
    one_based_categories: bool = False,
) -> None:
    """
    Export category assignments to a CSV file.

    Args:
        categories: Category assignments for each contour (0-based indices)
        matches: Match values for each contour
        contour_names: Names of contours
        filepath: Output CSV file path
        one_based_categories: If True, export category as 1-based (MATLAB convention).
            Default False keeps Python 0-based indices.

    Creates a CSV with columns: contour_name, category, match
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    offset = 1 if one_based_categories else 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["contour_name", "category", "match"])

        for name, cat, match in zip(contour_names, categories, matches):
            if np.isnan(cat):
                cat_str = "uncategorized"
            else:
                cat_str = str(int(cat) + offset)
            writer.writerow([name, cat_str, f"{match:.3f}"])


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load training results from a pickle file.

    Args:
        filepath: Path to pickle file created by export_results()

    Returns:
        Dictionary containing training results
    """
    with open(filepath, "rb") as f:
        return cast(Dict[str, Any], pickle.load(f))
