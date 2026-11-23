"""
Creates specifications of displayable charts. Each dictionary represents a different graph.
Used by results.py to display the graphs.

@author: Jaime Barquero Cebrian
"""


from pathlib import Path
from typing import Dict, List, Optional
from pickle import UnpicklingError
import numpy as np

from artwarp.io import runs
from artwarp.io.exporters import load_results


def build_charts(data)-> List[Dict]:
    """
    One dictionary per chart
    
    Parameters:
        data (dict): 
            categories,
            matches,
            weight_matrix,
            num_categories,
            num_iterations,
            converged,
            iteration_history,
            training_time,
            category_parent_names
    """
    iterations, reclassified = zip(*data["iteration_history"])
    cat_size_dic = _get_category_sizes(data)
    categories = cat_size_dic.keys()
    sizes = cat_size_dic.values()

    convergence = {
        "kind": "line",
        "title": "Convergence",
        "desc": "Contours that changed category in each iteration. Zero means converged.",
        "x": list(iterations),
        "x_label": "Iteration",
        "y_label": "Reclassifications",
        "y_format": "int",
        "y_zero": True,
        "series": [
            {"name": "Reclassified", "slot": 0, "values": list(reclassified)},
        ],
    }

    bar_sizes = {
        "kind": "bar", 
        "title": "Category Sizes", 
        "desc": "Size of each category of contours", 
        "categories": [str(cat+1) for cat in categories],
        "x_label": "Categories", 
        "y_label": "Contours", 
        "y_format": "int",
        "series": [
            {"name": "Size", "slot": 0, "values": list(sizes)},
        ]
    }   

    return [convergence, bar_sizes]

def _get_category_sizes(data) -> Dict[int, int]:
    """Get the number of samples in each category."""
    unique_cats, counts = np.unique(
        data["categories"][~np.isnan(data["categories"])], return_counts=True
    )
    return {int(cat): int(count) for cat, count in zip(unique_cats, counts)}

def load_charts(run) -> Optional[List[Dict]]:
    """
    Feeds run information from pkl file into build_charts.

    Parameters:
        run (.json): JSON file from the specific run. Contains information about the run.
    
    Returns:
        build_charts() (List[Dict]): List of chart dictionaries, with data to build each.
           [Convergence,
            Category Sizes
            ]
    """
    results_file = Path(run["path"])/runs.RESULTS_FILE

    try:
        data = load_results(str(results_file))
        if not isinstance(data, dict):
            return None
    except (OSError, UnpicklingError, EOFError):
        # Missing/Unreadable, Corrupted bytes, Truncated file
        return None
    return build_charts(data)

