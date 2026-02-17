"""
Data loading utilities for various contour file formats.

Supports:
- .ctr files (MATLAB format)
- .csv files (CSV format)
- .txt files (tab-delimited format)

@author: Pedro Gronda Garrigues
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pickle


def load_ctr_file(filepath: Path) -> Dict[str, Any]:
    """
    Load a .ctr file (MATLAB format).

    Args:
        filepath: Path to .ctr file

    Returns:
        Dictionary containing:
            - 'contour': Frequency contour array
            - 'tempres': Temporal resolution (if available)
            - 'ctrlength': Contour length in time (if available)

    Note:
        .ctr files are MATLAB .mat files containing frequency contour data.
        They may contain either 'fcontour' or 'freqContour' variables.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required to load .ctr files (MATLAB format)")

    try:
        mat_data = loadmat(str(filepath), squeeze_me=True)

        # try both var names (MATLAB uses either)
        if "fcontour" in mat_data:
            contour = np.array(mat_data["fcontour"], dtype=np.float64)
            if contour.ndim == 0:
                contour = np.array([float(contour)])
        elif "freqContour" in mat_data:
            freq_contour = np.array(mat_data["freqContour"], dtype=np.float64)
            if freq_contour.ndim == 0:
                freq_contour = np.array([float(freq_contour)])
            # drop last element (MATLAB compat)
            contour = freq_contour[:-1] if len(freq_contour) > 1 else freq_contour
        else:
            raise ValueError(f"File {filepath} does not contain 'fcontour' or 'freqContour'")

        # tempres if present
        tempres = mat_data.get("tempres", None)
        if tempres is not None:
            tempres = float(tempres)

        # extract contour length if available
        ctrlength = mat_data.get("ctrlength", None)
        if ctrlength is not None:
            ctrlength = float(ctrlength)
        elif tempres is not None:
            ctrlength = len(contour) * tempres

        return {"contour": contour, "tempres": tempres, "ctrlength": ctrlength}

    except Exception as e:
        raise ValueError(f"Error loading .ctr file {filepath}: {str(e)}")


def load_csv_file(
    filepath: Path, frequency_column: int = 0, skip_header: int = 1
) -> Dict[str, Any]:
    """
    Load a .csv file containing frequency contour data.

    Args:
        filepath: Path to .csv file
        frequency_column: Column index containing frequency values (0-indexed)
        skip_header: Number of header rows to skip

    Returns:
        Dictionary containing:
            - 'contour': Frequency contour array
            - 'tempres': Calculated temporal resolution
            - 'ctrlength': Contour length in time
    """
    try:
        # read csv
        data = pd.read_csv(filepath, header=None, skiprows=skip_header)

        if frequency_column >= len(data.columns):
            raise ValueError(
                f"Frequency column {frequency_column} not found "
                f"(file has {len(data.columns)} columns)"
            )

        # frequency contour
        freq_values = data.iloc[:, frequency_column].values
        contour = np.array(freq_values, dtype=np.float64)

        # drop last (MATLAB compat)
        if len(contour) > 1:
            contour = contour[:-1]

        # temporal resolution (MATLAB style)
        ctrlength = contour[-1] / 1000.0 if len(contour) > 0 else 0.0
        tempres = ctrlength / len(contour) if len(contour) > 0 else 0.0

        return {"contour": contour, "tempres": tempres, "ctrlength": ctrlength}

    except Exception as e:
        raise ValueError(f"Error loading CSV file {filepath}: {str(e)}")


def load_txt_file(
    filepath: Path, frequency_column: int = 0, delimiter: str = "\t"
) -> Dict[str, Any]:
    """
    Load a tab-delimited .txt file containing frequency contour data.

    Args:
        filepath: Path to .txt file
        frequency_column: Column index containing frequency values (0-indexed)
        delimiter: Column delimiter (default: tab)

    Returns:
        Dictionary containing:
            - 'contour': Frequency contour array
            - 'tempres': Calculated temporal resolution
            - 'ctrlength': Contour length in time
    """
    try:
        # read delimited
        data = pd.read_csv(filepath, sep=delimiter, header=None)

        if frequency_column >= len(data.columns):
            raise ValueError(
                f"Frequency column {frequency_column} not found "
                f"(file has {len(data.columns)} columns)"
            )

        # frequency contour
        freq_values = data.iloc[:, frequency_column].values
        contour = np.array(freq_values, dtype=np.float64)

        # drop last (MATLAB compat)
        if len(contour) > 1:
            contour = contour[:-1]

        # temporal resolution
        ctrlength = contour[-1] / 1000.0 if len(contour) > 0 else 0.0
        tempres = ctrlength / len(contour) if len(contour) > 0 else 0.0

        return {"contour": contour, "tempres": tempres, "ctrlength": ctrlength}

    except Exception as e:
        raise ValueError(f"Error loading text file {filepath}: {str(e)}")


def load_contours(
    directory: str,
    file_format: str = "auto",
    frequency_column: int = 0,
    pattern: str = "*",
    return_tempres: bool = False,
) -> Union[
    Tuple[List[NDArray[np.float64]], List[str]],
    Tuple[List[NDArray[np.float64]], List[str], List[Optional[float]]],
]:
    """
    Load all contour files from a directory.

    Args:
        directory: Path to directory containing contour files
        file_format: File format - 'ctr', 'csv', 'txt', or 'auto' (default)
            If 'auto', format is detected from file extension
        frequency_column: Column index for frequency data (for CSV/TXT files)
        pattern: Glob pattern for file matching (default: all files)
        return_tempres: If True, also return a list of temporal resolution
            (seconds per point) per contour; entries may be None if unknown.

    Returns:
        If return_tempres is False: (contours, names).
        If return_tempres is True: (contours, names, tempres_list) where
            tempres_list[i] is seconds per point for contour i, or None.

    Example:
        >>> contours, names = load_contours('./data', file_format='csv')
        >>> contours, names, tempres = load_contours('./data', return_tempres=True)
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # file extensions to look for
    if file_format == "auto":
        extensions = [".ctr", ".csv", ".txt"]
    elif file_format == "ctr":
        extensions = [".ctr"]
    elif file_format == "csv":
        extensions = [".csv"]
    elif file_format == "txt":
        extensions = [".txt"]
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    # find matching files
    files = []
    for ext in extensions:
        files.extend(sorted(dir_path.glob(f"{pattern}{ext}")))

    if len(files) == 0:
        raise ValueError(f"No contour files found in {directory}")

    # load each
    contours = []
    names = []
    tempres_list: List[Optional[float]] = [] if return_tempres else []

    for filepath in files:
        try:
            ext = filepath.suffix.lower()

            if ext == ".ctr":
                data = load_ctr_file(filepath)
            elif ext == ".csv":
                data = load_csv_file(filepath, frequency_column)
            elif ext == ".txt":
                data = load_txt_file(filepath, frequency_column)
            else:
                continue

            contours.append(data["contour"])
            names.append(filepath.stem)
            if return_tempres:
                tempres_list.append(data.get("tempres"))

        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {str(e)}")
            continue

    if len(contours) == 0:
        raise ValueError(f"No valid contour files could be loaded from {directory}")

    if return_tempres:
        return contours, names, tempres_list
    return contours, names


def load_mat_categorisation(filepath: str) -> Dict[str, Any]:
    """
    Load a MATLAB ARTwarp categorisation file (NET and optionally DATA).

    Mirrors the "Load Categorisation" workflow from MATLAB ARTwarp. Use this to
    load a .mat file saved after training (e.g. ARTwarp85FINAL.mat or
    ARTwarp85it001.mat).

    Args:
        filepath: Path to .mat file containing NET and optionally DATA

    Returns:
        Dictionary with keys:
            - weight_matrix: NDArray (max_features, num_categories)
            - num_categories: int
            - max_features: int (numFeatures from NET)
            - vigilance: float
            - bias: float
            - learning_rate: float
            - max_num_categories: int
            - max_num_iterations: int
            And if DATA is present:
            - contours: List[NDArray]
            - categories: NDArray (1-based as in MATLAB, converted to 0-based here)
            - matches: NDArray
            - contour_names: List[str]

    Raises:
        ImportError: If scipy is not installed
        ValueError: If file does not contain NET
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required to load MATLAB .mat files")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    mat = loadmat(str(path), struct_as_record=False, squeeze_me=True)

    if "NET" not in mat:
        raise ValueError(
            f"File {filepath} does not contain 'NET' (not an ARTwarp categorisation file)"
        )

    net = mat["NET"]

    # 0-d (squeezed) or (1,1) struct
    if hasattr(net, "shape") and net.shape == ():
        net = net.item()
    elif hasattr(net, "shape") and net.size == 1:
        net = net.flat[0]

    def _get_field(obj, name: str, default: Any = None) -> Any:
        # record array -> access by dtype.names
        if (
            hasattr(obj, "dtype")
            and hasattr(obj.dtype, "names")
            and obj.dtype.names
            and name in obj.dtype.names
        ):
            val = obj[name]
        elif hasattr(obj, name):
            val = getattr(obj, name)
        else:
            return default
        if hasattr(val, "shape") and val.size == 1 and val.ndim == 0:
            return val.item()
        return val

    weight = np.array(_get_field(net, "weight"), dtype=np.float64)
    if weight.ndim == 1:
        weight = weight.reshape(-1, 1)

    num_features = int(_get_field(net, "numFeatures", weight.shape[0]))
    num_categories = int(_get_field(net, "numCategories", weight.shape[1]))

    out: Dict[str, Any] = {
        "weight_matrix": weight,
        "num_categories": num_categories,
        "max_features": num_features,
        "vigilance": float(_get_field(net, "vigilance", 85.0)),
        "bias": float(_get_field(net, "bias", 0.0)),
        "learning_rate": float(_get_field(net, "learningRate", 0.1)),
        "max_num_categories": int(_get_field(net, "maxNumCategories", 50)),
        "max_num_iterations": int(_get_field(net, "maxNumIterations", 50)),
    }

    if "DATA" not in mat:
        return out

    data = mat["DATA"]
    if hasattr(data, "size") and data.size == 0:
        return out

    # DATA struct => contour, category, match, name, ...
    data = np.atleast_1d(data)
    n = data.size

    contours_list: List[NDArray[np.float64]] = []
    categories_list: List[float] = []
    matches_list: List[float] = []
    names_list: List[str] = []

    for i in range(n):
        row = data.flat[i]
        c = np.array(_get_field(row, "contour", []), dtype=np.float64)
        if c.ndim > 1:
            c = c.ravel()
        contours_list.append(c)
        cat = _get_field(row, "category", np.nan)
        if isinstance(cat, (int, float)) and not np.isnan(cat):
            categories_list.append(float(cat - 1))  # MATLAB 1-based => Python 0-based
        else:
            categories_list.append(np.nan)
        m = _get_field(row, "match", np.nan)
        matches_list.append(float(m) if m is not None and np.isfinite(m) else np.nan)
        name = _get_field(row, "name", "")
        names_list.append(str(name) if name is not None else "")

    out["contours"] = contours_list
    out["categories"] = np.array(categories_list, dtype=np.float64)
    out["matches"] = np.array(matches_list, dtype=np.float64)
    out["contour_names"] = names_list

    return out
