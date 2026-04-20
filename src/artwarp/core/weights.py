"""
Weight matrix management and update operations for ARTwarp.

This module handles:
- Weight matrix initialization
- Weight updates during learning
- Category addition
- Length adaptation using interpolation

@author: Pedro Gronda Garrigues
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from artwarp.core.dtw import unwarp


def initialize_weight_matrix(max_features: int) -> NDArray[np.float64]:
    """
    Initialize an empty weight matrix.

    Args:
        max_features: Maximum number of features (longest contour length)

    Returns:
        Empty weight matrix of shape (max_features, 0)
    """
    return np.zeros((max_features, 0), dtype=np.float64)


def add_new_category(
    weight_matrix: NDArray[np.float64], input_contour: NDArray[np.float64], max_features: int
) -> NDArray[np.float64]:
    """
    Add a new category to the weight matrix.

    Creates a new column in the weight matrix initialized with the input contour.
    The new category weight is padded with NaN to match max_features length.

    Args:
        weight_matrix: Current weight matrix, shape (max_features, num_categories)
        input_contour: Input contour to initialize new category, shape (n,)
        max_features: Maximum number of features

    Returns:
        Updated weight matrix with shape (max_features, num_categories + 1)

    Note:
        This implements the MATLAB ART_Add_New_Category.m function.
    """
    # new category vector, NaN-padded
    new_category: NDArray[np.float64] = np.full(max_features, np.nan, dtype=np.float64)
    new_category[: len(input_contour)] = input_contour

    # add as new column
    if weight_matrix.shape[1] == 0:
        # first category
        return new_category.reshape(-1, 1)
    else:
        # append to existing
        return np.column_stack([weight_matrix, new_category])


def delete_category(category_index: int, weight_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Delete a category column from the weight matrix.

    Args:
        category_index: 0-based category index to delete
        weight_matrix: Current weight matrix, shape (max_features, num_categories)

    Returns:
        Updated weight matrix with shape (max_features, num_categories - 1)
    """
    if weight_matrix.shape[1] == 0:
        raise ValueError("Cannot delete category from an empty weight matrix")
    if category_index < 0 or category_index >= weight_matrix.shape[1]:
        raise IndexError(
            f"Category index {category_index} out of bounds for "
            f"{weight_matrix.shape[1]} categories"
        )
    return np.delete(weight_matrix, category_index, axis=1)


def delete_category_reindex_assignments(
    category_index: int,
    weight_matrix: NDArray[np.float64],
    categories: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Remove one category column and shift category indices (MATLAB ARTwarp_Delete_Category.m).

    Contour assignments use 0-based category indices. NaN assignments are unchanged.
    """
    updated_weight = delete_category(category_index, weight_matrix)
    updated_categories = categories.copy()
    for i in range(len(categories)):
        cat = categories[i]
        if np.isnan(cat):
            continue
        ci = int(cat)
        if ci == category_index:
            raise ValueError(
                "Attempted to delete a non-empty category: a category was found which "
                "matches the deletion reference"
            )
        if ci > category_index:
            updated_categories[i] = ci - 1
    return updated_weight, updated_categories


def purge_empty_category_columns(
    weight_matrix: NDArray[np.float64],
    categories: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """
    Remove weight columns that have no assigned contours (orphan prototypes).

    This matches the post-iteration cleanup from experimental PR
    (``martion2007/delete_unused_categories``, merged as ``f671e5a``): after
    reassignments, some category indices may have zero samples while the column
    still exists. Those columns are deleted and remaining category indices are
    shifted down by one per removal, in ascending index order (same loop
    structure as the reference implementation).

    This is **not** part of MATLAB ``stable`` ``ARTwarp_Run_Categorisation.m``.
    It is optional and defaults off in ``ARTwarp``.

    Args:
        weight_matrix: Shape ``(max_features, num_categories)``.
        categories: Per-sample category indices; NaN = unassigned.

    Returns:
        ``(updated_weight_matrix, updated_categories, num_deleted)``.

    Raises:
        ValueError: If any finite category index is outside
            ``[0, num_categories-1]``.
    """
    n_col = int(weight_matrix.shape[1])
    if n_col == 0:
        return weight_matrix, categories, 0

    w = weight_matrix
    c = categories.copy()
    num_deleted = 0

    valid = ~np.isnan(c)
    if not np.any(valid):
        counts = np.zeros(n_col, dtype=np.int64)
    else:
        cc = c[valid].astype(np.int64, copy=False)
        if np.any(cc < 0) or np.any(cc >= n_col):
            raise ValueError(
                "purge_empty_category_columns: category index out of range "
                f"for weight_matrix with {n_col} columns"
            )
        bc = np.bincount(cc, minlength=n_col)
        counts = np.asarray(bc[:n_col], dtype=np.int64)

    i = 0
    while i < n_col:
        if counts[i] == 0:
            w = delete_category(i, w)
            c = np.where(c > i, c - 1, c)
            num_deleted += 1
            n_col = int(w.shape[1])
            valid = ~np.isnan(c)
            if n_col == 0:
                counts = np.zeros(0, dtype=np.int64)
            elif not np.any(valid):
                counts = np.zeros(n_col, dtype=np.int64)
            else:
                cc = c[valid].astype(np.int64, copy=False)
                if np.any(cc < 0) or np.any(cc >= n_col):
                    raise ValueError(
                        "purge_empty_category_columns: inconsistent state after delete"
                    )
                bc = np.bincount(cc, minlength=n_col)
                counts = np.asarray(bc[:n_col], dtype=np.int64)
        else:
            i += 1

    return w, c, num_deleted


def average_weights(
    weight_matrix: NDArray[np.float64],
    contour_lengths: NDArray[np.int64],
    categories: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Resample each category weight to the mean length of contours in that category.

    Matches MATLAB ARTwarp_Average_Weights.m: categories with no assigned contours
    (empty ``contourLengths``) are left unchanged.
    """
    num_features, num_categories = weight_matrix.shape
    weight = weight_matrix.copy()
    for cat in range(num_categories):
        mask = (categories == cat) & ~np.isnan(categories)
        cl = contour_lengths[mask]
        if cl.size == 0:
            continue
        col = weight[:, cat]
        i = col[col > 0]
        weight_length = len(i)
        if weight_length == 0:
            continue
        new_length = int(np.round(np.mean(cl)))
        new_length = max(1, new_length)
        x_old = np.arange(1, weight_length + 1, dtype=np.float64)
        x_new = np.linspace(1, weight_length, new_length)
        new_weight = np.interp(x_new, x_old, i.astype(np.float64, copy=False))
        weight[:, cat] = np.nan
        weight[:new_length, cat] = new_weight
    return weight


def update_weights(
    input_contour: NDArray[np.float64],
    weight_matrix: NDArray[np.float64],
    category_index: int,
    learning_rate: float,
    warp_function: NDArray[np.int32],
    compare_warped: bool = False,
) -> NDArray[np.float64]:
    """
    Update weight matrix for a category after successful match.

    This function implements the ART weight update rule with length adaptation.
    The weight vector moves toward the input contour according to the learning rate,
    and its length is adapted based on the difference between input and weight lengths.

    Args:
        input_contour: Input frequency contour, shape (n,)
        weight_matrix: Current weight matrix, shape (max_features, num_categories)
        category_index: Index of category to update
        learning_rate: Learning rate in range (0, 1]
            Higher values mean faster adaptation to new inputs
        warp_function: Warping function mapping weight to input, shape (m,)
            where m is the length of the weight vector
        compare_warped: If False (default), match MATLAB ``compare_warped == 0``: learn
            with length adaptation and unwarping. If True, match ``compare_warped == 1``:
            keep reference length equal to the current weight length (no unwarp pass).

    Returns:
        Updated weight matrix with same shape as input

    Algorithm:
        1. Extract current weight vector (remove NaN padding)
        2. Warp input contour to match weight length
        3. Update weight content: new_weight = old_weight + lr * (warped_input - old_weight)
        4. If compare_warped is False: new length and unwarp interpolation (MATLAB branch)
        5. If compare_warped is True: new_length = weight_length (MATLAB branch)
        6. Update weight matrix

    Note:
        This implements the MATLAB ART_Update_Weights.m function (ARTwarp_Update_Weights.m).
    """
    # extract current weight vector (MATLAB: find(weight(:,categoryNumber) > 0))
    weight_vector = weight_matrix[:, category_index]
    valid_mask = (weight_vector > 0) & np.isfinite(weight_vector)
    current_weight = weight_vector[valid_mask]
    weight_length = len(current_weight)
    input_length = len(input_contour)

    # check warp function length
    if len(warp_function) != weight_length:
        raise ValueError(
            f"Warp function length ({len(warp_function)}) must match "
            f"weight length ({weight_length})"
        )

    # warp input by warp func
    warped_input = input_contour[warp_function]

    # update weight with learning rate
    new_weight = current_weight + learning_rate * (warped_input - current_weight)

    if compare_warped:
        new_length = weight_length
        final_weight = new_weight
    else:
        # new length => adapt toward input length
        new_length = int(round(weight_length + learning_rate * (input_length - weight_length)))
        new_length = max(1, new_length)  # at least 1

        # unwarp function
        unwarp_function = unwarp(warp_function)

        # interpolate unwarp to new length
        if len(unwarp_function) > 1:
            old_indices = np.arange(len(unwarp_function))
            new_indices = np.linspace(0, len(unwarp_function) - 1, new_length)

            # interpolate unwarp
            f_unwarp = interp1d(
                old_indices, unwarp_function, kind="linear", fill_value="extrapolate"
            )
            interpolated_unwarp = f_unwarp(new_indices)

            # adjust unwarp with learning rate
            weight_old_indices = np.linspace(0, weight_length - 1, new_length)
            adjusted_unwarp = (
                weight_old_indices - (weight_old_indices - interpolated_unwarp) * learning_rate
            )
            adjusted_unwarp = np.clip(adjusted_unwarp, 0, weight_length - 1)
        else:
            adjusted_unwarp = np.zeros(new_length)

        # interpolate new weight to new length (adjusted unwarp)
        if len(new_weight) > 1:
            old_weight_indices = np.arange(len(new_weight))
            f_weight = interp1d(
                old_weight_indices, new_weight, kind="linear", fill_value="extrapolate"
            )
            final_weight = f_weight(adjusted_unwarp)
        else:
            final_weight = np.full(new_length, new_weight[0])

    # update weight matrix
    updated_matrix = weight_matrix.copy()
    updated_matrix[:, category_index] = np.nan  # clear old weight
    updated_matrix[:new_length, category_index] = final_weight

    return updated_matrix


def get_weight_contour(
    weight_matrix: NDArray[np.float64], category_index: int
) -> NDArray[np.float64]:
    """
    Extract weight contour for a specific category (MATLAB: weight(i,j) for i with weight(i,j) > 0).

    Args:
        weight_matrix: Weight matrix, shape (max_features, num_categories)
        category_index: Index of category to extract

    Returns:
        Weight contour for the category, shape (n,) where n <= max_features
    """
    weight_vector = weight_matrix[:, category_index]
    valid_mask = (weight_vector > 0) & np.isfinite(weight_vector)
    return weight_vector[valid_mask]


def count_active_categories(weight_matrix: NDArray[np.float64]) -> int:
    """
    Count the number of active (non-empty) categories in weight matrix.
    Uses weight > 0 to match MATLAB convention.
    """
    num_categories = weight_matrix.shape[1]
    active_count = 0
    for i in range(num_categories):
        col = weight_matrix[:, i]
        if np.any((col > 0) & np.isfinite(col)):
            active_count += 1
    return active_count
