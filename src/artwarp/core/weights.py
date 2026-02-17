"""
Weight matrix management and update operations for ARTwarp.

This module handles:
- Weight matrix initialization
- Weight updates during learning
- Category addition
- Length adaptation using interpolation

@author: Pedro Gronda Garrigues
"""

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
    new_category = np.full(max_features, np.nan, dtype=np.float64)
    new_category[: len(input_contour)] = input_contour

    # add as new column
    if weight_matrix.shape[1] == 0:
        # first category
        return new_category.reshape(-1, 1)
    else:
        # append to existing
        return np.column_stack([weight_matrix, new_category])


def update_weights(
    input_contour: NDArray[np.float64],
    weight_matrix: NDArray[np.float64],
    category_index: int,
    learning_rate: float,
    warp_function: NDArray[np.int32],
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

    Returns:
        Updated weight matrix with same shape as input

    Algorithm:
        1. Extract current weight vector (remove NaN padding)
        2. Warp input contour to match weight length
        3. Update weight content: new_weight = old_weight + lr * (warped_input - old_weight)
        4. Calculate new length: new_length = old_length + lr * (input_length - old_length)
        5. Unwarp and interpolate to new length
        6. Update weight matrix

    Note:
        This implements the MATLAB ART_Update_Weights.m function.
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
        f_unwarp = interp1d(old_indices, unwarp_function, kind="linear", fill_value="extrapolate")
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
        f_weight = interp1d(old_weight_indices, new_weight, kind="linear", fill_value="extrapolate")
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
