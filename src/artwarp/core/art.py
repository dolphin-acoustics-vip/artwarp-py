"""
Adaptive Resonance Theory (ART) neural network components.

This module implements the ART neural network operations used in ARTwarp:
- Category activation (bottom-up processing)
- Match calculation (top-down processing)
- Category validation against vigilance threshold

The ART network enables unsupervised learning with dynamic category creation
based on similarity and vigilance criteria.

@author: Pedro Gronda Garrigues
"""

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from artwarp.core.dtw import dynamic_time_warp


def activate_categories(
    input_contour: NDArray[np.float64],
    weight_matrix: NDArray[np.float64],
    bias: float,
    warp_factor_level: int = 3,
) -> Tuple[NDArray[np.float64], List[NDArray[np.int32]]]:
    """
    Activate categories for a given input contour (bottom-up processing).

    This function computes how well the input contour matches each existing
    category by performing DTW between the input and each category's weight vector.
    The activation value is biased to control category selectivity.

    Args:
        input_contour: Input frequency contour, shape (n,)
        weight_matrix: Matrix of category weights, shape (max_features, num_categories)
            Each column represents a category's prototype contour.
            NaN values indicate unused positions.
        bias: Bias parameter in range [0, 1]
            Higher bias reduces activation, making categories more selective.
        warp_factor_level: Maximum DTW warping factor

    Returns:
        Tuple containing:
            - activations: Array of activation values for each category, shape (num_categories,)
                Higher values indicate better matches.
            - warp_functions: List of warping functions for each category
                warp_functions[i] maps input_contour to weight_matrix[:, i]

    Note:
        Activation formula: activation = DTW_similarity * (1 - bias)
        This matches the MATLAB ARTwarp_Activate_Categories.m implementation.
    """
    num_features, num_categories = weight_matrix.shape

    if num_categories == 0:
        return np.array([]), []

    activations = np.zeros(num_categories, dtype=np.float64)
    warp_functions = []

    for cat_idx in range(num_categories):
        # weight vector for this category (MATLAB: find(weight(:,j) > 0))
        weight_vector = weight_matrix[:, cat_idx]
        valid_mask = (weight_vector > 0) & np.isfinite(weight_vector)
        weight_contour = weight_vector[valid_mask]

        if len(weight_contour) == 0:
            activations[cat_idx] = 0.0
            warp_functions.append(np.array([], dtype=np.int32))
            continue

        # perform DTW between weight contour (reference) and input contour
        similarity, warp_func = dynamic_time_warp(weight_contour, input_contour, warp_factor_level)

        # bias on activation
        activations[cat_idx] = similarity * (1.0 - bias)
        warp_functions.append(warp_func)

    return activations, warp_functions


def calculate_match(
    input_contour: NDArray[np.float64], weight_vector: NDArray[np.float64]
) -> float:
    """
    Calculate match value between warped input and category weight (top-down processing).

    This function validates whether the input contour matches a category well enough
    to be assigned to it. It computes the normalized similarity between the warped
    input and the category's weight vector.

    Args:
        input_contour: Warped input contour, shape (n,)
            This should already be warped according to the optimal warping function.
        weight_vector: Category weight vector, shape (m,)
            Must not contain NaN values.

    Returns:
        Match value as a percentage in range [0, 100]
        Higher values indicate better matches.

    Formula:
        For each position i where weight[i] > 0:
            similarity[i] = min(input[i], weight[i]) / max(input[i], weight[i])
        match = (sum of similarities / number of valid positions) * 100

    Note:
        This implements the MATLAB ART_Calculate_Match.m (i = find(weightVector > 0)).
    """
    if len(input_contour) == 0 or len(weight_vector) == 0:
        return 0.0

    # same length
    if len(input_contour) != len(weight_vector):
        raise ValueError(
            f"Input contour length ({len(input_contour)}) must match "
            f"weight vector length ({len(weight_vector)})"
        )

    # MATLAB: only weight > 0 positions count
    valid = (weight_vector > 0) & np.isfinite(weight_vector)
    if not np.any(valid):
        return 0.0

    numerator = np.minimum(input_contour, weight_vector)
    denominator = np.maximum(input_contour, weight_vector)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    similarities = numerator / denominator

    match = (np.sum(similarities[valid]) / np.sum(valid)) * 100.0
    return float(match)


def sort_categories_by_activation(
    activations: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Sort categories by activation value in descending order.

    This allows the network to search categories from best to worst match,
    implementing the category search strategy in ART.

    Args:
        activations: Array of category activation values, shape (num_categories,)

    Returns:
        Tuple containing:
            - sorted_activations: Activations in descending order
            - sorted_indices: Original category indices in descending order

    Note:
        Returns negative sort to get descending order (highest activation first).
    """
    sorted_indices = np.argsort(-activations).astype(np.int32)
    sorted_activations = activations[sorted_indices]

    return sorted_activations, sorted_indices


def check_resonance(match_value: float, vigilance: float) -> bool:
    """
    Check if match exceeds vigilance threshold (resonance condition).

    In ART, resonance occurs when the match between input and category
    exceeds the vigilance parameter. This determines whether the input
    is assigned to the category or if search continues.

    Args:
        match_value: Match value in range [0, 100]
        vigilance: Vigilance threshold in range [0, 100]

    Returns:
        True if match_value > vigilance (resonance achieved)
        False otherwise

    Note:
        The comparison uses > (not >=) to match MATLAB behavior.
    """
    return match_value > vigilance
