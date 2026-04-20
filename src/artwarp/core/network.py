"""
Main ARTwarp network implementation.

This module contains the ARTwarp class that orchestrates the complete
categorization algorithm, combining DTW and ART components.

@author: Pedro Gronda Garrigues
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from artwarp.core.art import (
    activate_categories,
    calculate_match,
    check_resonance,
    sort_categories_by_activation,
)
from artwarp.core.weights import (
    add_new_category,
    average_weights,
    delete_category_reindex_assignments,
    get_weight_contour,
    initialize_weight_matrix,
    purge_empty_category_columns,
    update_weights,
)


@dataclass
class TrainingResults:
    """
    Container for ARTwarp training results.

    Attributes:
        categories: Category assignment for each input, shape (num_samples,)
            NaN values indicate no category assignment
        matches: Match values for each input, shape (num_samples,)
        weight_matrix: Final weight matrix, shape (max_features, num_categories)
        num_categories: Number of categories created
        num_iterations: Number of iterations performed
        converged: Whether training converged (no reclassifications)
        iteration_history: List of (iteration, num_reclassifications) tuples
        training_time: Total training time in seconds
        category_parent_names: Mapping from 0-based category index to list of
            contour names assigned to that category after final convergence
            Mirrors REFCONTOURS.parent_ids from ARTwarp (MATLAB) stable-1
            Empty dict when not populated (e.g. results loaded from an older pickle)
    """

    categories: NDArray[np.float64]
    matches: NDArray[np.float64]
    weight_matrix: NDArray[np.float64]
    num_categories: int
    num_iterations: int
    converged: bool
    iteration_history: List[tuple] = field(default_factory=list)
    training_time: float = 0.0
    category_parent_names: Dict[int, List[str]] = field(default_factory=dict)

    def get_category_sizes(self) -> Dict[int, int]:
        """Get the number of samples in each category."""
        unique_cats, counts = np.unique(
            self.categories[~np.isnan(self.categories)], return_counts=True
        )
        return {int(cat): int(count) for cat, count in zip(unique_cats, counts)}

    def get_uncategorized_count(self) -> int:
        """Get number of samples that could not be categorized."""
        return int(np.sum(np.isnan(self.categories)))


class ARTwarp:
    """
    ARTwarp neural network for unsupervised categorization of frequency contours.

    This class implements the complete ARTwarp algorithm, which combines:
    - Dynamic Time Warping for contour similarity
    - Adaptive Resonance Theory for unsupervised clustering
    - Dynamic category creation based on vigilance threshold

    Parameters:
        vigilance: Match threshold for category assignment, range [1, 99]
            Higher values create more categories (stricter matching)
        learning_rate: Weight update rate, range (0, 1]
            Higher values mean faster adaptation to new inputs
        bias: Activation bias, range [0, 1]
            Higher bias makes categories more selective
        max_categories: Maximum number of categories to create
        max_iterations: Maximum number of training iterations
        warp_factor_level: Maximum DTW warping factor (default: 3)
        random_seed: Random seed for reproducibility (optional)
        verbose: Whether to print progress information
        recat_single_categories: If True, run MATLAB ``ARTwarp_Recat_Single_Cats`` after each
            iteration's sample loop (reassign lone-category contours when another category
            resonates). Default False matches MATLAB with ``recatSingleCats`` off.
        compare_warped: If True, use MATLAB ``compareWarped == 1`` weight updates (fixed
            reference length) and, after each iteration, ``ARTwarp_Average_Weights``.
            Default False matches MATLAB with ``compareWarped`` off.
        deprioritize_lone_category_search: If True, when the current sample is the **only**
            contour assigned to its category, try **other** categories in activation order
            **before** its current category (experimental behaviour from PR deleted_unused_categories
            ``delete_unused_categories`` / ``f671e5a``). **Not** in MATLAB ``stable``. Default False.
        purge_empty_categories: If True, after each iteration (after optional recat /
            ``compare_warped``), remove weight columns with **zero** assigned contours and
            reindex assignments (same PR deleted_unused_categories cleanup). **Not** in MATLAB ``stable``. Default False.

    Example:
        >>> network = ARTwarp(vigilance=85.0, learning_rate=0.1)
        >>> results = network.fit(contours)
        >>> print(f"Created {results.num_categories} categories")
    """

    def __init__(
        self,
        vigilance: float = 85.0,
        learning_rate: float = 0.1,
        bias: float = 0.0,
        max_categories: int = 100,
        max_iterations: int = 50,
        warp_factor_level: int = 3,
        random_seed: Optional[int] = None,
        verbose: bool = True,
        recat_single_categories: bool = False,
        compare_warped: bool = False,
        deprioritize_lone_category_search: bool = False,
        purge_empty_categories: bool = False,
    ):
        # validate params
        if not 1 <= vigilance <= 99:
            raise ValueError(f"Vigilance must be in range [1, 99], got {vigilance}")
        if not 0 < learning_rate <= 1:
            raise ValueError(f"Learning rate must be in range (0, 1], got {learning_rate}")
        if not 0 <= bias <= 1:
            raise ValueError(f"Bias must be in range [0, 1], got {bias}")
        if max_categories < 1:
            raise ValueError(f"Max categories must be positive, got {max_categories}")
        if max_iterations < 1:
            raise ValueError(f"Max iterations must be positive, got {max_iterations}")
        if warp_factor_level <= 1:
            raise ValueError(f"Warp factor level must be > 1, got {warp_factor_level}")

        self.vigilance = vigilance
        self.learning_rate = learning_rate
        self.bias = bias
        self.max_categories = max_categories
        self.max_iterations = max_iterations
        self.warp_factor_level = warp_factor_level
        self.verbose = verbose
        self._random_seed = random_seed
        self.recat_single_categories = recat_single_categories
        self.compare_warped = compare_warped
        self.deprioritize_lone_category_search = deprioritize_lone_category_search
        self.purge_empty_categories = purge_empty_categories

        # optional random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # network state (set in fit)
        self.weight_matrix: Optional[NDArray[np.float64]] = None
        self.num_categories: int = 0
        self.max_features: int = 0

    def fit(
        self, contours: List[NDArray[np.float64]], contour_names: Optional[List[str]] = None
    ) -> TrainingResults:
        """
        Train the ARTwarp network on a set of frequency contours.

        Args:
            contours: List of frequency contour arrays
                Each contour should be a 1D array of frequency values
            contour_names: Optional list of names for each contour

        Returns:
            TrainingResults object containing category assignments and network state

        Algorithm:
            For each iteration:
                Randomize sample order
                For each sample:
                    1. Activate all categories (bottom-up)
                    2. Sort categories by activation
                    3. Search for resonance:
                        - Calculate match with sorted categories in order
                        - If match > vigilance: assign and update weights
                        - Else: try next category
                        - If all fail: create new category (if allowed)
                Check convergence (no reclassifications)
        """
        start_time = time.time()

        num_samples = len(contours)
        if num_samples == 0:
            raise ValueError("No contours provided")

        # contour names if not given
        if contour_names is None:
            contour_names = [f"contour_{i:04d}" for i in range(num_samples)]

        if len(contour_names) != num_samples:
            raise ValueError(
                f"Number of names ({len(contour_names)}) must match "
                f"number of contours ({num_samples})"
            )

        # init network
        self.max_features = max(len(c) for c in contours)
        self.weight_matrix = initialize_weight_matrix(self.max_features)
        self.num_categories = 0

        # sample tracking
        categories: NDArray[np.float64] = np.full(num_samples, np.nan, dtype=np.float64)
        matches: NDArray[np.float64] = np.zeros(num_samples, dtype=np.float64)
        iteration_history = []

        if self.verbose:
            print("ARTwarp Training")
            print("================")
            print(f"Samples:         {num_samples}")
            print(f"Max contour len: {self.max_features}")
            print(f"Vigilance:       {self.vigilance}")
            print(f"Learning rate:   {self.learning_rate}")
            print(f"Bias:            {self.bias}")
            print(f"Max categories:  {self.max_categories}")
            print(f"Max iterations:  {self.max_iterations}")
            print(f"Warp factor:     {self.warp_factor_level}")
            print(f"Recat single cats: {self.recat_single_categories}")
            print(f"Compare warped:    {self.compare_warped}")
            print(
                f"Deprioritize lone: {self.deprioritize_lone_category_search}")
            print(
                f"Purge empty cats:  {self.purge_empty_categories}")
            if self._random_seed is not None:
                print(f"Random seed:     {self._random_seed}")
            print()

        # training loop
        _green = "\033[32m" if sys.stdout.isatty() else ""
        _red = "\033[31m" if sys.stdout.isatty() else ""
        _reset = "\033[0m" if sys.stdout.isatty() else ""
        categories_at_iter_start = self.num_categories

        for iteration in range(1, self.max_iterations + 1):
            # shuffle sample order
            sample_order = np.random.permutation(num_samples)
            num_reclassifications = 0

            for sample_idx in sample_order:
                old_category = categories[sample_idx]
                current_contour = contours[sample_idx]

                if np.isnan(old_category):
                    num_in_old_category = 0
                else:
                    num_in_old_category = int((categories == old_category).sum())

                # activate categories (bottom-up)
                if self.num_categories > 0:
                    activations, warp_functions = activate_categories(
                        current_contour, self.weight_matrix, self.bias, self.warp_factor_level
                    )

                    # sort by activation
                    sorted_acts, sorted_indices = sort_categories_by_activation(activations)
                else:
                    # no categories yet
                    sorted_indices = np.array([], dtype=np.int32)

                resonance = False
                max_match = 0.0

                # PR deleted_unused_categories (martion2007): if lone in category, try other prototypes first
                if (
                    self.deprioritize_lone_category_search
                    and num_in_old_category == 1
                    and sorted_indices.size > 0
                ):
                    oc = int(old_category)
                    if 0 <= oc < self.num_categories and np.any(sorted_indices == oc):
                        others = sorted_indices[sorted_indices != oc]
                        sorted_indices = np.append(others, oc).astype(np.int32)

                for cat_rank in range(len(sorted_indices)):

                    cat_idx = sorted_indices[cat_rank]

                    # weight contour + warp func
                    weight_contour = get_weight_contour(self.weight_matrix, cat_idx)
                    warp_func = warp_functions[cat_idx]

                    if len(warp_func) == 0:
                        continue

                    # warp input -> weight
                    warped_input = current_contour[warp_func]

                    # match (top-down)
                    match = calculate_match(warped_input, weight_contour)
                    max_match = max(max_match, match)

                    # resonance?
                    if check_resonance(match, self.vigilance):
                        # update weights
                        self.weight_matrix = update_weights(
                            current_contour,
                            self.weight_matrix,
                            cat_idx,
                            self.learning_rate,
                            warp_func,
                            compare_warped=self.compare_warped,
                        )
                        categories[sample_idx] = cat_idx
                        matches[sample_idx] = match
                        resonance = True
                        break

                # no resonance
                if not resonance:
                    if self.num_categories < self.max_categories:
                        # new category
                        self.weight_matrix = add_new_category(
                            self.weight_matrix, current_contour, self.max_features
                        )
                        categories[sample_idx] = self.num_categories
                        matches[sample_idx] = 100.0  # perfect match to itself
                        self.num_categories += 1
                    else:
                        # no category slot
                        categories[sample_idx] = np.nan
                        matches[sample_idx] = max_match

                # reclassifications
                if old_category != categories[sample_idx]:
                    if not (np.isnan(old_category) and np.isnan(categories[sample_idx])):
                        num_reclassifications += 1

            # MATLAB ARTwarp_Run_Categorisation.m: optional post-iteration recat, then average
            if self.recat_single_categories:
                categories, matches, self.weight_matrix, self.num_categories = (
                    self._recat_single_categories(contours, categories, matches)
                )

            if self.compare_warped:
                contour_lengths = np.array(
                    [len(c) for c in contours], dtype=np.int64
                )
                self.weight_matrix = average_weights(
                    self.weight_matrix, contour_lengths, categories
                )

            num_purged_empty = 0
            if self.purge_empty_categories and self.weight_matrix is not None:
                self.weight_matrix, categories, num_purged_empty = purge_empty_category_columns(
                    self.weight_matrix, categories
                )
                self.num_categories = int(self.weight_matrix.shape[1])

            iteration_history.append((iteration, num_reclassifications))
            new_cats_this_round = self.num_categories - categories_at_iter_start
            pct_reclass = (num_reclassifications / num_samples * 100) if num_samples else 0

            if self.verbose:
                color = _green if num_reclassifications == 0 else _red
                reclass_str = (
                    f"reclassified {num_reclassifications:4d} / {num_samples} "
                    f"({pct_reclass:5.1f}%){_reset}"
                )
                purge_str = (
                    f"  │  purged {num_purged_empty:2d} empty categories"
                    if self.purge_empty_categories
                    else ""
                )
                print(
                    f"  {color}iter {iteration:3d}{_reset}  │  "
                    f"categories {self.num_categories:3d}  "
                    f"(+{new_cats_this_round:2d} this round)  │  {reclass_str}"
                    f"{purge_str}"
                )
            categories_at_iter_start = self.num_categories

            # converged?
            if num_reclassifications == 0:
                if self.verbose:
                    print(f"\nConverged after {iteration} iterations")
                converged = True
                break
        else:
            # hit max iterations
            converged = False
            if self.verbose:
                print(f"\nStopped after {self.max_iterations} iterations (no convergence)")

        training_time = time.time() - start_time

        if self.verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Final categories: {self.num_categories}")
            print(f"Uncategorized: {np.sum(np.isnan(categories))}")

        # build provenance map: category index -> list of contour names assigned to it
        category_parent_names: Dict[int, List[str]] = {}
        for i in range(num_samples):
            cat = categories[i]
            if not np.isnan(cat):
                cat_int = int(cat)
                if cat_int not in category_parent_names:
                    category_parent_names[cat_int] = []
                category_parent_names[cat_int].append(contour_names[i])

        return TrainingResults(
            categories=categories,
            matches=matches,
            weight_matrix=self.weight_matrix,
            num_categories=self.num_categories,
            num_iterations=iteration if converged else self.max_iterations,
            converged=converged,
            iteration_history=iteration_history,
            training_time=training_time,
            category_parent_names=category_parent_names,
        )

    def predict(
        self, contours: List[NDArray[np.float64]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Predict categories for new contours using trained network.

        Args:
            contours: List of frequency contour arrays

        Returns:
            Tuple of (categories, matches) arrays

        Raises:
            RuntimeError: If network has not been trained yet
        """
        if self.weight_matrix is None or self.num_categories == 0:
            raise RuntimeError("Network must be trained before prediction")

        num_samples = len(contours)
        categories: NDArray[np.float64] = np.full(num_samples, np.nan, dtype=np.float64)
        matches: NDArray[np.float64] = np.zeros(num_samples, dtype=np.float64)

        for idx, contour in enumerate(contours):
            # activate categories
            activations, warp_functions = activate_categories(
                contour, self.weight_matrix, self.bias, self.warp_factor_level
            )

            # best category
            best_cat_idx = int(np.argmax(activations))
            weight_contour = get_weight_contour(self.weight_matrix, best_cat_idx)
            warp_func = warp_functions[best_cat_idx]

            if len(warp_func) > 0:
                warped_input = contour[warp_func]
                match = calculate_match(warped_input, weight_contour)

                if check_resonance(match, self.vigilance):
                    categories[idx] = best_cat_idx
                    matches[idx] = match
                else:
                    # no match above vigilance
                    categories[idx] = np.nan
                    matches[idx] = match
            else:
                categories[idx] = np.nan
                matches[idx] = 0.0

        return categories, matches

    def _recat_single_categories(
        self,
        contours: List[NDArray[np.float64]],
        categories: NDArray[np.float64],
        matches: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
        """
        MATLAB ARTwarp_Recat_Single_Cats.m — reassign lone contours when a better match exists.

        Operates on the same data layout as the MATLAB ``categories`` / ``matches`` vectors
        (0-based category indices; NaN for unassigned).
        """
        weight = self.weight_matrix
        if weight is None:
            raise RuntimeError("Weight matrix must be initialized before recat")
        num_samples = len(contours)
        num_categories = weight.shape[1]

        lone_mask = np.ones(num_samples, dtype=bool)
        lone_mask[np.isnan(categories)] = False
        for cat in range(num_categories):
            cat_mask = categories == cat
            if np.sum(cat_mask) != 1:
                lone_mask[cat_mask] = False
        lone_indices = np.flatnonzero(lone_mask)
        num_lone = int(lone_indices.size)
        single_cats_moved = 0

        if num_lone == 0:
            return categories, matches, weight, num_categories

        order = np.argsort(np.random.randn(num_lone))
        cats = categories.copy()
        mats = matches.copy()

        for step in range(num_lone):
            k = int(order[step])
            idx = int(lone_indices[k])
            current_data = contours[idx]
            old_category = int(cats[idx])

            activations, warp_functions = activate_categories(
                current_data, weight, self.bias, self.warp_factor_level
            )
            activations = activations.astype(np.float64, copy=True)
            activations[old_category] = np.nan
            if np.all(np.isnan(activations)):
                continue
            best_match_idx = int(np.nanargmax(activations))
            warp_func = warp_functions[best_match_idx]
            weight_contour = get_weight_contour(weight, best_match_idx)
            if len(warp_func) == 0:
                continue
            warped_input = current_data[warp_func]
            match_val = float(calculate_match(warped_input, weight_contour))

            if check_resonance(match_val, self.vigilance):
                weight = update_weights(
                    current_data,
                    weight,
                    best_match_idx,
                    self.learning_rate,
                    warp_func,
                    compare_warped=self.compare_warped,
                )
                old_cat_mask = cats == old_category
                mats[old_cat_mask] = match_val
                cats[old_cat_mask] = best_match_idx
                weight, cats = delete_category_reindex_assignments(old_category, weight, cats)
                num_categories = weight.shape[1]
                single_cats_moved += 1

        if self.verbose:
            print(
                f"Number of single-contour-category contours reclassified "
                f"{single_cats_moved:2d}"
            )

        self.weight_matrix = weight
        return cats, mats, weight, num_categories
