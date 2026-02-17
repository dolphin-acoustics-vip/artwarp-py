"""
Unit tests for ARTwarp network.

Tests the main network training and prediction functionality.

@author: Pedro Gronda Garrigues
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from artwarp.core.network import ARTwarp, TrainingResults


class TestARTwarpInitialization:
    """Tests for ARTwarp network initialization."""

    def test_default_initialization(self):
        """Test network with default parameters."""
        network = ARTwarp()

        assert network.vigilance == 85.0
        assert network.learning_rate == 0.1
        assert network.bias == 0.0
        assert network.max_categories == 100
        assert network.max_iterations == 50
        assert network.warp_factor_level == 3

    def test_custom_parameters(self):
        """Test network with custom parameters."""
        network = ARTwarp(
            vigilance=90.0,
            learning_rate=0.2,
            bias=0.1,
            max_categories=10,
            max_iterations=20,
            warp_factor_level=5,
        )

        assert network.vigilance == 90.0
        assert network.learning_rate == 0.2
        assert network.bias == 0.1
        assert network.max_categories == 10
        assert network.max_iterations == 20
        assert network.warp_factor_level == 5

    def test_invalid_vigilance(self):
        """Test that invalid vigilance raises error."""
        with pytest.raises(ValueError, match="Vigilance"):
            ARTwarp(vigilance=0.5)

        with pytest.raises(ValueError, match="Vigilance"):
            ARTwarp(vigilance=100.0)

    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises error."""
        with pytest.raises(ValueError, match="Learning rate"):
            ARTwarp(learning_rate=0.0)

        with pytest.raises(ValueError, match="Learning rate"):
            ARTwarp(learning_rate=1.5)

    def test_invalid_bias(self):
        """Test that invalid bias raises error."""
        with pytest.raises(ValueError, match="Bias"):
            ARTwarp(bias=-0.1)

        with pytest.raises(ValueError, match="Bias"):
            ARTwarp(bias=1.5)

    def test_random_seed_reproducibility(self):
        """Test that random seed gives reproducible results."""
        contours = [
            np.array([100.0, 200.0, 300.0]),
            np.array([150.0, 250.0, 350.0]),
            np.array([200.0, 300.0, 400.0]),
        ]

        # train two networks with same seed
        net1 = ARTwarp(vigilance=85.0, random_seed=42, verbose=False)
        results1 = net1.fit(contours)

        net2 = ARTwarp(vigilance=85.0, random_seed=42, verbose=False)
        results2 = net2.fit(contours)

        # => identical results
        assert_array_equal(results1.categories, results2.categories)
        assert_allclose(results1.weight_matrix, results2.weight_matrix)


class TestARTwarpTraining:
    """Tests for ARTwarp network training."""

    def test_simple_training(self):
        """Test training on simple dataset."""
        # simple contours
        contours = [
            np.array([100.0, 200.0, 300.0]),
            np.array([105.0, 205.0, 305.0]),  # similar to first
            np.array([500.0, 600.0, 700.0]),  # very different
        ]

        network = ARTwarp(vigilance=90.0, verbose=False, random_seed=42)
        results = network.fit(contours)

        # should create at least 2 categories (similar ones together, different one separate)
        assert results.num_categories >= 2

        # first two same category (if vigilance allows); third different
        assert results.categories[0] != results.categories[2]

    def test_identical_contours_one_category(self):
        """Test that identical contours create one category."""
        contours = [np.array([100.0, 200.0, 300.0])] * 5

        network = ARTwarp(vigilance=85.0, verbose=False, random_seed=42)
        results = network.fit(contours)

        # exactly 1 category
        assert results.num_categories == 1

        # all same category
        assert len(set(results.categories)) == 1

        # all matches should be ~100%
        assert np.all(results.matches > 99.0)

    def test_max_categories_limit(self):
        """Test that max_categories limit is respected."""
        # many different contours
        contours = [np.array([100.0 * i, 200.0 * i, 300.0 * i]) for i in range(1, 20)]

        network = ARTwarp(
            vigilance=99.0,  # high vigilance => separate categories
            max_categories=5,
            verbose=False,
            random_seed=42,
        )
        results = network.fit(contours)

        # should not exceed max_categories
        assert results.num_categories <= 5

    def test_convergence_detection(self):
        """Test that convergence is properly detected."""
        # simple dataset => quick convergence
        contours = [
            np.array([100.0, 200.0, 300.0]),
            np.array([100.0, 200.0, 300.0]),
        ]

        network = ARTwarp(vigilance=85.0, max_iterations=50, verbose=False, random_seed=42)
        results = network.fit(contours)

        # converged (no reclassifications)
        assert results.converged is True

        # should take fewer iterations than max
        assert results.num_iterations < network.max_iterations

    def test_empty_contours_list(self):
        """Test that empty contours list raises error."""
        network = ARTwarp(verbose=False)

        with pytest.raises(ValueError, match="No contours"):
            network.fit([])

    def test_contour_names(self):
        """Test training with custom contour names."""
        contours = [
            np.array([100.0, 200.0, 300.0]),
            np.array([150.0, 250.0, 350.0]),
        ]
        names = ["contour_A", "contour_B"]

        network = ARTwarp(verbose=False, random_seed=42)
        results = network.fit(contours, contour_names=names)

        # completes without errors
        assert results is not None

    def test_training_results_structure(self):
        """Test that training results have correct structure."""
        contours = [
            np.array([100.0, 200.0, 300.0]),
            np.array([150.0, 250.0, 350.0]),
        ]

        network = ARTwarp(verbose=False, random_seed=42)
        results = network.fit(contours)

        # result attributes
        assert isinstance(results, TrainingResults)
        assert len(results.categories) == len(contours)
        assert len(results.matches) == len(contours)
        assert results.weight_matrix is not None
        assert results.num_categories > 0
        assert results.num_iterations > 0
        assert isinstance(results.converged, bool)
        assert len(results.iteration_history) > 0
        assert results.training_time > 0


class TestARTwarpPrediction:
    """Tests for ARTwarp network prediction."""

    def test_predict_after_training(self):
        """Test prediction on new data after training."""
        # training data
        train_contours = [
            np.array([100.0, 200.0, 300.0]),
            np.array([500.0, 600.0, 700.0]),
        ]

        # test data (similar to training)
        test_contours = [
            np.array([105.0, 205.0, 305.0]),  # like first training
            np.array([505.0, 605.0, 705.0]),  # like second training
        ]

        # train
        network = ARTwarp(vigilance=85.0, verbose=False, random_seed=42)
        train_results = network.fit(train_contours)

        # predict
        categories, matches = network.predict(test_contours)

        assert len(categories) == len(test_contours)
        assert len(matches) == len(test_contours)

        # assigned to categories (not NaN)
        assert not np.isnan(categories[0])
        assert not np.isnan(categories[1])

    def test_predict_before_training_error(self):
        """Test that prediction before training raises error."""
        network = ARTwarp(verbose=False)
        test_contours = [np.array([100.0, 200.0, 300.0])]

        with pytest.raises(RuntimeError, match="must be trained"):
            network.predict(test_contours)

    def test_predict_dissimilar_contour(self):
        """Test prediction with very dissimilar contour."""
        # training data
        train_contours = [np.array([100.0, 200.0, 300.0])]

        # very different test data
        test_contours = [np.array([10000.0, 20000.0, 30000.0])]

        # train with high vigilance
        network = ARTwarp(vigilance=95.0, verbose=False, random_seed=42)
        network.fit(train_contours)

        # predict
        categories, matches = network.predict(test_contours)

        # might be unassigned (NaN) or low match
        if not np.isnan(categories[0]):
            assert matches[0] < 50.0


class TestTrainingResults:
    """Tests for TrainingResults dataclass."""

    def test_get_category_sizes(self):
        """Test getting category sizes from results."""
        categories = np.array([0, 0, 1, 1, 1, 2])
        matches = np.zeros(6)
        weight_matrix = np.zeros((10, 3))

        results = TrainingResults(
            categories=categories,
            matches=matches,
            weight_matrix=weight_matrix,
            num_categories=3,
            num_iterations=5,
            converged=True,
        )

        sizes = results.get_category_sizes()

        assert sizes[0] == 2
        assert sizes[1] == 3
        assert sizes[2] == 1

    def test_get_uncategorized_count(self):
        """Test counting uncategorized samples."""
        categories = np.array([0, 1, np.nan, 2, np.nan])
        matches = np.zeros(5)
        weight_matrix = np.zeros((10, 3))

        results = TrainingResults(
            categories=categories,
            matches=matches,
            weight_matrix=weight_matrix,
            num_categories=3,
            num_iterations=5,
            converged=True,
        )

        uncategorized = results.get_uncategorized_count()

        assert uncategorized == 2
