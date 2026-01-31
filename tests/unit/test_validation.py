"""
Unit tests for validation utilities.

Covers validate_contour, validate_contours, and validate_parameters
to achieve high coverage of artwarp.utils.validation.

@author: Pedro Gronda Garrigues
"""

import pytest
import numpy as np

from artwarp.utils.validation import (
    validate_contour,
    validate_contours,
    validate_parameters,
)


class TestValidateContour:
    """Tests for validate_contour()."""

    def test_valid_contour_passes(self):
        """Valid 1D positive numeric array passes."""
        contour = np.array([100.0, 200.0, 300.0])
        validate_contour(contour)

    def test_not_ndarray_raises(self):
        """Non-numpy array raises ValueError."""
        with pytest.raises(ValueError, match="must be a numpy array"):
            validate_contour([100.0, 200.0])

    def test_2d_raises(self):
        """2D array raises ValueError."""
        contour = np.array([[100.0, 200.0], [150.0, 250.0]])
        with pytest.raises(ValueError, match="1-dimensional"):
            validate_contour(contour)

    def test_empty_raises(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_contour(np.array([]))

    def test_non_numeric_dtype_raises(self):
        """Non-numeric dtype raises ValueError."""
        contour = np.array(["a", "b", "c"])
        with pytest.raises(ValueError, match="numeric"):
            validate_contour(contour)

    def test_nan_raises(self):
        """Contour with NaN raises ValueError."""
        contour = np.array([100.0, np.nan, 300.0])
        with pytest.raises(ValueError, match="NaN"):
            validate_contour(contour)

    def test_inf_raises(self):
        """Contour with inf raises ValueError."""
        contour = np.array([100.0, np.inf, 300.0])
        with pytest.raises(ValueError, match="infinite"):
            validate_contour(contour)

    def test_non_positive_raises(self):
        """Contour with zero or negative values raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            validate_contour(np.array([100.0, 0.0, 300.0]))
        with pytest.raises(ValueError, match="positive"):
            validate_contour(np.array([100.0, -1.0, 300.0]))


class TestValidateContours:
    """Tests for validate_contours()."""

    def test_valid_list_passes(self):
        """Valid list of contours passes."""
        contours = [
            np.array([100.0, 200.0]),
            np.array([150.0, 250.0, 350.0]),
        ]
        validate_contours(contours)

    def test_not_list_raises(self):
        """Non-list raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            validate_contours(np.array([100.0, 200.0]))

    def test_empty_list_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_contours([])

    def test_invalid_contour_at_index_raises(self):
        """Invalid contour in list raises with index in message."""
        contours = [
            np.array([100.0, 200.0]),
            np.array([100.0, np.nan]),  # invalid at index 1 (NaN)
        ]
        with pytest.raises(ValueError, match="Invalid contour at index 1"):
            validate_contours(contours)


class TestValidateParameters:
    """Tests for validate_parameters()."""

    def test_valid_parameters_pass(self):
        """Valid parameter set passes."""
        validate_parameters(
            vigilance=85.0,
            learning_rate=0.1,
            bias=0.0,
            max_categories=50,
            max_iterations=50,
            warp_factor_level=3,
        )

    def test_vigilance_low_raises(self):
        with pytest.raises(ValueError, match="Vigilance.*[1, 99]"):
            validate_parameters(0.5, 0.1, 0.0, 50, 50, 3)

    def test_vigilance_high_raises(self):
        with pytest.raises(ValueError, match="Vigilance.*[1, 99]"):
            validate_parameters(100.0, 0.1, 0.0, 50, 50, 3)

    def test_learning_rate_zero_raises(self):
        with pytest.raises(ValueError, match="Learning rate"):
            validate_parameters(85.0, 0.0, 0.0, 50, 50, 3)

    def test_learning_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="Learning rate"):
            validate_parameters(85.0, 1.5, 0.0, 50, 50, 3)

    def test_bias_negative_raises(self):
        with pytest.raises(ValueError, match="Bias must be in range"):
            validate_parameters(85.0, 0.1, -0.1, 50, 50, 3)

    def test_bias_above_one_raises(self):
        with pytest.raises(ValueError, match="Bias must be in range"):
            validate_parameters(85.0, 0.1, 1.5, 50, 50, 3)

    def test_max_categories_non_int_raises(self):
        with pytest.raises(ValueError, match="Max categories"):
            validate_parameters(85.0, 0.1, 0.0, 50.5, 50, 3)

    def test_max_categories_zero_raises(self):
        with pytest.raises(ValueError, match="Max categories"):
            validate_parameters(85.0, 0.1, 0.0, 0, 50, 3)

    def test_max_iterations_non_int_raises(self):
        with pytest.raises(ValueError, match="Max iterations"):
            validate_parameters(85.0, 0.1, 0.0, 50, 50.5, 3)

    def test_max_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="Max iterations"):
            validate_parameters(85.0, 0.1, 0.0, 50, 0, 3)

    def test_warp_factor_level_one_raises(self):
        with pytest.raises(ValueError, match="Warp factor level"):
            validate_parameters(85.0, 0.1, 0.0, 50, 50, 1)

    def test_warp_factor_level_non_int_raises(self):
        with pytest.raises(ValueError, match="Warp factor level"):
            validate_parameters(85.0, 0.1, 0.0, 50, 50, 3.0)
