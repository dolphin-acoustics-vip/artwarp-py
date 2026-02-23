"""
Unit tests for validation and utility modules.

Covers validate_contour, validate_contours, validate_parameters (artwarp.utils.validation)
and numba_available, report_numba_status, check_numba (artwarp.utils.numba_check).

@author: Pedro Gronda Garrigues
"""

import io
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


class TestNumbaCheck:
    """Tests for artwarp.utils.numba_check (numba_available, report_numba_status, check_numba)."""

    def test_numba_available_returns_bool(self):
        """numba_available() returns a boolean."""
        from artwarp.utils.numba_check import numba_available

        assert isinstance(numba_available(), bool)

    def test_report_numba_status_numba_available_returns_true(self):
        """When Numba is available, report_numba_status returns True and prints installed message."""
        from artwarp.utils.numba_check import report_numba_status

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=True):
            result = report_numba_status(stream=stream)
        assert result is True
        assert "installed" in stream.getvalue() and "performance" in stream.getvalue()

    def test_report_numba_status_numba_not_available_returns_false(self):
        """When Numba is not available, report_numba_status returns False and prints warning."""
        from artwarp.utils.numba_check import report_numba_status

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=False):
            result = report_numba_status(stream=stream)
        assert result is False
        assert "not installed" in stream.getvalue()
        assert "pip install numba" in stream.getvalue()

    def test_check_numba_numba_available_returns_true(self):
        """When Numba is available, check_numba returns True (no install prompt)."""
        from artwarp.utils.numba_check import check_numba

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=True):
            result = check_numba(offer_install=False, stream=stream)
        assert result is True
        assert "installed" in stream.getvalue()

    def test_check_numba_numba_not_available_no_install_returns_false(self):
        """When Numba not available and offer_install=False, check_numba returns False."""
        from artwarp.utils.numba_check import check_numba

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=False):
            result = check_numba(offer_install=False, stream=stream)
        assert result is False
        assert "not installed" in stream.getvalue()

    def test_check_numba_numba_not_available_offer_install_not_tty_returns_false(self):
        """When Numba not available and stdin is not a TTY, no prompt; returns False."""
        from artwarp.utils.numba_check import check_numba

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=False), patch(
            "sys.stdin.isatty", return_value=False
        ):
            result = check_numba(offer_install=True, stream=stream)
        assert result is False
        assert "Install Numba now" not in stream.getvalue()

    def test_check_numba_numba_not_available_no_pip_no_conda_returns_false(self):
        """When Numba not available, TTY, but no pip/conda, prints manual message and returns False."""
        from artwarp.utils.numba_check import check_numba

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=False), patch(
            "sys.stdin.isatty", return_value=True
        ), patch("artwarp.utils.numba_check._pip_available", return_value=False), patch(
            "artwarp.utils.numba_check._conda_available", return_value=False
        ):
            result = check_numba(offer_install=True, stream=stream)
        assert result is False
        assert "pip and conda not detected" in stream.getvalue()

    def test_check_numba_numba_not_available_user_declines_install_returns_false(self):
        """When user declines install (input n), check_numba returns False."""
        from artwarp.utils.numba_check import check_numba

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=False), patch(
            "sys.stdin.isatty", return_value=True
        ), patch("artwarp.utils.numba_check._pip_available", return_value=True), patch(
            "artwarp.utils.numba_check._conda_available", return_value=False
        ), patch(
            "builtins.input", return_value="n"
        ):
            result = check_numba(offer_install=True, stream=stream)
        assert result is False

    def test_check_numba_numba_not_available_user_accepts_install_pip_success(self):
        """When user accepts install and pip install succeeds, check_numba returns True."""
        from artwarp.utils.numba_check import check_numba

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=False), patch(
            "sys.stdin.isatty", return_value=True
        ), patch("artwarp.utils.numba_check._pip_available", return_value=True), patch(
            "artwarp.utils.numba_check._conda_available", return_value=False
        ), patch(
            "builtins.input", return_value="y"
        ), patch(
            "subprocess.run", return_value=MagicMock(returncode=0)
        ):
            result = check_numba(offer_install=True, stream=stream)
        assert result is True
        assert "Numba installed successfully" in stream.getvalue()

    def test_check_numba_numba_not_available_pip_install_fails_returns_false(self):
        """When user accepts install but pip install fails, check_numba returns False."""
        from artwarp.utils.numba_check import check_numba

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=False), patch(
            "sys.stdin.isatty", return_value=True
        ), patch("artwarp.utils.numba_check._pip_available", return_value=True), patch(
            "artwarp.utils.numba_check._conda_available", return_value=False
        ), patch(
            "builtins.input", return_value="y"
        ), patch(
            "subprocess.run", return_value=MagicMock(returncode=1)
        ):
            result = check_numba(offer_install=True, stream=stream)
        assert result is False
        assert "Install failed" in stream.getvalue()

    def test_check_numba_eof_during_prompt_returns_false(self):
        """When EOFError during install prompt (e.g. non-interactive), returns False."""
        from artwarp.utils.numba_check import check_numba

        stream = io.StringIO()
        with patch("artwarp.utils.numba_check.numba_available", return_value=False), patch(
            "sys.stdin.isatty", return_value=True
        ), patch("artwarp.utils.numba_check._pip_available", return_value=True), patch(
            "builtins.input", side_effect=EOFError
        ):
            result = check_numba(offer_install=True, stream=stream)
        assert result is False
