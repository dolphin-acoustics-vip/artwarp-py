"""
Tests for MATLAB compatibility (Load Categorisation, 1-based export, etc.).

@author: Pedro Gronda Garrigues
"""

import tempfile
from pathlib import Path

import pytest
import numpy as np

from artwarp.io.loaders import load_mat_categorisation
from artwarp.io.exporters import export_category_assignments, export_reference_contours
from artwarp.utils.resample import resample_contours


class TestLoadMatCategorisation:
    """Tests for loading MATLAB ARTwarp .mat files."""

    def test_load_net_only(self):
        """Load .mat with NET only (no DATA)."""
        try:
            from scipy.io import savemat, loadmat
        except ImportError:
            pytest.skip("scipy required for .mat I/O")
        weight = np.array([[100.0, 200.0], [150.0, 250.0], [np.nan, np.nan]], dtype=np.float64)
        net = {
            "weight": weight,
            "numFeatures": np.array(3),
            "numCategories": np.array(2),
            "maxNumCategories": np.array(50),
            "vigilance": np.array(85.0),
            "bias": np.array(0.0),
            "maxNumIterations": np.array(50),
            "learningRate": np.array(0.1),
        }
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            savemat(path, {"NET": net})
            out = load_mat_categorisation(path)
        finally:
            Path(path).unlink(missing_ok=True)
        assert "weight_matrix" in out
        np.testing.assert_array_almost_equal(out["weight_matrix"], weight)
        assert out["num_categories"] == 2
        assert out["max_features"] == 3
        assert out["vigilance"] == 85.0
        assert out["bias"] == 0.0
        assert out["learning_rate"] == 0.1
        assert "contours" not in out

    def test_missing_net_raises(self):
        """File without NET raises ValueError."""
        try:
            from scipy.io import savemat
        except ImportError:
            pytest.skip("scipy required")
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            savemat(path, {"foo": np.array(1)})
            with pytest.raises(ValueError, match="does not contain 'NET'"):
                load_mat_categorisation(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_mat_categorisation("/nonexistent/ARTwarp85FINAL.mat")


class TestExportOneBased:
    """Tests for 1-based category export (MATLAB convention)."""

    def test_export_category_assignments_one_based(self):
        """export_category_assignments with one_based_categories=True."""
        categories = np.array([0.0, 1.0, np.nan, 2.0])
        matches = np.array([95.0, 88.0, 50.0, 92.0])
        names = ["a", "b", "c", "d"]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_category_assignments(categories, matches, names, path, one_based_categories=True)
            content = Path(path).read_text()
        finally:
            Path(path).unlink(missing_ok=True)
        assert "contour_name,category,match" in content
        assert "a,1," in content
        assert "b,2," in content
        assert "c,uncategorized," in content
        assert "d,3," in content

    def test_export_category_assignments_zero_based(self):
        """export_category_assignments with one_based_categories=False (default)."""
        categories = np.array([0.0, 1.0])
        matches = np.array([95.0, 88.0])
        names = ["a", "b"]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_category_assignments(categories, matches, names, path)
            content = Path(path).read_text()
        finally:
            Path(path).unlink(missing_ok=True)
        assert "a,0," in content
        assert "b,1," in content


class TestResampleContours:
    """Tests for resample_contours (MATLAB resample option)."""

    def test_resample_reduces_length(self):
        """Larger sample_interval_sec should yield fewer points."""
        contour = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
        tempres = 0.01
        out = resample_contours([contour], tempres, sample_interval_sec=0.02)
        assert len(out) == 1
        assert len(out[0]) <= len(contour)
        assert out[0][0] == contour[0]
        assert out[0][-1] == contour[-1]

    def test_resample_single_tempres_for_all(self):
        """Single float tempres applied to all contours."""
        contours = [
            np.array([100.0, 200.0]),
            np.array([150.0, 250.0, 350.0]),
        ]
        out = resample_contours(contours, tempres=0.01, sample_interval_sec=0.01)
        assert len(out) == 2
        np.testing.assert_allclose(out[0], contours[0])
        np.testing.assert_allclose(out[1], contours[1])

    def test_resample_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match"):
            resample_contours(
                [np.array([1.0, 2.0])],
                tempres=[0.01, 0.02],
                sample_interval_sec=0.01,
            )


class TestRefContourExport:
    """Reference contour export matches MATLAB SaveRefContours.m."""

    def test_default_prefix_is_refContour(self):
        """Default filename prefix is refContour (MATLAB SaveRefContours.m)."""
        weight_matrix = np.full((5, 1), np.nan)
        weight_matrix[:3, 0] = [100.0, 200.0, 300.0]
        with tempfile.TemporaryDirectory() as d:
            export_reference_contours(weight_matrix, d)
            files = list(Path(d).glob("*.csv"))
            assert len(files) == 1
            assert files[0].name == "refContour_1.csv"
