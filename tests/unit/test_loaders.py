"""
Unit tests for data loaders (io/loaders.py).

Covers load_ctr_file, load_csv_file, load_txt_file, load_contours,
and load_mat_categorisation to improve coverage.

@author: Pedro Gronda Garrigues
"""

import tempfile
from pathlib import Path

import pytest
import numpy as np

from artwarp.io.loaders import (
    load_ctr_file,
    load_csv_file,
    load_txt_file,
    load_contours,
    load_mat_categorisation,
)


class TestLoadCtrFile:
    """Tests for load_ctr_file()."""

    def test_load_fcontour(self):
        """Load .ctr with 'fcontour' variable."""
        try:
            from scipy.io import loadmat, savemat
        except ImportError:
            pytest.skip("scipy required for .ctr")
        contour = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".ctr", delete=False) as f:
            path = Path(f.name)
        try:
            savemat(str(path), {"fcontour": contour})
            data = load_ctr_file(path)
            np.testing.assert_array_almost_equal(data["contour"], contour)
            assert data["tempres"] is None
            assert data["ctrlength"] is None
        finally:
            path.unlink(missing_ok=True)

    def test_load_fcontour_scalar_becomes_1d(self):
        """Single scalar fcontour becomes 1D array of one element."""
        try:
            from scipy.io import savemat
        except ImportError:
            pytest.skip("scipy required")
        with tempfile.NamedTemporaryFile(suffix=".ctr", delete=False) as f:
            path = Path(f.name)
        try:
            savemat(str(path), {"fcontour": np.array(100.0)})
            data = load_ctr_file(path)
            assert data["contour"].shape == (1,)
            assert data["contour"][0] == 100.0
        finally:
            path.unlink(missing_ok=True)

    def test_load_freq_contour_drops_last(self):
        """Load with 'freqContour' drops last element (MATLAB behaviour)."""
        try:
            from scipy.io import savemat
        except ImportError:
            pytest.skip("scipy required")
        freq = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".ctr", delete=False) as f:
            path = Path(f.name)
        try:
            savemat(str(path), {"freqContour": freq})
            data = load_ctr_file(path)
            np.testing.assert_array_almost_equal(data["contour"], [100.0, 200.0, 300.0])
        finally:
            path.unlink(missing_ok=True)

    def test_load_ctr_with_tempres_and_ctrlength(self):
        """Load .ctr with tempres and ctrlength in file."""
        try:
            from scipy.io import savemat
        except ImportError:
            pytest.skip("scipy required")
        contour = np.array([100.0, 200.0], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".ctr", delete=False) as f:
            path = Path(f.name)
        try:
            savemat(str(path), {"fcontour": contour, "tempres": 0.01, "ctrlength": 0.02})
            data = load_ctr_file(path)
            assert data["tempres"] == 0.01
            assert data["ctrlength"] == 0.02
        finally:
            path.unlink(missing_ok=True)

    def test_load_ctr_ctrlength_inferred_from_tempres(self):
        """When ctrlength missing but tempres present, ctrlength = len * tempres."""
        try:
            from scipy.io import savemat
        except ImportError:
            pytest.skip("scipy required")
        contour = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".ctr", delete=False) as f:
            path = Path(f.name)
        try:
            savemat(str(path), {"fcontour": contour, "tempres": 0.01})
            data = load_ctr_file(path)
            assert data["ctrlength"] == 0.03
        finally:
            path.unlink(missing_ok=True)

    def test_load_ctr_no_fcontour_or_freq_raises(self):
        """File without fcontour/freqContour raises ValueError."""
        try:
            from scipy.io import savemat
        except ImportError:
            pytest.skip("scipy required")
        with tempfile.NamedTemporaryFile(suffix=".ctr", delete=False) as f:
            path = Path(f.name)
        try:
            savemat(str(path), {"other": np.array([1.0, 2.0])})
            with pytest.raises(ValueError, match="does not contain"):
                load_ctr_file(path)
        finally:
            path.unlink(missing_ok=True)


class TestLoadCsvFile:
    """Tests for load_csv_file()."""

    def test_load_csv_basic(self):
        """Load CSV with frequency in column 0."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("header\n100.0\n200.0\n300.0\n")
            path = Path(f.name)
        try:
            data = load_csv_file(path, frequency_column=0, skip_header=1)
            contour = data["contour"]
            assert len(contour) == 2  # last element dropped (MATLAB compat)
            np.testing.assert_array_almost_equal(contour, [100.0, 200.0])
            assert "tempres" in data
            assert "ctrlength" in data
        finally:
            path.unlink(missing_ok=True)

    def test_load_csv_frequency_column_out_of_range_raises(self):
        """Frequency column index >= columns raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("a,b\n1,2\n")
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="not found"):
                load_csv_file(path, frequency_column=5, skip_header=1)
        finally:
            path.unlink(missing_ok=True)


class TestLoadTxtFile:
    """Tests for load_txt_file()."""

    def test_load_txt_tab_delimited(self):
        """Load tab-delimited file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("100.0\t0\n200.0\t0\n300.0\t0\n")
            path = Path(f.name)
        try:
            data = load_txt_file(path, frequency_column=0)
            contour = data["contour"]
            assert len(contour) == 2
            np.testing.assert_array_almost_equal(contour, [100.0, 200.0])
        finally:
            path.unlink(missing_ok=True)

    def test_load_txt_frequency_column_out_of_range_raises(self):
        """Frequency column index >= columns raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("1\t2\n")
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="not found"):
                load_txt_file(path, frequency_column=5)
        finally:
            path.unlink(missing_ok=True)


class TestLoadContours:
    """Tests for load_contours()."""

    def test_load_contours_csv_directory(self):
        """Load contours from directory of CSV files."""
        with tempfile.TemporaryDirectory() as d:
            dir_path = Path(d)
            (dir_path / "a.csv").write_text("h\n100.0\n200.0\n")
            (dir_path / "b.csv").write_text("h\n150.0\n250.0\n350.0\n")
            contours, names = load_contours(d, file_format="csv", frequency_column=0)
            assert len(contours) == 2
            assert names == ["a", "b"]
            assert len(contours[0]) == 1
            assert len(contours[1]) == 2

    def test_load_contours_with_return_tempres(self):
        """load_contours with return_tempres=True returns third element."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "one.csv").write_text("h\n100.0\n200.0\n")
            contours, names, tempres_list = load_contours(
                d, file_format="csv", frequency_column=0, return_tempres=True
            )
            assert len(tempres_list) == 1
            assert tempres_list[0] is not None

    def test_load_contours_directory_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            load_contours("/nonexistent_dir_xyz_123", file_format="csv")

    def test_load_contours_not_a_directory_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(NotADirectoryError, match="Not a directory"):
                load_contours(path, file_format="csv")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_contours_unknown_format_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="Unknown file format"):
                load_contours(d, file_format="unknown")

    def test_load_contours_no_files_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="No contour files found"):
                load_contours(d, file_format="csv")

    def test_load_contours_auto_detects_extensions(self):
        """file_format='auto' looks for .ctr, .csv, .txt."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.csv").write_text("h\n1.0\n2.0\n")
            contours, names = load_contours(d, file_format="auto", frequency_column=0)
            assert len(contours) == 1
            assert names == ["x"]


class TestLoadMatCategorisationExtended:
    """Additional tests for load_mat_categorisation (DATA branch, 1-d weight, etc.)."""

    def test_load_mat_data_empty_returns_net_only(self):
        """When DATA is present but empty (size 0), return same as NET only."""
        try:
            from scipy.io import savemat
        except ImportError:
            pytest.skip("scipy required")
        weight = np.array([[100.0], [150.0]], dtype=np.float64)
        net = {
            "weight": weight,
            "numFeatures": 2,
            "numCategories": 1,
            "vigilance": 85.0,
            "bias": 0.0,
            "learningRate": 0.1,
            "maxNumCategories": 50,
            "maxNumIterations": 50,
        }
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            savemat(path, {"NET": net, "DATA": np.array([])})
            out = load_mat_categorisation(path)
            assert out["num_categories"] == 1
            assert "contours" not in out
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_mat_weight_1d_reshaped_to_2d(self):
        """1-d weight array is reshaped to (n, 1)."""
        try:
            from scipy.io import savemat
        except ImportError:
            pytest.skip("scipy required")
        weight_1d = np.array([100.0, 150.0], dtype=np.float64)
        net = {
            "weight": weight_1d,
            "numFeatures": 2,
            "numCategories": 1,
            "vigilance": 85.0,
            "bias": 0.0,
            "learningRate": 0.1,
            "maxNumCategories": 50,
            "maxNumIterations": 50,
        }
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            savemat(path, {"NET": net})
            out = load_mat_categorisation(path)
            assert out["weight_matrix"].ndim == 2
            assert out["weight_matrix"].shape[1] == 1
            np.testing.assert_array_almost_equal(out["weight_matrix"].ravel(), weight_1d)
        finally:
            Path(path).unlink(missing_ok=True)
