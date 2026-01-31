"""
Unit tests for Dynamic Time Warping implementation.

Tests mathematical correctness, edge cases, and performance.

When Numba is available, the JIT path is used and the Python fallback in dtw.py
is not executed, so coverage of dtw.py can be low. The tests in
TestDynamicTimeWarpPythonPath and TestUnwarpPythonPath force the Python path
(NUMBA_AVAILABLE=False) so that both code paths are covered.

@author: Pedro Gronda Garrigues
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from unittest.mock import patch

from artwarp.core.dtw import (
    compute_similarity_matrix,
    dynamic_time_warp,
    unwarp,
)


class TestComputeSimilarityMatrix:
    """Tests for similarity matrix computation."""
    
    def test_identical_contours(self):
        """Test that identical contours give 100% similarity."""
        u1 = np.array([100.0, 200.0, 300.0])
        u2 = np.array([100.0, 200.0, 300.0])
        
        M = compute_similarity_matrix(u1, u2)
        
        assert M.shape == (3, 3)
        assert_allclose(np.diag(M), 100.0, rtol=1e-10)
    
    def test_different_lengths(self):
        """Test similarity matrix with different length contours."""
        u1 = np.array([100.0, 200.0])
        u2 = np.array([100.0, 200.0, 300.0])
        
        M = compute_similarity_matrix(u1, u2)
        
        assert M.shape == (2, 3)
        assert M[0, 0] == 100.0  # 100/100
        assert M[1, 1] == 100.0  # 200/200
    
    def test_similarity_calculation(self):
        """Test correct similarity percentage calculation."""
        u1 = np.array([100.0])
        u2 = np.array([200.0])
        
        M = compute_similarity_matrix(u1, u2)
        
        # min(100,200)/max(100,200)*100 => 50.0
        assert_allclose(M[0, 0], 50.0, rtol=1e-10)
    
    def test_vectorization(self):
        """Test that vectorized computation is correct."""
        u1 = np.array([100.0, 200.0, 300.0])
        u2 = np.array([150.0, 250.0, 350.0])
        
        M = compute_similarity_matrix(u1, u2)
        
        # verify a few specific values
        # M[0,0] = min(100,150)/max(100,150)*100 = 100/150*100 = 66.67
        assert_allclose(M[0, 0], 100.0/150.0*100.0, rtol=1e-10)
        # M[1,1] = min(200,250)/max(200,250)*100 = 200/250*100 = 80.0
        assert_allclose(M[1, 1], 200.0/250.0*100.0, rtol=1e-10)


class TestDynamicTimeWarp:
    """Tests for DTW algorithm."""
    
    def test_identical_contours_perfect_match(self):
        """Test that identical contours give perfect match."""
        u1 = np.array([100.0, 200.0, 300.0, 400.0])
        u2 = u1.copy()
        
        similarity, warp_func = dynamic_time_warp(u1, u2)
        
        assert_allclose(similarity, 100.0, rtol=1e-6)
        assert len(warp_func) == len(u1)
        # each element maps to itself
        assert_array_equal(warp_func, np.arange(len(u1)))
    
    def test_length_ratio_constraint(self):
        """Test that length ratio constraint is enforced."""
        # contours that differ too much in length
        u1 = np.array([100.0, 200.0])
        u2 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
        
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=2)
        
        # => zero similarity, empty warp
        assert similarity == 0.0
        assert len(warp_func) == 0

    def test_length_ratio_equals_warp_factor_rejected(self):
        """MATLAB warp.m: reject when ratio >= warpFactorLevel (not only >)."""
        # u1 len 3, u2 len 6 => ratio = 6/(3-1) = 3.0
        u1 = np.array([100.0, 200.0, 300.0])
        u2 = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0])
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=3)
        # 3 >= 3 -> must reject (MATLAB behavior)
        assert similarity == 0.0
        assert len(warp_func) == 0
    
    def test_simple_time_stretch(self):
        """Test DTW on a simple time-stretched contour."""
        # original contour
        u1 = np.array([100.0, 200.0, 300.0, 200.0, 100.0])
        # time-stretched (each element repeated)
        u2 = np.array([100.0, 100.0, 200.0, 200.0, 300.0, 300.0, 200.0, 200.0, 100.0, 100.0])
        
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=3)
        
        # should be high similarity (>80%)
        assert similarity > 80.0
        assert len(warp_func) == len(u1)
    
    def test_warp_factor_level_effect(self):
        """Test that warp_factor_level controls flexibility."""
        # use lengths so ratio = 5/3 < 2 (no rejection for level 2 or 5)
        u1 = np.array([100.0, 200.0, 300.0, 250.0])
        u2 = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
        
        sim1, warp1 = dynamic_time_warp(u1, u2, warp_factor_level=2)
        sim2, warp2 = dynamic_time_warp(u1, u2, warp_factor_level=5)
        
        # both should succeed (length ratio 5/3 < 2)
        assert sim1 > 0.0
        assert sim2 > 0.0
    
    def test_single_element_contours(self):
        """Test DTW with single-element contours."""
        u1 = np.array([100.0])
        u2 = np.array([100.0])
        
        similarity, warp_func = dynamic_time_warp(u1, u2)
        
        assert_allclose(similarity, 100.0, rtol=1e-6)
        assert_array_equal(warp_func, [0])
    
    def test_different_frequency_ranges(self):
        """Test DTW with contours in different frequency ranges."""
        u1 = np.array([100.0, 200.0, 300.0])
        u2 = np.array([200.0, 400.0, 600.0])  # doubled frequencies
        
        similarity, warp_func = dynamic_time_warp(u1, u2)
        
        # full DP (MATLAB parity) chooses an optimal path; diagonal would give 50%, but
        # the Itakura-constrained path can yield a higher normalized similarity (~66.67%)
        assert 0.0 <= similarity <= 100.0
        assert len(warp_func) == len(u1)
        assert_allclose(similarity, 200.0 / 3.0, rtol=0.01)  # full-DP ~66.67%


class TestDynamicTimeWarpPythonPath:
    """Run DTW with Python fallback (NUMBA_AVAILABLE=False) for coverage of dtw.py."""

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_identical_contours_python_path(self):
        """Identical contours via Python DP path."""
        u1 = np.array([100.0, 200.0, 300.0, 400.0])
        u2 = u1.copy()
        similarity, warp_func = dynamic_time_warp(u1, u2)
        assert_allclose(similarity, 100.0, rtol=1e-6)
        assert len(warp_func) == len(u1)
        assert_array_equal(warp_func, np.arange(len(u1)))

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_length_ratio_reject_python_path(self):
        """Length ratio rejection (no Numba, so Python path still does early return)."""
        u1 = np.array([100.0, 200.0])
        u2 = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=2)
        assert similarity == 0.0
        assert len(warp_func) == 0

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_simple_time_stretch_python_path(self):
        """Time-stretched contour via full Python DP."""
        u1 = np.array([100.0, 200.0, 300.0, 200.0, 100.0])
        u2 = np.array([100.0, 100.0, 200.0, 200.0, 300.0, 300.0, 200.0, 200.0, 100.0, 100.0])
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=3)
        assert similarity > 80.0
        assert len(warp_func) == len(u1)

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_short_contours_simple_alignment_python_path(self):
        """Very short contours use simple alignment (m==n diagonal)."""
        u1 = np.array([100.0, 200.0])
        u2 = np.array([105.0, 195.0])
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=3)
        assert similarity > 0.0
        assert len(warp_func) == 2
        assert_array_equal(warp_func, np.arange(2))

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_short_contours_m_shorter_python_path(self):
        """Very short contours m < n: map m to closest n (length ratio must pass)."""
        # m=3, n=4 -> ratio 4/(3-1)=2 < 3, so not rejected; min(3,4)=3 < 6 and |4-3|=1 <= 3 -> simple alignment
        u1 = np.array([100.0, 200.0, 300.0])
        u2 = np.array([100.0, 150.0, 200.0, 250.0])
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=3)
        assert similarity > 0.0
        assert len(warp_func) == 3
        assert np.all((warp_func >= 0) & (warp_func <= 3))

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_short_contours_n_shorter_python_path(self):
        """Very short contours n < m: map m to closest n (else branch with clip)."""
        # m=4,n=3 => same, simple alignment
        u1 = np.array([100.0, 150.0, 200.0, 250.0])
        u2 = np.array([100.0, 200.0, 300.0])
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=3)
        assert similarity > 0.0
        assert len(warp_func) == 4
        assert np.all((warp_func >= 0) & (warp_func <= 2))

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_full_python_dp_stage(self):
        """Contours long enough to hit full Python DP (early + general stage)."""
        u1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 400.0, 300.0])
        u2 = np.array([100.0, 200.0, 300.0, 350.0, 400.0, 500.0, 400.0, 300.0])
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=3)
        assert 0.0 <= similarity <= 100.0
        assert len(warp_func) == len(u1)

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_python_dp_warp_factor_2(self):
        """Python DP with warp_factor_level=2 to hit different loop bounds."""
        # length ratio < 2: m=4, n=5 => 5/3 < 2
        u1 = np.array([100.0, 200.0, 300.0, 400.0])
        u2 = np.array([100.0, 150.0, 200.0, 250.0, 400.0])
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=2)
        assert 0.0 <= similarity <= 100.0
        assert len(warp_func) == len(u1)

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_python_dp_identical_longer(self):
        """Python path with identical contours (diagonal path, general stage)."""
        u1 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
        u2 = u1.copy()
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=3)
        assert_allclose(similarity, 100.0, rtol=1e-5)
        assert_array_equal(warp_func, np.arange(len(u1)))


class TestUnwarp:
    """Tests for unwarp function."""
    
    def test_identity_warp(self):
        """Test unwarp on identity mapping."""
        warp_func = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        unwarp_func = unwarp(warp_func)
        
        assert_array_equal(unwarp_func, warp_func)
    
    def test_compression_warp(self):
        """Test unwarp on time compression."""
        # every other element
        warp_func = np.array([0, 2, 4, 6], dtype=np.int32)
        unwarp_func = unwarp(warp_func)
        
        assert len(unwarp_func) == 7  # 0 through 6
        assert unwarp_func[0] == 0
        assert unwarp_func[2] == 1
        assert unwarp_func[4] == 2
        # forward-fill for missing indices
        assert unwarp_func[1] == unwarp_func[0]
    
    def test_expansion_warp(self):
        """Test unwarp on time expansion."""
        # Repeated elements
        warp_func = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        unwarp_func = unwarp(warp_func)
        
        assert len(unwarp_func) == 3
        # first occurrence of each index
        assert unwarp_func[0] == 0
        assert unwarp_func[1] == 2
        assert unwarp_func[2] == 4
    
    def test_empty_warp(self):
        """Test unwarp on empty array."""
        warp_func = np.array([], dtype=np.int32)
        unwarp_func = unwarp(warp_func)
        
        assert len(unwarp_func) == 0


class TestUnwarpPythonPath:
    """Run unwarp with Python fallback (NUMBA_AVAILABLE=False) for coverage."""

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_identity_warp_python_path(self):
        warp_func = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        unwarp_func = unwarp(warp_func)
        assert_array_equal(unwarp_func, warp_func)

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_compression_warp_python_path(self):
        warp_func = np.array([0, 2, 4, 6], dtype=np.int32)
        unwarp_func = unwarp(warp_func)
        assert len(unwarp_func) == 7
        assert unwarp_func[0] == 0
        assert unwarp_func[2] == 1
        assert unwarp_func[1] == unwarp_func[0]

    @patch("artwarp.core.dtw.NUMBA_AVAILABLE", False)
    def test_expansion_warp_python_path(self):
        warp_func = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        unwarp_func = unwarp(warp_func)
        assert len(unwarp_func) == 3
        assert unwarp_func[0] == 0
        assert unwarp_func[1] == 2
        assert unwarp_func[2] == 4


class TestDTWPerformance:
    """Performance and stress tests for DTW."""
    
    @pytest.mark.slow
    def test_large_contours(self):
        """Test DTW on large contours (stress test)."""
        np.random.seed(42)
        # large contours
        u1 = 1000.0 + 500.0 * np.sin(np.linspace(0, 4*np.pi, 500))
        u2 = 1000.0 + 500.0 * np.sin(np.linspace(0, 4*np.pi, 600))
        
        similarity, warp_func = dynamic_time_warp(u1, u2, warp_factor_level=3)
        
        # should complete without error
        assert similarity > 0.0
        assert len(warp_func) == len(u1)
    
    def test_many_small_warps(self):
        """Test many DTW operations (typical use case)."""
        np.random.seed(42)
        base_contour = np.array([100.0, 200.0, 300.0, 400.0, 300.0, 200.0, 100.0])
        
        # simulate multiple comparisons
        for i in range(10):
            noise = np.random.randn(len(base_contour)) * 10.0
            noisy_contour = base_contour + noise
            
            similarity, warp_func = dynamic_time_warp(base_contour, noisy_contour)
            
            # high similarity despite noise
            assert similarity > 70.0
