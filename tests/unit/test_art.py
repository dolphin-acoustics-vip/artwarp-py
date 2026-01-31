"""
Unit tests for ART neural network components.

Tests category activation, match calculation, and resonance checking.

@author: Pedro Gronda Garrigues
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from artwarp.core.art import (
    activate_categories,
    calculate_match,
    sort_categories_by_activation,
    check_resonance
)


class TestActivateCategories:
    """Tests for category activation function."""
    
    def test_empty_weight_matrix(self):
        """Test activation with no categories yet."""
        input_contour = np.array([100.0, 200.0, 300.0])
        weight_matrix = np.zeros((10, 0))  # no categories
        bias = 0.0
        
        activations, warp_funcs = activate_categories(
            input_contour, weight_matrix, bias
        )
        
        assert len(activations) == 0
        assert len(warp_funcs) == 0
    
    def test_single_category_perfect_match(self):
        """Test activation with one category that matches perfectly."""
        input_contour = np.array([100.0, 200.0, 300.0])
        weight_matrix = np.full((10, 1), np.nan)
        weight_matrix[:3, 0] = input_contour  # Perfect match
        bias = 0.0
        
        activations, warp_funcs = activate_categories(
            input_contour, weight_matrix, bias
        )
        
        assert len(activations) == 1
        assert_allclose(activations[0], 100.0, rtol=1e-6)
        assert len(warp_funcs[0]) == len(input_contour)
    
    def test_bias_effect(self):
        """Test that bias reduces activation."""
        input_contour = np.array([100.0, 200.0, 300.0])
        weight_matrix = np.full((10, 1), np.nan)
        weight_matrix[:3, 0] = input_contour
        
        # different bias values
        act_no_bias, _ = activate_categories(input_contour, weight_matrix, 0.0)
        act_half_bias, _ = activate_categories(input_contour, weight_matrix, 0.5)
        act_high_bias, _ = activate_categories(input_contour, weight_matrix, 0.9)
        
        # Higher bias should give lower activation
        assert act_no_bias[0] > act_half_bias[0]
        assert act_half_bias[0] > act_high_bias[0]
        
        # spot-check values
        assert_allclose(act_half_bias[0], act_no_bias[0] * 0.5, rtol=1e-6)
        assert_allclose(act_high_bias[0], act_no_bias[0] * 0.1, rtol=1e-6)
    
    def test_multiple_categories(self):
        """Test activation with multiple categories."""
        input_contour = np.array([100.0, 200.0, 300.0])
        weight_matrix = np.full((10, 3), np.nan)
        
        # cat 1: exact match
        weight_matrix[:3, 0] = input_contour
        # cat 2: different contour
        weight_matrix[:3, 1] = [150.0, 250.0, 350.0]
        # cat 3: very different
        weight_matrix[:3, 2] = [500.0, 600.0, 700.0]
        
        activations, warp_funcs = activate_categories(
            input_contour, weight_matrix, 0.0
        )
        
        assert len(activations) == 3
        # Category 1 should have highest activation
        assert activations[0] > activations[1]
        assert activations[1] > activations[2]


class TestCalculateMatch:
    """Tests for match calculation function."""
    
    def test_identical_contours(self):
        """Test perfect match with identical contours."""
        contour1 = np.array([100.0, 200.0, 300.0])
        contour2 = np.array([100.0, 200.0, 300.0])
        
        match = calculate_match(contour1, contour2)
        
        assert_allclose(match, 100.0, rtol=1e-10)
    
    def test_different_contours(self):
        """Test match with different contours."""
        contour1 = np.array([100.0, 200.0])
        contour2 = np.array([200.0, 400.0])  # doubled
        
        match = calculate_match(contour1, contour2)
        
        # each point 50% similarity => min/max = 0.5
        assert_allclose(match, 50.0, rtol=1e-10)
    
    def test_empty_contours(self):
        """Test match with empty contours."""
        contour1 = np.array([])
        contour2 = np.array([])
        
        match = calculate_match(contour1, contour2)
        
        assert match == 0.0
    
    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        contour1 = np.array([100.0, 200.0])
        contour2 = np.array([100.0, 200.0, 300.0])
        
        with pytest.raises(ValueError, match="must match"):
            calculate_match(contour1, contour2)

    def test_match_uses_only_positive_weights(self):
        """MATLAB: match uses only positions where weight > 0 (find(weightVector > 0))."""
        # same-length; weight has one zero (excluded)
        input_contour = np.array([100.0, 200.0, 300.0])
        weight_vector = np.array([100.0, 0.0, 300.0])  # middle 0
        match = calculate_match(input_contour, weight_vector)
        # only positions 0 and 2 count => 100
        assert_allclose(match, 100.0, rtol=1e-10)
    
    def test_match_formula(self):
        """Test that match formula is correct."""
        # we can compute expected: point 0 => 100/150, point 1 => 1.0, point 2 => 300/600 => avg ~72.22%
        contour1 = np.array([100.0, 200.0, 300.0])
        contour2 = np.array([150.0, 200.0, 600.0])
        
        match = calculate_match(contour1, contour2)
        expected = ((100.0/150.0 + 1.0 + 0.5) / 3.0) * 100.0
        
        assert_allclose(match, expected, rtol=1e-10)


class TestSortCategoriesByActivation:
    """Tests for category sorting function."""
    
    def test_descending_order(self):
        """Test that categories are sorted in descending order."""
        activations = np.array([30.0, 90.0, 50.0, 70.0])
        
        sorted_acts, sorted_indices = sort_categories_by_activation(activations)
        
        # descending order
        assert_array_equal(sorted_acts, [90.0, 70.0, 50.0, 30.0])
        assert_array_equal(sorted_indices, [1, 3, 2, 0])
    
    def test_empty_activations(self):
        """Test sorting empty activation array."""
        activations = np.array([])
        
        sorted_acts, sorted_indices = sort_categories_by_activation(activations)
        
        assert len(sorted_acts) == 0
        assert len(sorted_indices) == 0
    
    def test_single_activation(self):
        """Test sorting single activation."""
        activations = np.array([50.0])
        
        sorted_acts, sorted_indices = sort_categories_by_activation(activations)
        
        assert_array_equal(sorted_acts, [50.0])
        assert_array_equal(sorted_indices, [0])
    
    def test_duplicate_activations(self):
        """Test sorting with duplicate activation values."""
        activations = np.array([50.0, 80.0, 50.0, 80.0])
        
        sorted_acts, sorted_indices = sort_categories_by_activation(activations)
        
        # duplicate order may vary, sorted overall
        assert sorted_acts[0] >= sorted_acts[1]
        assert sorted_acts[1] >= sorted_acts[2]
        assert sorted_acts[2] >= sorted_acts[3]


class TestCheckResonance:
    """Tests for resonance checking function."""
    
    def test_match_above_vigilance(self):
        """Test resonance when match exceeds vigilance."""
        assert check_resonance(85.5, 85.0) is True
        assert check_resonance(90.0, 85.0) is True
        assert check_resonance(100.0, 99.0) is True
    
    def test_match_below_vigilance(self):
        """Test no resonance when match below vigilance."""
        assert check_resonance(84.9, 85.0) is False
        assert check_resonance(50.0, 85.0) is False
        assert check_resonance(0.0, 85.0) is False
    
    def test_match_equals_vigilance(self):
        """Test that equal match and vigilance gives no resonance."""
        # MATLAB uses > (not >=)
        assert check_resonance(85.0, 85.0) is False
    
    def test_boundary_conditions(self):
        """Test boundary conditions for vigilance."""
        # very high vigilance
        assert check_resonance(99.9, 99.0) is True
        assert check_resonance(98.9, 99.0) is False
        
        # very low vigilance
        assert check_resonance(1.1, 1.0) is True
        assert check_resonance(0.9, 1.0) is False
