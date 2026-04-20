"""Unit tests for weight matrix helpers (MATLAB parity)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from artwarp.core.weights import (
    average_weights,
    delete_category_reindex_assignments,
    purge_empty_category_columns,
    update_weights,
)


class TestAverageWeights:
    """ARTwarp_Average_Weights.m — empty categories keep prior columns."""

    def test_skips_categories_with_no_assigned_contours(self):
        w = np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
            ],
            dtype=np.float64,
        )
        lengths = np.array([3, 3], dtype=np.int64)
        cats = np.array([0.0, 1.0], dtype=np.float64)
        out = average_weights(w, lengths, cats)
        assert_allclose(out[:, 0], w[:, 0])
        assert_allclose(out[:, 1], w[:, 1])
        assert_allclose(out[:, 2], w[:, 2])

    def test_resamples_to_mean_length(self):
        w = np.zeros((5, 1), dtype=np.float64)
        w[:3, 0] = [1.0, 2.0, 3.0]
        lengths = np.array([5], dtype=np.int64)
        cats = np.array([0.0])
        out = average_weights(w, lengths, cats)
        assert out.shape == w.shape
        assert np.sum(np.isfinite(out[:, 0])) == 5


class TestDeleteCategoryReindex:
    """ARTwarp_Delete_Category.m"""

    def test_reindexes_assignments(self):
        w = np.ones((3, 3), dtype=np.float64)
        # column 1 has no assignments; categories 0 and 2 are used
        cats = np.array([0.0, 2.0, 0.0])
        nw, nc = delete_category_reindex_assignments(1, w, cats)
        assert nw.shape[1] == 2
        assert_array_equal(nc, np.array([0.0, 1.0, 0.0]))

    def test_rejects_delete_nonempty(self):
        w = np.ones((2, 2), dtype=np.float64)
        cats = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="non-empty"):
            delete_category_reindex_assignments(0, w, cats)


class TestPurgeEmptyCategoryColumns:
    """Marco's orphan-column purge (optional in ARTwarp.fit)."""

    def test_removes_unreferenced_middle_column(self):
        w = np.ones((2, 3), dtype=np.float64)
        c = np.array([0.0, 0.0, 2.0], dtype=np.float64)
        nw, nc, nd = purge_empty_category_columns(w, c)
        assert nd == 1
        assert nw.shape == (2, 2)
        assert_array_equal(nc, np.array([0.0, 0.0, 1.0]))

    def test_no_deletions_when_all_used(self):
        w = np.ones((2, 2), dtype=np.float64)
        c = np.array([0.0, 1.0], dtype=np.float64)
        nw, nc, nd = purge_empty_category_columns(w, c)
        assert nd == 0
        assert_array_equal(nw, w)
        assert_array_equal(nc, c)

    def test_rejects_category_index_out_of_range(self):
        w = np.ones((2, 2), dtype=np.float64)
        c = np.array([0.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="out of range"):
            purge_empty_category_columns(w, c)


class TestUpdateWeightsCompareWarped:
    """ART_Update_Weights compare_warped branch."""

    def test_compare_warped_keeps_weight_length(self):
        wm = np.full((5, 1), np.nan, dtype=np.float64)
        wm[:3, 0] = [10.0, 20.0, 30.0]
        inp = np.array([12.0, 22.0, 32.0, 40.0], dtype=np.float64)
        warp = np.array([0, 1, 2], dtype=np.int32)
        out = update_weights(inp, wm, 0, 0.5, warp, compare_warped=True)
        assert np.sum(np.isfinite(out[:, 0])) == 3
