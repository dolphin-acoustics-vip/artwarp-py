"""
Unit tests for visualization functions.

Tests the visualization module to ensure proper figure creation,
correct data representation, and error handling.

@author: Pedro Gronda Garrigues
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

# try to import matplotlib, skip tests if not available
try:
    import matplotlib

    matplotlib.use("Agg")  # use non-interactive backend for testing
    import matplotlib.pyplot as plt

    from artwarp.visualization import (
        create_paper_figure,
        create_results_report,
        plot_art_schematic,
        plot_category_distribution,
        plot_category_dendrogram,
        plot_category_embedding,
        plot_category_similarity_matrix,
        plot_confusion_matrix,
        plot_contour_length_distribution,
        plot_contours_by_category,
        plot_convergence_history,
        plot_discovery_curve,
        plot_dtw_alignment,
        plot_label_vs_category,
        plot_match_distribution,
        plot_per_category_match_quality,
        plot_reference_contours,
        plot_resampling_before_after,
        plot_run_stability,
        plot_training_summary,
        plot_vigilance_sweep,
        plot_warp_constraint,
    )

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from artwarp import ARTwarp
from artwarp.core.network import TrainingResults


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestVisualizationFunctions:
    """Tests for visualization functions."""

    @pytest.fixture
    def sample_results(self):
        """Create sample training results for testing."""
        # simple synthetic data
        contours = [
            np.array([100.0, 200.0, 300.0]),
            np.array([110.0, 210.0, 310.0]),
            np.array([500.0, 600.0, 700.0]),
        ]

        network = ARTwarp(vigilance=85.0, random_seed=42, verbose=False)
        results = network.fit(contours)

        return results, contours

    def test_plot_training_summary(self, sample_results):
        """Test training summary plot creation."""
        results, contours = sample_results
        names = ["contour_1", "contour_2", "contour_3"]

        fig = plot_training_summary(results, names, figsize=(16, 10))

        assert fig is not None
        assert len(fig.get_axes()) > 0  # should have multiple subplots
        plt.close(fig)

    def test_plot_reference_contours(self, sample_results):
        """Test reference contours plot."""
        results, _ = sample_results

        fig = plot_reference_contours(results.weight_matrix)

        assert fig is not None
        assert len(fig.get_axes()) >= results.num_categories
        plt.close(fig)

    def test_plot_reference_contours_empty(self):
        """Test reference contours with empty weight matrix."""
        weight_matrix = np.zeros((10, 0))

        with pytest.raises(ValueError, match="No categories"):
            plot_reference_contours(weight_matrix)

    def test_plot_category_distribution(self, sample_results):
        """Test category distribution plot."""
        results, _ = sample_results

        fig = plot_category_distribution(results, figsize=(10, 6))

        assert fig is not None
        assert len(fig.get_axes()) == 1
        plt.close(fig)

    def test_plot_convergence_history(self, sample_results):
        """Test convergence history plot."""
        results, _ = sample_results

        fig = plot_convergence_history(results, figsize=(10, 6))

        assert fig is not None
        assert len(fig.get_axes()) == 1
        plt.close(fig)

    def test_plot_convergence_history_no_data(self):
        """Test convergence history with no iteration history."""
        results = TrainingResults(
            categories=np.array([0, 1]),
            matches=np.array([90.0, 85.0]),
            weight_matrix=np.zeros((5, 2)),
            num_categories=2,
            num_iterations=1,
            converged=True,
            iteration_history=[],  # empty history
            training_time=1.0,
        )

        fig = plot_convergence_history(results)

        assert fig is not None
        plt.close(fig)

    def test_plot_contours_by_category(self, sample_results):
        """Test per-category contours plot."""
        results, contours = sample_results
        names = ["contour_1", "contour_2", "contour_3"]

        # first category
        category_sizes = results.get_category_sizes()
        if len(category_sizes) > 0:
            first_cat = sorted(category_sizes.keys())[0]

            # ref contour
            ref_contour = results.weight_matrix[:, first_cat]
            valid_mask = ~np.isnan(ref_contour)
            ref_contour_clean = ref_contour[valid_mask]

            fig = plot_contours_by_category(
                contours, results.categories, first_cat, names, ref_contour_clean
            )

            assert fig is not None
            plt.close(fig)

    def test_plot_contours_by_category_empty(self, sample_results):
        """Test per-category plot with non-existent category."""
        results, contours = sample_results

        # Use a category ID that doesn't exist
        fig = plot_contours_by_category(contours, results.categories, 999, contour_names=None)

        assert fig is not None
        plt.close(fig)

    def test_plot_match_distribution(self, sample_results):
        """Test match distribution plot."""
        results, _ = sample_results

        fig = plot_match_distribution(results, figsize=(10, 6))

        assert fig is not None
        assert len(fig.get_axes()) == 1
        plt.close(fig)

    def test_plot_discovery_curve(self, sample_results):
        """Test discovery curve plot (cumulative categories vs sample order)."""
        results, _ = sample_results

        fig = plot_discovery_curve(results, title="Test species", figsize=(10, 6))

        assert fig is not None
        assert len(fig.get_axes()) == 1
        plt.close(fig)

    def test_plot_discovery_curve_no_data(self):
        """Test discovery curve with no samples."""
        results = TrainingResults(
            categories=np.array([]),
            matches=np.array([]),
            weight_matrix=np.zeros((5, 0)),
            num_categories=0,
            num_iterations=0,
            converged=False,
            iteration_history=[],
            training_time=0.0,
        )
        fig = plot_discovery_curve(results)
        assert fig is not None
        plt.close(fig)

    def test_plot_match_distribution_no_categories(self):
        """Test match distribution with no categorized samples."""
        results = TrainingResults(
            categories=np.array([np.nan, np.nan]),
            matches=np.array([0.0, 0.0]),
            weight_matrix=np.zeros((5, 0)),
            num_categories=0,
            num_iterations=1,
            converged=False,
            iteration_history=[],
            training_time=1.0,
        )

        fig = plot_match_distribution(results)

        assert fig is not None
        plt.close(fig)

    def test_create_results_report(self, sample_results, tmp_path):
        """Test comprehensive report generation."""
        results, contours = sample_results
        names = ["contour_1", "contour_2", "contour_3"]

        output_dir = tmp_path / "test_report"

        saved_files = create_results_report(
            results,
            contours,
            names,
            output_dir=str(output_dir),
            dpi=100,  # low DPI for faster tests
        )

        # check that files were created
        assert len(saved_files) > 0
        assert "training_summary" in saved_files
        assert "reference_contours" in saved_files
        assert "discovery_curve" in saved_files

        # verify files exist
        for path in saved_files.values():
            assert Path(path).exists()

    def test_figure_save_functionality(self, sample_results, tmp_path):
        """Test that figures can be saved to files."""
        results, _ = sample_results

        output_file = tmp_path / "test_figure.png"

        fig = plot_category_distribution(results, save_path=str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0
        plt.close(fig)

    def test_custom_figure_sizes(self, sample_results):
        """Test custom figure size specification."""
        results, _ = sample_results

        # test different figure sizes
        for figsize in [(8, 6), (12, 8), (16, 10)]:
            fig = plot_category_distribution(results, figsize=figsize)
            assert fig.get_size_inches().tolist() == list(figsize)
            plt.close(fig)

    def test_dpi_specification(self, sample_results, tmp_path):
        """Test DPI specification for saved figures."""
        results, _ = sample_results

        output_file = tmp_path / "test_dpi.png"

        # different DPI values
        for dpi in [72, 150, 300]:
            fig = plot_category_distribution(results, save_path=str(output_file), dpi=dpi)
            plt.close(fig)

            assert output_file.exists()
            output_file.unlink()  # Remove for next iteration


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestVisualizationEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_category(self):
        """Test visualization with only one category."""
        contours = [np.array([100.0, 200.0, 300.0])] * 3
        network = ARTwarp(vigilance=85.0, random_seed=42, verbose=False)
        results = network.fit(contours)

        fig = plot_reference_contours(results.weight_matrix)
        assert fig is not None
        plt.close(fig)

    def test_many_categories(self):
        """Test visualization with many categories."""
        # diverse contours => many categories
        contours = [np.array([100.0 * i, 200.0 * i, 300.0 * i]) for i in range(1, 11)]
        network = ARTwarp(vigilance=99.0, random_seed=42, verbose=False)
        results = network.fit(contours)

        fig = plot_reference_contours(results.weight_matrix)
        assert fig is not None
        plt.close(fig)

    def test_with_uncategorized_samples(self):
        """Test visualization with uncategorized samples."""
        contours = [
            np.array([100.0, 200.0, 300.0]),
            np.array([10000.0, 20000.0, 30000.0]),  # very different
        ]
        network = ARTwarp(
            vigilance=95.0, max_categories=1, random_seed=42, verbose=False  # limit to 1 category
        )
        results = network.fit(contours)

        # uncategorized samples handled
        fig = plot_category_distribution(results)
        assert fig is not None
        plt.close(fig)

    def test_very_many_categories_renders(self):
        """Test that training summary, category distribution, and reference contours
        render correctly with >25 categories (legend off, thinned x-axis labels)."""
        n_cats = 30
        n_features = 20
        n_samples = 50
        # synthetic: 30 categories, weight (n_features, n_cats)
        weight_matrix = np.random.RandomState(42).randn(n_features, n_cats) * 10 + 100
        weight_matrix[:, 0] = np.nan  # one col NaN (edge case)
        categories = np.random.RandomState(43).randint(0, n_cats, size=n_samples).astype(float)
        categories[0] = np.nan  # one uncategorized
        matches = np.random.RandomState(44).uniform(70, 100, size=n_samples)
        results = TrainingResults(
            categories=categories,
            matches=matches,
            weight_matrix=weight_matrix,
            num_categories=n_cats,
            num_iterations=10,
            converged=True,
            iteration_history=[(i, 2) for i in range(10)],
            training_time=1.5,
        )
        # training summary: ref contours panel, no per-cat legend
        fig1 = plot_training_summary(results, figsize=(12, 8))
        assert fig1 is not None
        assert len(fig1.get_axes()) >= 6
        plt.close(fig1)
        # standalone category dist: thinned x-axis labels
        fig2 = plot_category_distribution(results, figsize=(10, 6))
        assert fig2 is not None
        assert len(fig2.get_axes()) == 1
        plt.close(fig2)
        # ref contours grid: many categories
        fig3 = plot_reference_contours(weight_matrix, figsize=(14, 10))
        assert fig3 is not None
        plt.close(fig3)


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
class TestNewVisualizations:
    """Tests for additional visualizations (algorithm, diagnostics, reporting)."""

    @pytest.fixture
    def sample_results(self):
        """Create sample training results for new visualization tests."""
        contours = [
            np.array([100.0, 200.0, 300.0]),
            np.array([110.0, 210.0, 310.0]),
            np.array([500.0, 600.0, 700.0]),
        ]
        network = ARTwarp(vigilance=85.0, random_seed=42, verbose=False)
        results = network.fit(contours)
        return results, contours

    def test_plot_dtw_alignment(self):
        """Test DTW alignment plot between two contours."""
        u1 = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
        u2 = np.array([105.0, 160.0, 210.0, 260.0, 310.0])
        fig = plot_dtw_alignment(u1, u2, warp_factor_level=3, figsize=(8, 6))
        assert fig is not None
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_plot_art_schematic(self):
        """Test ART decision schematic (static diagram)."""
        fig = plot_art_schematic(figsize=(8, 5))
        assert fig is not None
        plt.close(fig)

    def test_plot_warp_constraint(self):
        """Test Itakura warp constraint illustration."""
        fig = plot_warp_constraint(warp_factor_level=3, m=15, n=20, figsize=(6, 6))
        assert fig is not None
        plt.close(fig)

    def test_plot_per_category_match_quality(self, sample_results):
        """Test per-category match quality violin plot."""
        results, _ = sample_results
        fig = plot_per_category_match_quality(results, figsize=(8, 5))
        assert fig is not None
        plt.close(fig)

    def test_plot_category_similarity_matrix(self, sample_results):
        """Test category similarity matrix heatmap."""
        results, _ = sample_results
        fig = plot_category_similarity_matrix(results.weight_matrix, warp_factor_level=3)
        assert fig is not None
        plt.close(fig)

    def test_plot_category_embedding(self, sample_results):
        """Test category MDS embedding."""
        results, _ = sample_results
        fig = plot_category_embedding(results.weight_matrix, warp_factor_level=3)
        assert fig is not None
        plt.close(fig)

    def test_plot_resampling_before_after(self):
        """Test resampling before/after plot."""
        contour = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
        fig = plot_resampling_before_after(contour, tempres=0.01, sample_interval_sec=0.02)
        assert fig is not None
        assert len(fig.get_axes()) == 2
        plt.close(fig)

    def test_plot_contour_length_distribution(self):
        """Test contour length and tempres distribution histograms."""
        contours = [np.array([1.0, 2.0] * i) for i in range(5, 15)]
        fig = plot_contour_length_distribution(contours, figsize=(10, 4))
        assert fig is not None
        plt.close(fig)

    def test_plot_contour_length_distribution_with_tempres(self):
        """Test contour length distribution with tempres list."""
        contours = [np.array([1.0, 2.0, 3.0]) for _ in range(5)]
        tempres_list = [0.01, 0.02, 0.01, 0.02, 0.01]
        fig = plot_contour_length_distribution(contours, tempres_list=tempres_list)
        assert fig is not None
        plt.close(fig)

    def test_plot_vigilance_sweep(self, sample_results):
        """Test vigilance sweep plot."""
        results, contours = sample_results
        sweep = []
        for v in [80.0, 85.0, 90.0]:
            net = ARTwarp(vigilance=v, random_seed=42, verbose=False)
            sweep.append((v, net.fit(contours)))
        fig = plot_vigilance_sweep(sweep, figsize=(8, 5))
        assert fig is not None
        plt.close(fig)

    def test_plot_run_stability(self):
        """Test run stability histogram."""
        num_cats = [3, 4, 3, 4, 3, 4, 3]
        fig = plot_run_stability(num_cats, figsize=(6, 4))
        assert fig is not None
        plt.close(fig)

    def test_create_paper_figure(self, sample_results):
        """Test paper-ready multi-panel figure."""
        results, contours = sample_results
        names = ["a", "b", "c"]
        fig = create_paper_figure(results, contours, contour_names=names, figsize=(12, 8))
        assert fig is not None
        assert len(fig.get_axes()) >= 3
        plt.close(fig)

    def test_plot_category_dendrogram(self, sample_results):
        """Test category dendrogram (requires scipy)."""
        pytest.importorskip("scipy")
        results, _ = sample_results
        fig = plot_category_dendrogram(results.weight_matrix, warp_factor_level=3)
        assert fig is not None
        plt.close(fig)

    def test_plot_confusion_matrix(self, sample_results):
        """Test confusion matrix with ground truth labels."""
        results, _ = sample_results
        ground_truth = ["A", "B", "A"]
        fig = plot_confusion_matrix(ground_truth, results.categories, figsize=(6, 5))
        assert fig is not None
        plt.close(fig)

    def test_plot_label_vs_category(self, sample_results):
        """Test label vs category stacked bar."""
        results, _ = sample_results
        ground_truth = ["A", "B", "A"]
        fig = plot_label_vs_category(ground_truth, results.categories, figsize=(8, 5))
        assert fig is not None
        plt.close(fig)
