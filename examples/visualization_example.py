"""
Comprehensive visualization example for ARTwarp.

This script demonstrates all visualization capabilities of ARTwarp-py,
including:
- Training summary plots
- Reference contour visualization
- Category distribution analysis
- Convergence history
- Per-category contour plots
- Comprehensive report generation

@author: Pedro Gronda Garrigues
"""

import numpy as np
from artwarp import ARTwarp

# import visualization functions
from artwarp.visualization import (
    plot_training_summary,
    plot_reference_contours,
    plot_category_distribution,
    plot_convergence_history,
    plot_contours_by_category,
    plot_match_distribution,
    create_results_report,
)


def create_synthetic_contours():
    """Create synthetic frequency contours for demonstration."""
    np.random.seed(42)
    
    # Type 1: Ascending contours (100-300 Hz)
    type1 = [
        np.array([100.0, 150.0, 200.0, 250.0, 300.0]) + np.random.randn(5) * 10
        for _ in range(8)
    ]
    
    # Type 2: Descending contours (300-100 Hz)
    type2 = [
        np.array([300.0, 250.0, 200.0, 150.0, 100.0]) + np.random.randn(5) * 10
        for _ in range(8)
    ]
    
    # Type 3: U-shaped contours
    type3 = [
        np.array([200.0, 150.0, 100.0, 150.0, 200.0]) + np.random.randn(5) * 10
        for _ in range(8)
    ]
    
    # Type 4: Inverted U-shaped
    type4 = [
        np.array([100.0, 150.0, 200.0, 150.0, 100.0]) + np.random.randn(5) * 10
        for _ in range(8)
    ]
    
    # combine all contours
    all_contours = type1 + type2 + type3 + type4
    
    # create names
    names = (
        [f"ascending_{i}" for i in range(8)] +
        [f"descending_{i}" for i in range(8)] +
        [f"ushaped_{i}" for i in range(8)] +
        [f"inverted_u_{i}" for i in range(8)]
    )
    
    return all_contours, names


def main():
    """Run comprehensive visualization example."""
    print("=" * 70)
    print("ARTwarp Visualization Example")
    print("=" * 70)
    
    # create synthetic data
    print("\n1. Creating synthetic frequency contours...")
    contours, names = create_synthetic_contours()
    print(f"   Created {len(contours)} contours across 4 types")
    
    # train network
    print("\n2. Training ARTwarp network...")
    network = ARTwarp(
        vigilance=85.0,
        learning_rate=0.1,
        bias=0.0,
        max_categories=10,
        max_iterations=50,
        warp_factor_level=3,
        random_seed=42,
        verbose=False  # suppress training output -> cleaner demo
    )
    
    results = network.fit(contours, contour_names=names)
    
    print(f"   Training complete!")
    print(f"   - Categories: {results.num_categories}")
    print(f"   - Iterations: {results.num_iterations}")
    print(f"   - Converged: {results.converged}")
    print(f"   - Time: {results.training_time:.2f}s")
    
    # individual visualization examples
    print("\n3. Creating individual visualizations...")
    
    # Plot 1: Training Summary (comprehensive overview)
    print("   a) Creating training summary plot...")
    try:
        import matplotlib.pyplot as plt
        
        fig1 = plot_training_summary(results, names, figsize=(16, 10))
        plt.savefig('visualization_training_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print("      ✓ Saved: visualization_training_summary.png")
    except ImportError:
        print("      ✗ Matplotlib not available")
        return
    
    # Plot 2: Reference Contours
    print("   b) Creating reference contours plot...")
    fig2 = plot_reference_contours(results.weight_matrix, figsize=(12, 8))
    plt.savefig('visualization_reference_contours.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("      ✓ Saved: visualization_reference_contours.png")
    
    # Plot 3: Category Distribution
    print("   c) Creating category distribution plot...")
    fig3 = plot_category_distribution(results, figsize=(10, 6))
    plt.savefig('visualization_category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("      ✓ Saved: visualization_category_distribution.png")
    
    # Plot 4: Convergence History
    print("   d) Creating convergence history plot...")
    fig4 = plot_convergence_history(results, figsize=(10, 6))
    plt.savefig('visualization_convergence_history.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("      ✓ Saved: visualization_convergence_history.png")
    
    # Plot 5: Match Distribution
    print("   e) Creating match distribution plot...")
    fig5 = plot_match_distribution(results, figsize=(10, 6))
    plt.savefig('visualization_match_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print("      ✓ Saved: visualization_match_distribution.png")
    
    # Plot 6: Per-category contours (for first category)
    print("   f) Creating per-category contours plot...")
    category_sizes = results.get_category_sizes()
    if len(category_sizes) > 0:
        first_category = sorted(category_sizes.keys())[0]
        ref_contour = results.weight_matrix[:, first_category]
        valid_mask = ~np.isnan(ref_contour)
        ref_contour_clean = ref_contour[valid_mask]
        
        fig6 = plot_contours_by_category(
            contours, results.categories, first_category,
            names, ref_contour_clean, figsize=(12, 8)
        )
        plt.savefig(f'visualization_category_{first_category}_contours.png', 
                   dpi=300, bbox_inches='tight')
        plt.close(fig6)
        print(f"      ✓ Saved: visualization_category_{first_category}_contours.png")
    
    # comprehensive report
    print("\n4. Creating comprehensive visualization report...")
    report_files = create_results_report(
        results, contours, names,
        output_dir='./visualization_report',
        dpi=300
    )
    
    print(f"\n   Report generated with {len(report_files)} figures:")
    for name, path in sorted(report_files.items()):
        print(f"      - {name}: {path}")
    
    # print summary statistics
    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)
    print(f"\nIndividual Plots: 6 figures saved")
    print(f"Comprehensive Report: {len(report_files)} figures in ./visualization_report/")
    print("\nFigures created:")
    print("  1. training_summary.png - Complete overview with multiple panels")
    print("  2. reference_contours.png - Category prototypes")
    print("  3. category_distribution.png - Sample distribution across categories")
    print("  4. convergence_history.png - Training convergence over iterations")
    print("  5. match_distribution.png - Distribution of match scores")
    print("  6. category_X_contours.png - Individual contours per category")
    
    print("\nAll visualizations use professional scientific plotting standards:")
    print("  ✓ High resolution (300 DPI)")
    print("  ✓ Clear labels and titles")
    print("  ✓ Consistent color schemes")
    print("  ✓ Grid lines for readability")
    print("  ✓ Publication-ready quality")
    
    print("\n" + "=" * 70)
    print("Visualization example completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
