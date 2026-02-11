"""
Simple example demonstrating ARTwarp usage.

This script creates synthetic frequency contours and categorizes them
using the ARTwarp algorithm.

@author: Pedro Gronda Garrigues
"""

import numpy as np
from artwarp import ARTwarp

def create_synthetic_contours():
    """Create synthetic frequency contours for testing."""
    # create three types of contours with some variation
    
    # Type 1: Ascending contours (100-300 Hz)
    type1 = [
        np.array([100.0, 150.0, 200.0, 250.0, 300.0]) + np.random.randn(5) * 5
        for _ in range(5)
    ]
    
    # Type 2: Descending contours (300-100 Hz)
    type2 = [
        np.array([300.0, 250.0, 200.0, 150.0, 100.0]) + np.random.randn(5) * 5
        for _ in range(5)
    ]
    
    # Type 3: U-shaped contours
    type3 = [
        np.array([200.0, 150.0, 100.0, 150.0, 200.0]) + np.random.randn(5) * 5
        for _ in range(5)
    ]
    
    # combine all contours
    all_contours = type1 + type2 + type3
    
    # create names
    names = (
        [f"ascending_{i}" for i in range(5)] +
        [f"descending_{i}" for i in range(5)] +
        [f"ushaped_{i}" for i in range(5)]
    )
    
    return all_contours, names


def main():
    """Run simple ARTwarp example."""
    print("ARTwarp Simple Example")
    print("=" * 60)
    
    # create synthetic data
    print("\n1. Creating synthetic frequency contours...")
    contours, names = create_synthetic_contours()
    print(f"   Created {len(contours)} contours")
    
    # init network
    print("\n2. Initializing ARTwarp network...")
    network = ARTwarp(
        vigilance=85.0,
        learning_rate=0.1,
        bias=0.0,
        max_categories=10,
        max_iterations=50,
        warp_factor_level=3,
        random_seed=42,
        verbose=True
    )
    
    # train
    print("\n3. Training network...")
    results = network.fit(contours, contour_names=names)
    
    # display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nTotal contours:     {len(contours)}")
    print(f"Categories created: {results.num_categories}")
    print(f"Iterations:         {results.num_iterations}")
    print(f"Converged:          {results.converged}")
    print(f"Training time:      {results.training_time:.2f} seconds")
    
    print("\nCategory assignments:")
    for i, (name, cat, match) in enumerate(zip(names, results.categories, results.matches)):
        cat_str = f"{int(cat)}" if not np.isnan(cat) else "None"
        print(f"  {name:20s} -> Category {cat_str:3s} (match: {match:5.1f}%)")
    
    print("\nCategory sizes:")
    for cat, size in sorted(results.get_category_sizes().items()):
        print(f"  Category {cat}: {size} contours")
    
    if results.get_uncategorized_count() > 0:
        print(f"\nUncategorized: {results.get_uncategorized_count()} contours")
    
    # test prediction
    print("\n4. Testing prediction on new contours...")
    test_contours = [
        np.array([105.0, 155.0, 205.0, 255.0, 305.0]),  # similar to type 1
        np.array([305.0, 255.0, 205.0, 155.0, 105.0]),  # similar to type 2
    ]
    test_names = ["test_ascending", "test_descending"]
    
    pred_categories, pred_matches = network.predict(test_contours)
    
    print("\nPredictions:")
    for name, cat, match in zip(test_names, pred_categories, pred_matches):
        cat_str = f"{int(cat)}" if not np.isnan(cat) else "None"
        print(f"  {name:20s} -> Category {cat_str:3s} (match: {match:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
