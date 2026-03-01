# ARTwarp-py Visualization Guide

## Overview

ARTwarp-py includes visualization capabilities for analyzing and presenting training results. All visualization functions follow scientific plotting best practices and produce research-style figures (to be validated by actual researchers).

This plotting functionality, specific to CLI, serves as a MATLAB rendering alternative. They are generated using the pickle file generated from training (`train` command in CLI), providing good result encapsulation!

## Features

- **Training Summary**: Multi-panel overview of complete training results
- **Reference Contours**: Category prototypes visualization
- **Category Distribution**: Sample distribution across categories
- **Convergence History**: Training convergence monitoring
- **Per-Category Analysis**: Detailed view of individual categories
- **Match Distribution**: Statistical analysis of match scores
- **Discovery Curve**: Cumulative category discovery vs sample order (MATLAB DiscoveryCurves.m equivalent)
- **Reports**: Automated report generation
- **Algorithm**: DTW alignment plot, ART decision schematic, Itakura warp-constraint diagram
- **Diagnostics**: Per-category match quality, category similarity matrix, category embedding (MDS), category dendrogram
- **Data**: Resampling before/after, contour length and temporal resolution distribution
- **Parameter studies**: Vigilance sweep, run stability (multi-seed)
- **Publication**: Single paper-ready multi-panel figure
- **Ground truth**: Confusion matrix, label-vs-category distribution (when labels are available)

## Quick Start

```python
from artwarp import ARTwarp, load_contours
from artwarp.visualization import plot_training_summary

# train network
contours, names = load_contours('./data')
network = ARTwarp(vigilance=85.0)
results = network.fit(contours)

# create visualizations
import matplotlib.pyplot as plt
fig = plot_training_summary(results, contour_names=names)
plt.show()
```

## Available Functions

### 1. Training Summary

```python
from artwarp.visualization import plot_training_summary

fig = plot_training_summary(
    results,
    contour_names=names,
    figsize=(16, 10),
    save_path='training_summary.png',
    dpi=300
)
```

Creates a multi-panel figure showing:
- All reference contours overlaid
- Category size distribution
- Match value distribution
- Convergence history
- Category statistics table
- Match values by category (violin plot)

**Parameters:**
- `results`: TrainingResults object
- `contour_names`: Optional list of contour names
- `figsize`: Figure size in inches (width, height)
- `save_path`: Optional path to save figure
- `dpi`: Resolution for saved figure (default: 300, I suggest you don't change this)

### 2. Reference Contours

```python
from artwarp.visualization import plot_reference_contours

fig = plot_reference_contours(
    results.weight_matrix,
    figsize=(12, 8),
    save_path='reference_contours.png',
    show_grid=True
)
```

Displays all category prototypes (reference contours) in a grid layout.

**Parameters:**
- `weight_matrix`: Weight matrix from trained network
- `figsize`: Figure size (auto-calculated if None)
- `save_path`: Optional path to save figure
- `dpi`: Resolution for saved figure
- `show_grid`: Whether to show grid lines

### 3. Category Distribution

```python
from artwarp.visualization import plot_category_distribution

fig = plot_category_distribution(
    results,
    figsize=(10, 6),
    save_path='category_dist.png'
)
```

Bar chart showing # of samples in each category.

### 4. Convergence History

```python
from artwarp.visualization import plot_convergence_history

fig = plot_convergence_history(
    results,
    figsize=(10, 6),
    save_path='convergence.png'
)
```

Line plot showing reclassifications per iteration, indicating convergence.

### 5. Contours by Category

```python
from artwarp.visualization import plot_contours_by_category

# get ref contour for category
ref_contour = results.weight_matrix[:, category_id]
valid_mask = ~np.isnan(ref_contour)
ref_contour_clean = ref_contour[valid_mask]

fig = plot_contours_by_category(
    contours,
    results.categories,
    category_id=0,
    contour_names=names,
    reference_contour=ref_contour_clean,
    figsize=(12, 8),
    max_contours=20
)
```

Shows all contours assigned to a specific category with the reference contour overlaid.

**Parameters:**
- `contours`: List of frequency contour arrays
- `categories`: Category assignments
- `category_id`: Category to visualize
- `contour_names`: Optional contour names
- `reference_contour`: Optional reference contour to overlay
- `max_contours`: Maximum contours to plot (for readability)

### 6. Match Distribution

```python
from artwarp.visualization import plot_match_distribution

fig = plot_match_distribution(
    results,
    figsize=(10, 6),
    bins=30
)
```

Histogram showing distribution of match scores with mean and median lines.

### 7. Discovery Curve

```python
from artwarp.visualization import plot_discovery_curve

fig = plot_discovery_curve(
    results,
    title="Tursiops truncatus",  # optional: species or dataset name
    figsize=(10, 6),
    save_path='discovery_curve.png'
)
```

Plots the **category discovery curve**: for each sample index (in training order), the curve shows how many distinct categories have been observed so far. As you move through the dataset, the curve rises when a new category is encountered and stays flat when the sample belongs to an already-seen category. Equivalent to the original MATLAB `DiscoveryCurves.m` logic.

Useful for assessing how quickly the dataset reveals its category diversity and for comparing discovery rates across species or conditions.

**Parameters:**
- `results`: TrainingResults object (categories in sample order)
- `title`: Optional figure title (e.g. species or dataset name)
- `figsize`: Figure size in inches
- `save_path`: Optional path to save figure
- `dpi`: Resolution for saved figure (default: 300)

### 8. Report

```python
from artwarp.visualization import create_results_report

# basic usage
saved_files = create_results_report(
    results,
    contours,
    contour_names=names,
    output_dir='./artwarp_report',
    dpi=300,
)
# returns dict mapping figure names to file paths
print(f"Created {len(saved_files)} figures")

# with temporal resolution data (enables the tempres histogram)
contours, names, tempres_list = load_contours('./data', return_tempres=True)
saved_files = create_results_report(
    results,
    contours,
    contour_names=names,
    output_dir='./artwarp_report',
    dpi=300,
    tempres_list=tempres_list,
)
```

Generates a complete set of visualizations saved to a directory. If `include_additional=True` (default), also creates an **additional/** subdirectory with extra algorithm and diagnostic figures.

**Generated Figures (main directory):**
- training_summary.png
- reference_contours.png
- category_distribution.png
- convergence_history.png
- match_distribution.png
- discovery_curve.png
- category_X_contours.png (for each category)

**Generated Figures (additional/ subdirectory, when include_additional=True):**
- art_schematic.png — ART decision flow diagram
- warp_constraint.png — Itakura band illustration
- per_category_match_quality.png — Match score distribution by category
- category_similarity_matrix.png — Pairwise DTW similarity heatmap
- category_embedding.png — MDS acoustic-space embedding of category prototypes
- contour_length_distribution.png — Length histogram; tempres histogram if `tempres_list` is supplied
- dtw_alignment.png — DTW path for the first contour/prototype pair whose length ratio is within `warp_factor_level`
- resampling_before_after.png — First contour before/after resampling (demo)
- paper_figure.png — Single multi-panel publication figure
- category_dendrogram.png — Hierarchical clustering of prototypes (requires scipy)

**Parameters:**
- `include_additional` (default `True`) — generate the `additional/` figures
- `warp_factor_level` (default `3`) — DTW warp factor used for similarity/alignment plots
- `tempres_list` (default `None`) — per-contour temporal resolution (s/sample); enables the tempres histogram in `contour_length_distribution.png`. The CLI (`artwarp-py plot`) passes this automatically.

**Returns:** Dictionary mapping figure names to file paths (keys like `"additional/art_schematic"`, etc.)

### 9. DTW Alignment

```python
from artwarp.visualization import plot_dtw_alignment

fig = plot_dtw_alignment(
    reference_contour,
    comparison_contour,
    warp_factor_level=3,
    figsize=(8, 10),
    save_path='dtw_alignment.png'
)
```

Two-panel figure (stacked vertically): the top panel overlays both contours on the same time axis; the bottom panel shows the optimal DTW warping path on the alignment grid, with the allowed Itakura band shaded and the no-warp diagonal drawn for reference.

> **Note:** raises `ValueError` if the length ratio between `u1` and `u2` exceeds `warp_factor_level` (no valid alignment path exists). When using this via `create_results_report`, the function automatically searches for the first compatible contour/prototype pair.

### 10. ART Decision Schematic

```python
from artwarp.visualization import plot_art_schematic

fig = plot_art_schematic(figsize=(13, 5), save_path='art_schematic.png')
```

Static decision-flow diagram: input contour → DTW to all prototypes → sort by activation → vigilance test → commit (update weights) or create new category. Each step is labelled and color-coded.

### 11. Warp Constraint (Itakura Band)

```python
from artwarp.visualization import plot_warp_constraint

fig = plot_warp_constraint(warp_factor_level=3, m=20, n=25, save_path='warp_constraint.png')
```

Illustrates the allowed (i, j) band for DTW given `warp_factor_level`. Explains temporal flexibility of the algorithm.

### 12. Per-Category Match Quality

```python
from artwarp.visualization import plot_per_category_match_quality

fig = plot_per_category_match_quality(results, figsize=(10, 6), save_path='per_cat_match.png')
```

Violin/box of match scores by category. Assesses how well samples in each category match their prototype.

### 13. Category Similarity Matrix

```python
from artwarp.visualization import plot_category_similarity_matrix

fig = plot_category_similarity_matrix(
    results.weight_matrix,
    warp_factor_level=3,
    figsize=(8, 7),
    save_path='category_similarity.png'
)
```

Heatmap of pairwise DTW similarity between category prototypes. Identifies which categories are acoustically close.

### 14. Category Embedding (MDS)

```python
from artwarp.visualization import plot_category_embedding

fig = plot_category_embedding(
    results.weight_matrix,
    warp_factor_level=3,
    figsize=(8, 6),
    save_path='category_embedding.png'
)
```

2D acoustic-space embedding of all category prototypes. Pairwise DTW distances between prototypes are reduced to two dimensions using classical Multidimensional Scaling (MDS). The x-axis (Dimension 1) captures the largest source of variation in the distance matrix; the y-axis (Dimension 2) captures the next largest. Categories that appear close together are acoustically similar (small DTW distance); widely separated points represent perceptually distinct call types. Category indices are annotated on each point.

### 15. Resampling Before/After

```python
from artwarp.visualization import plot_resampling_before_after

fig = plot_resampling_before_after(
    contour,
    tempres=0.01,
    sample_interval_sec=0.02,
    title='Example contour',
    save_path='resampling.png'
)
```

One contour before and after resampling (time in seconds). Demonstrates temporal normalization.

### 16. Contour Length and Tempres Distribution

```python
from artwarp.visualization import plot_contour_length_distribution

fig = plot_contour_length_distribution(
    contours,
    tempres_list=tempres_list,  # optional
    figsize=(12, 5),
    save_path='length_tempres_dist.png'
)
```

Histograms of contour lengths and (if provided) temporal resolution. For mixed-resolution datasets and reporting.

### 17. Vigilance Sweep

```python
from artwarp.visualization import plot_vigilance_sweep

sweep = [(v, ARTwarp(vigilance=v, random_seed=42).fit(contours)) for v in [75, 85, 95]]
fig = plot_vigilance_sweep(sweep, figsize=(10, 6), save_path='vigilance_sweep.png')
```

Number of categories (and mean match) vs vigilance. For parameter choice and methods figures.

### 18. Run Stability

```python
from artwarp.visualization import plot_run_stability

num_cats = [len(ARTwarp(vigilance=85, random_seed=s).fit(contours).get_category_sizes()) for s in range(10)]
fig = plot_run_stability(num_cats, figsize=(8, 5), save_path='run_stability.png')
```

Distribution of number of categories across multiple runs (e.g. different seeds). For reproducibility reporting.

### 19. Paper-Ready Multi-Panel Figure

```python
from artwarp.visualization import create_paper_figure

fig = create_paper_figure(
    results,
    contours,
    contour_names=names,
    title='Species X',
    figsize=(16, 12),
    save_path='paper_figure.png'
)
```

Single publication-ready figure: reference contours, discovery curve, category distribution.

### 20. Category Dendrogram

```python
from artwarp.visualization import plot_category_dendrogram

fig = plot_category_dendrogram(
    results.weight_matrix,
    warp_factor_level=3,
    figsize=(10, 6),
    save_path='dendrogram.png'
)
```

Hierarchical clustering of category prototypes (DTW distance). Requires `scipy`. Shows which categories are similar.

### 21. Confusion Matrix (Ground Truth)

```python
from artwarp.visualization import plot_confusion_matrix

fig = plot_confusion_matrix(
    ground_truth_labels,        # list of labels per sample
    results.categories,
    class_names=None,           # optional list of readable class names
    figsize=(8, 7),
    save_path='confusion.png'
)
```

Rows = ground truth, columns = ART category. Use when labels are available for evaluation.

**Parameters:**
- `ground_truth_labels`: List of per-sample ground-truth labels (str or int)
- `categories`: Category assignments from `TrainingResults`
- `class_names`: Optional list of human-readable class names (replaces numeric labels on axes)
- `figsize`: Figure size in inches
- `save_path`: Optional path to save figure
- `dpi`: Resolution for saved figure (default: 300)

### 22. Label vs Category Distribution

```python
from artwarp.visualization import plot_label_vs_category

fig = plot_label_vs_category(
    ground_truth_labels,
    results.categories,
    figsize=(10, 6),
    save_path='label_vs_category.png'
)
```

Stacked bar: per ART category, proportion of each ground-truth label. Complements the confusion matrix.

## Customization

This is for research paper customization!

Feel free to modify the @plotting.py functions however you wish.

### Figure Size

All functions accept `figsize` parameter:

```python
fig = plot_category_distribution(results, figsize=(12, 8))  # larger
fig = plot_category_distribution(results, figsize=(6, 4))   # smaller
```

### Resolution

Control output resolution with `dpi` parameter:

```python
# for screen display
fig = plot_training_summary(results, dpi=100)

# for print/publication
fig = plot_training_summary(results, dpi=300)

# for high-quality posters
fig = plot_training_summary(results, dpi=600)
```

### Saving Figures

Two ways to save figures:

**Method 1: Using save_path parameter**
```python
fig = plot_category_distribution(
    results,
    save_path='output.png',
    dpi=300
)
```

**Method 2: Manual save**
```python
import matplotlib.pyplot as plt

fig = plot_category_distribution(results)
plt.savefig('output.png', dpi=300, bbox_inches='tight')
plt.close(fig)
```

### File Formats

Matplotlib supports multiple output formats:

```python
# PNG (raster -> good for screen)
plt.savefig('figure.png', dpi=300)

# PDF (vector -> good for print)
plt.savefig('figure.pdf')

# SVG (vector -> good for editing)
plt.savefig('figure.svg')

# EPS (vector -> publication standard)
plt.savefig('figure.eps')
```

## Best Practices (:D)

### 1. Publication-Ready Figures

```python
fig = plot_training_summary(
    results,
    names,
    figsize=(16, 10),
    save_path='figure.pdf',  # vector format
    dpi=300
)
```

### 2. Batch Processing

```python
# generate visualizations for multiple runs
for vigilance in [75.0, 85.0, 95.0]:
    network = ARTwarp(vigilance=vigilance)
    results = network.fit(contours)
    
    plot_training_summary(
        results,
        save_path=f'summary_v{vigilance}.png',
        dpi=300
    )
```

### 3. Interactive Exploration

```python
import matplotlib.pyplot as plt

# create figure
fig = plot_training_summary(results, contour_names=names)

# show interactively
plt.show()  # opens interactive window

# zoom, pan, and save from GUI
```

### 4. Programmatic Analysis

```python
# generate report and analyze
files = create_results_report(results, contours, contour_names=names)

# process generated files
for name, path in files.items():
    print(f"Generated {name}: {path}")
    
```

## Examples

### Example 1: Quick Visualization

```python
from artwarp import ARTwarp, load_contours
from artwarp.visualization import plot_training_summary
import matplotlib.pyplot as plt

# load and train
contours, names = load_contours('./data')
network = ARTwarp(vigilance=85.0)
results = network.fit(contours)

# visualize
plot_training_summary(results, contour_names=names)
plt.show()
```

### Example 2: Publication Figure

```python
# high-quality figure for publication
fig = plot_training_summary(
    results,
    contour_names=names,
    figsize=(16, 10)
)

# save in multiple formats
fig.savefig('figure.pdf', dpi=300, bbox_inches='tight')
fig.savefig('figure.png', dpi=600, bbox_inches='tight')
fig.savefig('figure.eps', dpi=300, bbox_inches='tight')
```

### Example 3: Custom Analysis

```python
import matplotlib.pyplot as plt
from artwarp.visualization import (
    plot_reference_contours,
    plot_category_distribution
)

# create custom figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Reference contours
plot_reference_contours(results.weight_matrix)

# Plot 2: Category distribution
plot_category_distribution(results)

plt.tight_layout()
plt.savefig('custom_analysis.png', dpi=300)
```

### Example 4: Automated Report

```python
# generate complete report
files = create_results_report(
    results,
    contours,
    contour_names=names,
    output_dir='./report_2024',
    dpi=300
)

print(f"Generated report with {len(files)} figures:")
for name, path in files.items():
    print(f"  {name}: {path}")
```

## Integration with CLI

### plot command

Generate the same visualization report from the CLI (after training):

```bash
# train and save results
artwarp-py train --input-dir ./data --output results.pkl --export-refs --vigilance 85

# generate visualization report
artwarp-py plot --results results.pkl --input-dir ./data --output-dir ./report
```

**Options:**
- `-r, --results`: Path to the .pkl file from `train`
- `-i, --input-dir`: Directory of contour files (same as used for training)
- `-o, --output-dir`: Where to save figures (default: `./report`)
- `--format`, `--freq-column`: Same as `train` (must match how contours were loaded)
- `--dpi`: Figure resolution (default: 300)

For custom layouts or single plots, use the Python API (`create_results_report`, `plot_training_summary`, etc.) as in the examples above.

## Troubleshooting

### Matplotlib Not Found

```bash
pip install matplotlib
```

### Non-Interactive Backend

If running on server without display:

```python
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend (!!!)

from artwarp.visualization import plot_training_summary

# now works without display
fig = plot_training_summary(results, save_path='output.png')
```

### Memory Issues

For large reports:

```python
import matplotlib.pyplot as plt

# close figures after saving
fig = plot_category_distribution(results, save_path='dist.png')
plt.close(fig)  # free memory
```

### Font Issues

If fonts don't render properly:

```python
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
# or: 'Arial', 'Helvetica', 'Times New Roman'
```

## Advanced Usage

### Custom Color Schemes

```python
import matplotlib.pyplot as plt

# set custom color scheme before plotting
plt.style.use('seaborn-v0_8-darkgrid')  # Or other styles

fig = plot_training_summary(results, contour_names=names)
```

### Combining with Other Tools

```python
# use with pandas for analysis
import pandas as pd

# create DataFrame from results
df = pd.DataFrame({
    'category': results.categories,
    'match': results.matches
})

# analyze
print(df.groupby('category')['match'].describe())

# then visualize
plot_training_summary(results)
```

### Animation

```python
from matplotlib.animation import FuncAnimation

# animate convergence history
fig, ax = plt.subplots()

def update(frame):
    # update plot for each iteration
    pass

anim = FuncAnimation(fig, update, frames=results.num_iterations)
anim.save('convergence.gif', writer='pillow')
```

## Performance Tips

1. **Use DPI wisely**: Lower DPI (100-150) for screen, higher (300+) for print
2. **Close figures**: Use `plt.close(fig)` to free memory
3. **Batch mode**: Use `matplotlib.use('Agg')` for non-interactive
4. **Limit contours**: Use `max_contours` parameter for readability

## Support

For issues or questions about visualization:
- Check example: `examples/visualization_example.py`
- Review tests: `tests/unit/test_visualization.py`
- See documentation: This file (VISUALIZATION.md)

## Author Note

I was purposefully very rigorous with the plotting. Please take advantage of this .md file to see common uses for the API functionality.

Feel free to recycle some of these code cells for your own personal script use!

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)