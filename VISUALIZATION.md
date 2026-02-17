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
- **Reports**: Automated report generation

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

### 7. Report

```python
from artwarp.visualization import create_results_report

saved_files = create_results_report(
    results,
    contours,
    contour_names=names,
    output_dir='./artwarp_report',
    dpi=300
)
# returns dict mapping figure names to file paths

print(f"Created {len(saved_files)} figures")
```

Generates a complete set of visualizations saved to a directory.

**Generated Figures:**
- training_summary.png
- reference_contours.png
- category_distribution.png
- convergence_history.png
- match_distribution.png
- category_X_contours.png (for each category)

**Returns:** Dictionary mapping figure names to file paths

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