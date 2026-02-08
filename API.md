# ARTwarp-py API Reference

## Core API

### ARTwarp Class

The main class for training and prediction.

```python
from artwarp import ARTwarp

network = ARTwarp(
    vigilance=85.0,
    learning_rate=0.1,
    bias=0.0,
    max_categories=50,
    max_iterations=50,
    warp_factor_level=3,
    random_seed=None,
    verbose=True
)
```

#### Parameters

- **vigilance** (float, default=85.0): Match threshold for category assignment, range [1, 99]
  - Higher values create more categories -> stricter matching
  - Lower values create fewer categories -> more permissive
  
- **learning_rate** (float, default=0.1): Weight update rate, range (0, 1]
  - Higher values mean faster adaptation to new inputs
  - Lower values preserve existing category prototypes better
  
- **bias** (float, default=0.0): Activation bias, range [0, 1]
  - Higher bias makes categories more selective
  - Usually kept at 0.0 for standard ARTwarp
  
- **max_categories** (int, default=100): Maximum number of categories to create
  - Prevents unlimited category growth
  - Inputs exceeding this limit may be left uncategorized
  - CLI `train` uses default 50 (`--max-categories`)
  
- **max_iterations** (int, default=50): Maximum number of training iterations
  - Training stops early if convergence is reached
  - More iterations allow better category refinement
  
- **warp_factor_level** (int, default=3): Maximum DTW warping factor
  - Controls time compression/expansion flexibility
  - Higher values allow more temporal distortion
  - Original ARTwarp uses 3
  
- **random_seed** (int, optional): Random seed for reproducibility
  - Set to integer for deterministic results
  - Leave None for different results each run
  
- **verbose** (bool, default=True): Whether to print progress information

#### Methods

##### fit()

Train the network on frequency contours.

```python
results = network.fit(contours, contour_names=None)
```

**Parameters**:
- `contours` (List[NDArray]): List of frequency contour arrays
- `contour_names` (List[str], optional): Names for each contour

**Returns**:
- `TrainingResults`: Object containing training results

**Example**:
```python
contours = [
    np.array([100.0, 200.0, 300.0]),
    np.array([150.0, 250.0, 350.0]),
]
results = network.fit(contours)
print(f"Created {results.num_categories} categories")
```

##### predict()

Predict categories for new contours using trained network.

```python
categories, matches = network.predict(contours)
```

**Parameters**:
- `contours` (List[NDArray]): List of frequency contour arrays

**Returns**:
- `categories` (NDArray): Category assignments (NaN for uncategorized)
- `matches` (NDArray): Match values for each contour

**Example**:
```python
test_contours = [np.array([105.0, 205.0, 305.0])]
categories, matches = network.predict(test_contours)
print(f"Assigned to category {categories[0]} with {matches[0]:.1f}% match")
```

### TrainingResults Class

Container for training results.

```python
@dataclass
class TrainingResults:
    categories: NDArray[np.float64]
    matches: NDArray[np.float64]
    weight_matrix: NDArray[np.float64]
    num_categories: int
    num_iterations: int
    converged: bool
    iteration_history: List[tuple]
    training_time: float
```

#### Attributes

- **categories**: Category assignment for each input (shape: num_samples)
- **matches**: Match values for each input (shape: num_samples)
- **weight_matrix**: Final weight matrix (shape: max_features × num_categories)
- **num_categories**: Number of categories created
- **num_iterations**: Number of iterations performed
- **converged**: Whether training converged (no reclassifications)
- **iteration_history**: List of (iteration, num_reclassifications) tuples
- **training_time**: Total training time in seconds

#### Methods

##### get_category_sizes()

Get the number of samples in each category.

```python
sizes = results.get_category_sizes()
# returns: {0: 10, 1: 15, 2: 8}  # category_id: count
```

##### get_uncategorized_count()

Get number of samples that could not be categorized.

```python
uncategorized = results.get_uncategorized_count()
```

## Data Loading API

### load_mat_categorisation()

Load a MATLAB ARTwarp .mat file (NET and optionally DATA). Matches "Load Categorisation" in MATLAB.

```python
from artwarp.io.loaders import load_mat_categorisation

data = load_mat_categorisation("ARTwarp85FINAL.mat")
# data['weight_matrix'], data['num_categories'], data['vigilance'], ...
# if DATA was saved: data['contours'], data['categories'] (0-based), data['contour_names']
```

### resample_contours()

Resample contours to a uniform temporal resolution (MATLAB "Resample Contours" option).

```python
from artwarp.utils import resample_contours

# tempres: seconds per point per contour (float or list); sample_interval_sec: target interval
resampled = resample_contours(contours, tempres=0.01, sample_interval_sec=0.02)
# or with per-contour tempres (e.g. from load_contours(..., return_tempres=True)):
resampled = resample_contours(contours, tempres_list, sample_interval_sec=0.02)
```

### load_contours()

Load contour files from a directory.

```python
from artwarp import load_contours

contours, names = load_contours(
    directory='./data',
    file_format='auto',
    frequency_column=0,
    pattern='*'
)
# w/ temporal resolution (for resampling):
contours, names, tempres_list = load_contours('./data', return_tempres=True)
```

**Parameters**:
- `directory` (str): Path to directory containing contour files
- `file_format` (str): File format - 'ctr', 'csv', 'txt', or 'auto' (default)
- `frequency_column` (int): Column index for frequency data (CSV/TXT files)
- `pattern` (str): Glob pattern for file matching
- `return_tempres` (bool, default=False): If True, also return a list of temporal resolution (seconds per point) per contour; entries may be None if unknown (e.g. some .ctr files)

**Returns**:
- If `return_tempres` is False: `(contours, names)`.
- If `return_tempres` is True: `(contours, names, tempres_list)` where `tempres_list[i]` is seconds per point for contour i, or None.

**Example**:
```python
# load all CSV files from directory
contours, names = load_contours('./whistles', file_format='csv')

# load with tempres for resampling (use default tempres for any None)
contours, names, tempres = load_contours('./data', return_tempres=True)
tempres = [t or 0.01 for t in tempres]
resampled = resample_contours(contours, tempres, 0.02)
```

## Export API

### export_results()

Export training results to pickle file.

```python
from artwarp.io.exporters import export_results

export_results(
    results,
    filepath='results.pkl',
    include_names=True,
    contour_names=names
)
```

### export_reference_contours()

Export reference contours (category prototypes) to CSV files.

```python
from artwarp.io.exporters import export_reference_contours

export_reference_contours(
    weight_matrix=results.weight_matrix,
    output_dir='./references',
    prefix='refContour',        # default; matches MATLAB SaveRefContours.m
    one_based_filenames=True    # default; refContour_1.csv, refContour_2.csv, ...
)
```

### export_category_assignments()

Export category assignments to CSV file.

```python
from artwarp.io.exporters import export_category_assignments

export_category_assignments(
    categories=results.categories,
    matches=results.matches,
    contour_names=names,
    filepath='assignments.csv',
    one_based_categories=False   # set True for MATLAB-style 1-based category indices
)
```

## Low-Level API

### Dynamic Time Warping

```python
from artwarp.core.dtw import dynamic_time_warp

similarity, warp_function = dynamic_time_warp(
    u1=reference_contour,
    u2=comparison_contour,
    warp_factor_level=3
)
```

### ART Components

```python
from artwarp.core.art import (
    activate_categories,
    calculate_match,
    check_resonance
)

# activate categories (warp_factor_level -> must match network training)
activations, warp_funcs = activate_categories(
    input_contour,
    weight_matrix,
    bias=0.0,
    warp_factor_level=3
)

# calculate match
match = calculate_match(warped_input, weight_vector)

# check resonance
resonance = check_resonance(match_value=85.5, vigilance=85.0)
```

## Command-Line Interface

Ensure the `artwarp-py` command is on your PATH (activate your virtual environment first, e.g. `source venv/bin/activate` or `source venv/bin/activate.fish` in Fish). Alternatively run: `python -m artwarp.cli.main <command> ...`.

### train

Train a network on contour files. Parameters that affect the **ARTwarp network** (vigilance, learning_rate, bias, max_categories, max_iterations, warp_factor_level, random_seed, verbose) are passed into the `ARTwarp` constructor. The options **resample**, **sample-interval**, and **tempres** are **preprocessing** only: they resample contours before training and are not network parameters.

If you wish to preprocess in a separate Python script, call the functions listed in the example code cells above.

**Full option list**:

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input-dir` | (required) | Directory containing contour files |
| `-o`, `--output` | (required) | Output file for trained model (.pkl) |
| `--format` | auto | Input format: auto, ctr, csv, txt |
| `--freq-column` | 0 | Frequency column index (CSV/TXT) |
| `--vigilance` | 85.0 | Match threshold [1, 99] → `ARTwarp(vigilance=...)` |
| `--learning-rate` | 0.1 | Weight update rate → `ARTwarp(learning_rate=...)` |
| `--bias` | 0.0 | Activation bias → `ARTwarp(bias=...)` |
| `--max-categories` | 50 | Max categories → `ARTwarp(max_categories=...)` |
| `--max-iterations` | 50 | Max iterations → `ARTwarp(max_iterations=...)` |
| `--warp-factor` | 3 | DTW warping factor → `ARTwarp(warp_factor_level=...)` |
| `--seed` | None | Random seed → `ARTwarp(random_seed=...)` |
| `--export-refs` | false | Export reference contours to CSV |
| `--export-categories` | false | Export category assignments to CSV |
| `-q`, `--quiet` | false | Suppress progress → `ARTwarp(verbose=False)` |
| `--resample` | false | **Preprocessing**: resample contours to uniform temporal resolution before training |
| `--sample-interval` | 0.02 | **Preprocessing**: target sampling interval (seconds) when `--resample` |
| `--tempres` | 0.01 | **Preprocessing**: default temporal resolution (sec/point) for contours without it, when `--resample` |

**Example (basic)**:

```bash
artwarp-py train \
    --input-dir ./contours \
    --output results.pkl \
    --format csv \
    --freq-column 1 \
    --vigilance 85 \
    --learning-rate 0.1 \
    --max-categories 50 \
    --max-iterations 50 \
    --export-refs \
    --export-categories
```

**Example (with resampling, MATLAB-aligned)**:

```bash
artwarp-py train -i ./contours -o results.pkl --resample --sample-interval 0.02 --tempres 0.01
```

When `--resample` is used, contours are loaded with `return_tempres=True`, then `resample_contours(contours, tempres_list, sample_interval)` is called; contours without tempres use `--tempres`. The resampled contours are then passed to `ARTwarp(...).fit(contours, names)`.

### predict

Predict categories for new contours.

```bash
artwarp-py predict \
    --model results.pkl \
    --input-dir ./new_contours \
    --output predictions.csv
```

### export

Export results to various formats.

```bash
artwarp-py export \
    --results results.pkl \
    --output-dir ./output \
    --export-type all
```

### plot

Generate a visualization report (training summary, reference contours, convergence, etc.) from saved results. 

Requires the same contour directory and format as used for training.

```bash
artwarp-py plot \
    --results results.pkl \
    --input-dir ./contours \
    --output-dir ./report \
    --format csv \
    --freq-column 0 \
    --dpi 300
```

## Type Hints

All functions include complete type hints:

```python
def dynamic_time_warp(
    u1: NDArray[np.float64],
    u2: NDArray[np.float64],
    warp_factor_level: int = 3
) -> Tuple[float, NDArray[np.int32]]:
    """..."""
```

## Error Handling

The API raises informative exceptions:

```python
# invalid parameters
ValueError: Vigilance must be in range [1, 99], got 100.0

# empty data
ValueError: No contours provided

# prediction before training
RuntimeError: Network must be trained before prediction

# file not found
FileNotFoundError: Directory not found: ./data
```

## Best Practices

### Memory Management

For large datasets (1000+ contours):
```python
# process in batches
batch_size = 100
for i in range(0, len(contours), batch_size):
    batch = contours[i:i+batch_size]
    # process batch...
```

### Reproducibility

Always use random seed for reproducible results:
```python
network = ARTwarp(random_seed=42)
```

### Parameter Tuning

Start with default parameters, then adjust:
1. **Vigilance**: Most important parameter
   - Too high → many categories, potential overfitting
   - Too low → few categories, potential undergeneralization
   
2. **Learning Rate**: 
   - 0.1 is usually good
   - Higher for rapidly changing data
   - Lower for stable categories

3. **Max Iterations**:
   - 50 is usually sufficient
   - Increase if not converging

### Performance Tips

1. Use Numba for large datasets:
   ```bash
   pip install numba
   ```

2. Pre-filter obviously invalid contours

3. Normalize contour lengths if possible [!!!]

4. Use batch processing for very large datasets

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)