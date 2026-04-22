# ARTwarp-py Quick Reference

三三三三 Zooooooom

(for the lazy people, like me :D )

## Installation

For more rigorous environment setup, view INSTALLATION.md!

```bash
pip install numpy scipy pandas
# optional: pip install numba matplotlib
```

## Python API - 3 Lines to Results

```python
from artwarp import ARTwarp, load_contours
contours, names = load_contours('./data')
results = ARTwarp(vigilance=85.0).fit(contours)
```

## Command Line - One Command

Activate your virtual environment first so `artwarp-py` is on your PATH (e.g. `source venv/bin/activate` or, in Fish, `source venv/bin/activate.fish`). You can also use the **interactive launcher** `./run.sh` (no arguments) for a menu that prompts for all options (Train / Plot / Predict / Export). When you run the CLI (or `./run.sh`), a Numba status line is shown; if Numba is not installed, you’ll see a warning and, in an interactive terminal, an option to install it automatically.

Then, for direct commands:

```bash
artwarp-py train --input-dir ./contours --output results.pkl --vigilance 85

# resampling is on by default (MATLAB ARTwarp_cli_mode); explicit interval / skip:
artwarp-py train --input-dir ./contours --output results.pkl --sample-interval 0.01
artwarp-py train --input-dir ./contours --output results.pkl --no-resample

# optional MATLAB training flags (same names as GUI: recatSingleCats, compareWarped)
artwarp-py train --input-dir ./contours --output results.pkl --recat-single-categories --compare-warped

# optional extensions (non-MATLAB stable; search reorder + orphan column purge)
artwarp-py train --input-dir ./contours --output results.pkl \
  --deprioritize-lone-category-search --purge-empty-categories

# generate visualization report
artwarp-py plot --results results.pkl --input-dir ./contours --output-dir ./report
```

Without activating, use the venv’s Python: `./venv/bin/python -m artwarp.cli.main train ...` (or `... plot ...`).

## Key Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| vigilance | 1-99 | 85.0 | Higher = more categories |
| learning_rate | 0-1 | 0.1 | Higher = faster adaptation |
| bias | 0-1 | 0.0 | Higher = more selective |
| max_categories | 1+ | 50 | Limit category creation |
| max_iterations | 1+ | 50 | Max training iterations |
| warp_factor_level | 2+ | 3 | Max time warping |
| recat_single_categories | on/off | off | MATLAB `recatSingleCats`: lone-contour recat after each iteration |
| compare_warped | on/off | off | MATLAB `compareWarped`: warped weight branch + average weights each iteration |
| deprioritize_lone_category_search | on/off | off | Try other categories before current when sample is alone in its category |
| purge_empty_categories | on/off | off | Each iteration, delete weight columns with zero assigned contours |

**CLI-only (train)**: resample **on** by default (`--no-resample` to match MATLAB `resample=0`); `--sample-interval SEC` (default **0.01**), `--tempres SEC` (default **0.01**). **`--recat-single-categories`** and **`--compare-warped`** mirror MATLAB `ARTwarp_Run_Categorisation.m` optional steps (default off). **`--deprioritize-lone-category-search`** and **`--purge-empty-categories`** are optional behaviours (default off). Interactive `./run.sh` prompts for resampling (default yes), bias (default **1e-6**), MATLAB parity flags, then extensions.

## Common Use Cases

### Basic Categorization
```python
network = ARTwarp(vigilance=85.0)
results = network.fit(contours)
print(f"Created {results.num_categories} categories")

# visualize
from artwarp.visualization import plot_training_summary
plot_training_summary(results)
```

### Prediction on New Data
```python
categories, matches = network.predict(new_contours)
```

### Export and Visualize Results
```python
from artwarp.io.exporters import (
    export_results,
    export_reference_contours,
    export_reference_contour_metadata,
)
from artwarp.visualization import create_results_report

# export data
export_results(results, 'results.pkl')
export_reference_contours(results.weight_matrix, './references')
# provenance: UUID per prototype + parent contour names (auto-written by --export-refs)
export_reference_contour_metadata(results.category_parent_names, './references/metadata.csv')

# generate visual report
create_results_report(results, contours, names, output_dir='./report')
```

### Load Different Formats (done automatically in CLI)
```python
# CSV files
contours, names = load_contours('./data', file_format='csv', frequency_column=0)

# MATLAB .ctr files
contours, names = load_contours('./data', file_format='ctr')

# with temporal resolution (for resampling)
contours, names, tempres = load_contours('./data', return_tempres=True)
```

### Resample Before Training (CLI or API)
```bash
artwarp-py train -i ./contours -o results.pkl
# or custom interval (CLI default sample interval is 0.01 s)
artwarp-py train -i ./contours -o results.pkl --sample-interval 0.02
```
```python
from artwarp.utils import resample_contours
contours, names, tempres = load_contours('./data', return_tempres=True)
tempres = [t or 0.01 for t in tempres]
contours = resample_contours(contours, tempres, 0.01)
results = network.fit(contours, contour_names=names)
```

## Results Structure

```python
results.categories             # array of category assignments
results.matches                # array of match values
results.weight_matrix          # category prototypes
results.num_categories         # num of categories
results.converged              # convergence status
results.training_time          # time in seconds
results.category_parent_names  # {cat_idx: [contour_name, ...]} provenance map
```

## Troubleshooting

### Too Many Categories?
Increase vigilance: `ARTwarp(vigilance=90.0)`

### Too Few Categories?
Decrease vigilance: `ARTwarp(vigilance=75.0)`

### Not Converging?
Increase max_iterations: `ARTwarp(max_iterations=100)`

### Slow Performance?
Install Numba: `pip install numba`

### Memory Issues?
Process in batches or reduce max_categories

## File Formats

### Input Contours
- **.ctr**: MATLAB format (requires scipy)
- **.csv**: CSV with frequency column
- **.txt**: Tab-delimited text

All formats should contain frequency values over time.

### Output Files
- **.pkl**: Python pickle (complete results, includes provenance)
- **.csv**: Category assignments, reference contours, or provenance metadata (`metadata.csv`)

## Testing

```bash
# run all tests (use venv Python so dependencies are found)
python -m pytest tests/ -v

# run specific test file
python -m pytest tests/unit/test_dtw.py -v

# run without slow tests
python -m pytest tests/ -m "not slow"
```

## Examples

```bash
# basic example
python examples/simple_example.py

# basic visualization
python examples/visualization_example.py
```

## Documentation

- **docs/README.md**:              Overview and features
- **docs/user/INSTALLATION.md**:   Detailed installation guide
- **docs/user/API.md**:            Complete API reference
- **docs/user/VISUALIZATION.md**:  Visualization guide
- **docs/user/ARCHITECTURE.md**:   Internal design details
- **docs/dev/PROJECT_SUMMARY.md**: Project summary (in `docs/dev/` for developers)

## Performance Tips

1. **Use random seed for reproducibility**: `ARTwarp(random_seed=42)`
    - This is particuarly useful to ensure for random initial reference contour selection
    - Same logic as in original MATLAB code
2. **Start with default parameters**: Adjust vigilance first
3. **Pre-filter invalid contours**: Remove outliers before training
4. **Batch large datasets**: Process 100-500 contours at a time

## Common Patterns

### Reproducible Training
```python
network = ARTwarp(vigilance=85.0, random_seed=42)
results = network.fit(contours)
```

### Parameter Sweep
```python
for vigilance in [75.0, 80.0, 85.0, 90.0, 95.0]:
    network = ARTwarp(vigilance=vigilance, random_seed=42)
    results = network.fit(contours)
    print(f"V={vigilance}: {results.num_categories} categories")
```

### Cross-Validation
```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(contours, test_size=0.2)
network = ARTwarp(vigilance=85.0).fit(train_data)
categories, matches = network.predict(test_data)
```

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Unknown command: artwarp-py` (Fish) | CLI not on PATH | Activate venv: `source venv/bin/activate.fish` (or `activate` on bash) |
| `Vigilance must be in range [1, 99]` | Invalid vigilance | Use 1-99 |
| `No contours provided` | Empty list | Load data first |
| `Network must be trained` | Predict before fit | Train first |
| `Directory not found` | Bad path | Check path |

## Support

- Issues: GitHub issue tracker
- Documentation: README and API.md in this directory (`docs/user/`)
- Examples: examples/ directory
- Contact: pgg6@st-andrews.ac.uk      (university email)
           pgrondagarrigues@gmail.com (if email above expired post-graduation)

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)