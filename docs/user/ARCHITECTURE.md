# ARTwarp-py Architecture

This document describes the internal architecture of ARTwarp-py.

Although it's not necessary for end-users to fully understand this document, it might be a useful entry-point for potential developers (in which case, view docs/).

## Overview

ARTwarp-py is organized into modular components that separate concerns and enable
easy testing, maintenance, and extension.

```
artwarp-py/
├── src/artwarp/          # main package
│   ├── core/             # core algorithm implementations
│   ├── io/               # input/output operations
│   ├── utils/            # utility functions
│   └── cli/              # command-line interface
├── tests/                # test suite
├── examples/             # usage examples
├── img/                  # README banner
└── docs/                 # documentation
```

## Core Modules

### 1. Dynamic Time Warping (`core/dtw.py`)

**Purpose**: Measure similarity between frequency contours with temporal flexibility.

**Key Functions**:
- `compute_similarity_matrix()`: Vectorized point-wise similarity calculation
- `dynamic_time_warp()`: Main DTW algorithm with Itakura parallelogram constraints; calls the Numba core when available
- `unwarp()`: Invert warping functions

**Optimization Techniques**:
- NumPy broadcasting for similarity matrix (eliminates nested loops)
- **Numba JIT-compiled core** (`_dtw_core_numba`) for the DP and backtrace—the main reason this implementation runs **faster than MATLAB** (see below)
- Early termination for length ratio violations
- Efficient path backtracing

**Mathematical Equivalence**:
- Implements the same algorithm as original MATLAB repo's `warp.m`
- Maintains identical constraint logic
- Uses same similarity formula: `min(f1,f2)/max(f1,f2) * 100`

#### DTW core: Numba JIT kernel (`_dtw_core_numba`)

The hot path that makes ARTwarp-py run **remarkably fast** (faster than MATLAB) is the Numba-JIT-compiled DTW core in `core/dtw.py`: `_dtw_core_numba`. It is the same logic as the Python fallback but compiled to native code.

**Inputs**
- **M**: an *m*×*n* similarity matrix (e.g. from `compute_similarity_matrix(u1, u2)`), where `M[i,j]` is the point-wise similarity between contour 1 at index *i* and contour 2 at index *j*.
- **m, n**: lengths of the two contours.
- **wfl**: warp factor level (e.g. 3), the maximum allowed step in the warping band.

**Outputs**
- **Normalized similarity**: total similarity along the best path divided by *m* (average similarity, 0–100 scale).
- **Warp function**: `warp_func[i] = j` means “contour 1 index *i* is aligned to contour 2 index *j*”.

So this function is *DTW with an Itakura-style band, implemented as a fast Numba kernel.*

**Algorithm: DTW with Itakura parallelogram constraints**

DTW aligns two sequences (here, frequency contours) by maximizing cumulative similarity along a path. The path is **globally constrained** to a band (Itakura parallelogram in the (*i*,*j*) plane):

- **Vertical move**: same *j*, next *i* (step 0).
- **Diagonal move**: advance *i* and *j* (step in {1, …, *wfl*}).
- **wfl** limits: at most *wfl* consecutive vertical steps, and at most *wfl* as the jump in *j* in one step.

**Dynamic programming**
- **N[i,j]**: best cumulative similarity to reach cell (*i*, *j*) along an allowed path. Recurrence: `N[i,j] = M[i,j] + max` over predecessors `(i-1, j-step)` with `step ∈ {0,1,…,wfl}` within the band.
- **p[i,j]**: chosen step stored as `-step`; backtrace from (*i*, *j*) goes to (*i*-1, *j* + *p*[*i*,*j*]).
- **k[i,j]**: run length of consecutive vertical (step 0) moves; used to enforce “at most *wfl* vertical steps in a row” (when *k* ≥ *wfl*, next step must move in *j*).

**Stages**
- **Early stage** (small *i*): band shape near the origin is handled with a separate loop (fewer *j* values).
- **General stage**: for each row *i*, *j* runs only in the band (*j_start* to *j_end* from the Itakura parallelogram).

**Backtrace**  
From (*m*-1, *n*-1) walk backwards using *p*; `warp_func[i]` is the *j* at row *i* on the optimal path.

**Summary**: Given a similarity matrix **M** and warp factor level **wfl**, the Numba core solves the Itakura-constrained DTW (maximize cumulative similarity in the band) and returns the normalized similarity and the warp function mapping the first contour’s indices to the second’s.

### 2. ART Neural Network (`core/art.py`)

**Purpose**: Implement Adaptive Resonance Theory components.

**Key Functions**:
- `activate_categories()`: Bottom-up processing (input to category activation)
- `calculate_match()`: Top-down processing (category validation)
- `sort_categories_by_activation()`: Category search ordering
- `check_resonance()`: Vigilance threshold comparison

**ART Cycle**:
```
Input → Activate Categories → Sort by Activation
  ↓                                ↓
Update Weights ← Match OK? ← Calculate Match
  ↓                ↓ NO
Done          Try Next Category
```

### 3. Weight Management (`core/weights.py`)

**Purpose**: Handle category prototype storage and updates.

**Key Functions**:
- `initialize_weight_matrix()`: Create empty weight matrix
- `add_new_category()`: Add new category column
- `update_weights()`: Apply learning rule with length adaptation
- `get_weight_contour()`: Extract category prototype

**Weight Update Algorithm**:
1. Warp input to match weight length using DTW result
2. Update content: `new_weight = old_weight + lr * (warped_input - old_weight)`
3. Calculate new length: `new_length = old_length + lr * (input_length - old_length)`
4. Unwarp and interpolate to new length
5. Store in weight matrix

### 4. Main Network (`core/network.py`)

**Purpose**: Orchestrate the complete ARTwarp algorithm.

**Class**: `ARTwarp`

Constructor parameters (all passed from CLI `train` where applicable):

- `vigilance` (float, default 85.0): Match threshold [1, 99]
- `learning_rate` (float, default 0.1): Weight update rate (0, 1]
- `bias` (float, default 0.0): Activation bias [0, 1]
- `max_categories` (int, default 100): Maximum categories (CLI `--max-categories` default is 50)
- `max_iterations` (int, default 50): Maximum training iterations
- `warp_factor_level` (int, default 3): DTW warping factor (CLI `--warp-factor`)
- `random_seed` (int, optional): Reproducibility (CLI `--seed`)
- `verbose` (bool, default True): Progress output (CLI `-q`/`--quiet` inverts)

Resampling is **not** a network parameter: `--resample`, `--sample-interval`, and `--tempres` are applied in the CLI by loading contours with `return_tempres=True`, calling `resample_contours(contours, tempres_list, sample_interval)`, then passing the resampled contours to `ARTwarp(...).fit(contours, names)`.

- Implements training loop
- Handles convergence detection
- Provides prediction interface

**Training Algorithm**:
```python
for iteration in range(max_iterations):
    shuffle(samples)
    num_changes = 0
    
    for sample in samples:
        # activate and sort categories
        activations = activate_categories(sample)
        sorted_categories = sort(activations)
        
        # search for resonance
        for category in sorted_categories:
            match = calculate_match(sample, category)
            
            if match > vigilance:
                update_weights(category, sample)
                assign(sample, category)
                break
        else:
            # no resonance found
            if num_categories < max_categories:
                create_new_category(sample)
            
        if category_changed:
            num_changes += 1
    
    if num_changes == 0:
        converged = True
        break
```

### 5. Data I/O (`io/loaders.py`, `io/exporters.py`)

**Purpose**: Handle various file formats for input and output.

**Supported Input Formats**:
- `.ctr` files (MATLAB format via scipy.io.loadmat)
- `.csv` files (frequency in specified column)
- `.txt` files (tab-delimited)

**Output Formats**:
- Pickle files (`.pkl`) for complete results
- CSV files for category assignments
- Individual CSV files for reference contours

### 6. Validation (`utils/validation.py`)

**Purpose**: Input validation and error checking.

**Validation Types**:
- Contour format validation (numpy array, 1D, positive values)
- Parameter range validation
- Length compatibility checks

### 7. Resampling (`utils/resample.py`)

**Purpose**: Optional resampling of contours to a uniform temporal resolution (preprocessing only; not part of the ARTwarp class).

**Key Function**: `resample_contours(contours, tempres, sample_interval_sec)`  
Matches MATLAB ARTwarp "Resample Contours" option: same formula as `interp1(1:length(contour), contour, 1:sampleInterval/tempres:length(contour))`. Use before `fit()` when contours have different sampling rates.

**CLI** (`train` command): `--resample` enables resampling before training. `--sample-interval SEC` (default 0.02) is the target sampling interval in seconds. `--tempres SEC` (default 0.01) is the default temporal resolution (seconds per point) for contours that do not provide it (e.g. some .ctr files). When `--resample` is set, contours are loaded with `return_tempres=True`, missing tempres are filled with `--tempres`, then `resample_contours(contours, tempres_list, sample_interval)` is called; the resampled contours are passed to `ARTwarp(...).fit(contours, names)`. These options are not passed into the ARTwarp constructor.

## Alignment with MATLAB ARTwarp

This implementation is a rewrite of [artwarp](https://github.com/dolphin-acoustics-vip/artwarp); the following keeps behavior aligned.

| Aspect | MATLAB | Python |
|--------|--------|--------|
| **Category indices** | 1-based (1 .. numCategories) | 0-based (0 .. num_categories-1) internally; use `one_based_categories=True` in `export_category_assignments` for .csv |
| **Reference contour export** | SaveRefContours.m: `refContour_1.csv`, `%7.1f` per line | `export_reference_contours(..., prefix="refContour")` (default), same numeric format |
| **Valid weight positions** | `find(weight > 0)` for activation, match, update | `(weight > 0) & np.isfinite(weight)` in art.py and weights.py |
| **DTW length ratio** | Reject when `max(m,n)/(min(m,n)-1) >= warpFactorLevel` | Same; check runs first (before short-contour branch) |
| **Load saved run** | "Load Categorisation" (.mat with NET, DATA) | `load_mat_categorisation(filepath)` in `artwarp.io.loaders` |
| **Resample option** | GUI: resample to sampleInterval (ms) using tempres | `resample_contours(contours, tempres, sample_interval_sec)` in `artwarp.utils`; CLI: `--resample --sample-interval 0.02 --tempres 0.01` |
| **.ctr / .csv / .txt** | Load_Data, Load_CSV_Data, Load_TabDelim_Data | `load_contours(..., file_format='ctr'|'csv'|'txt')`, same contour and tempres; optional `return_tempres=True` for resampling |

Algorithm steps (warp, unwarp, activate, match, update_weights, add_new_category, training loop and convergence) match the MATLAB logic.

### Feature parity with MATLAB artwarp

| MATLAB | Python | Notes |
|--------|--------|--------|
| warp.m, unwarp.m | core/dtw.py | Same algorithm, length ratio and constraints aligned |
| ARTwarp_Calculate_Match, _Activate_Categories, _Add_New_Category, _Update_Weights | core/art.py, core/weights.py | Same logic; valid = weight > 0 |
| ARTwarp_Run_Categorisation | core/network.py `fit()` | Same loop, convergence, resample option via `resample_contours()` |
| Load_Data, Load_CSV_Data, Load_TabDelim_Data | io/loaders.py `load_contours()` | .ctr, .csv, .txt; same contour/tempres handling |
| SaveRefContours.m | io/exporters.py `export_reference_contours()` | refContour_1.csv, %7.1f |
| Load Categorisation (.mat) | io/loaders.py `load_mat_categorisation()` | NET + optional DATA |
| Resample option (GUI) | utils/resample.py `resample_contours()`; CLI `train --resample` | Same formula; CLI uses `--sample-interval`, `--tempres` |
| Get_Parameters, Create_Figure, Plot_Net, Plot_Net2 | CLI + API + visualization/ | No GUI; params via constructor/CLI; plots via `plot_*` |
| ARTwarp_Assess_Net | — | Not implemented (species misclassification diagnostic) |
| ARTwarp_Test_Net (assign best, no vigilance) | `predict()` uses vigilance | Can add `assign_best_only` if needed |
| TempRes3.m (CSV→.ctr) | — | Unnecessary; we load CSV directly |
| DiscoveryCurves.m, RenameFiles.m, whistplot_RefContours.m | — | Standalone/helper scripts; not core algorithm |

The **core algorithm and I/O** are complete and aligned. The only missing “ARTwarp” behaviors are: optional Assess_Net-style diagnostic, optional Test_Net “assign best category only” mode, and the standalone helper scripts (DiscoveryCurves, RenameFiles, whistplot).

## Data Structures

### Contour Representation

Contours are represented as 1D NumPy arrays of frequency values:
```python
contour = np.array([100.0, 200.0, 300.0, 250.0, 150.0])
```

### Weight Matrix Structure

The weight matrix stores category prototypes:
- Shape: `(max_features, num_categories)`
- Each column is a category prototype
- NaN values pad shorter contours to common length
- Example:
  ```
  [[100.0, 500.0],
   [200.0, 600.0],
   [300.0, 700.0],
   [  NaN,   NaN]]  # Category 1 has length 3, category 2 has length 3
  ```

### Training Results

`TrainingResults` dataclass contains:
```python
@dataclass
class TrainingResults:
    categories: NDArray[np.float64]      # category assignments
    matches: NDArray[np.float64]         # match values
    weight_matrix: NDArray[np.float64]   # final weights
    num_categories: int                  # number of categories created
    num_iterations: int                  # iterations performed
    converged: bool                      # convergence status
    iteration_history: List[tuple]       # per-iteration statistics
    training_time: float                 # total training time
```

## Performance Optimizations

### 1. Vectorization

Replace nested loops with NumPy operations:
- **Before** (MATLAB): Nested loops for similarity matrix
- **After** (Python): Broadcasting operation, significantly faster :D

### 2. Caching Potential

Current implementation doesn't cache DTW results, but this is a major optimization opportunity:
- Many contours compared to same categories repeatedly
- DTW results don't change until weights update
- Cache hit rate could be >90% in later iterations

Author note:

- With regards to caching, reusing DTW results is tricky because weights change every time a category is updated, so cache invalidation is non-trivial and could easily be wrong. More importantly is parallel processing opportunities (seen below).

### 3. Parallel Processing

Opportunities for parallelization (TODO):
- Category activation (each category independent)
- Batch processing multiple contours
- Could use `multiprocessing` or `joblib`

### 4. JIT Compilation (DTW core)

The main performance win is **Numba JIT** on the DTW core (`_dtw_core_numba` in `core/dtw.py`): the dynamic-programming fill and backtrace run as compiled native code, which is why ARTwarp-py can run faster than MATLAB. See the [DTW core: Numba JIT kernel](#dtw-core-numba-jit-kernel-_dtw_core_numba) subsection above. Additional Numba (e.g. match calculation, weight interpolation) could be added later.

### 5. Memory Efficiency

- Sparse matrix representations (TODO)
- Lazy evaluation of DTW when possible
- Streaming processing for large datasets

## Extension Points

### Adding New DTW Variants

Implement alternative DTW algorithms by:
1. Creating new function in `core/dtw.py`
2. Following same signature as `dynamic_time_warp()`
3. Update `activate_categories()` to use new function

### Adding New File Formats

Add new loaders in `io/loaders.py`:
1. Implement `load_<format>_file()` function
2. Return dictionary with `contour`, `tempres`, `ctrlength`
3. Update `load_contours()` to support new format

### Custom Visualization

Extend visualization by:
1. Adding functions in `visualization/plotting.py` (or a new module under `visualization/`)
2. Using results from `TrainingResults`
3. Plotting reference contours, category assignments, etc.

## Testing Strategy

### Unit Tests
- Test individual functions in isolation
- Mock dependencies
- Fast execution (<1s)
- Example: `test_dtw.py`

### Integration Tests
- Test component interactions
- Use small real-world data
- Medium execution (1-10s)
- Example: Training on synthetic contours

## Code Quality Standards

### Type Hints
All functions have complete type hints:

```python
def dynamic_time_warp(
    u1: NDArray[np.float64],
    u2: NDArray[np.float64],
    warp_factor_level: int = 3
) -> Tuple[float, NDArray[np.int32]]:
    ...
```

### Documentation
- Module-level docstrings explain purpose
- Function docstrings use Google style
- Include Args, Returns, Raises, Examples
- Algorithm descriptions for complex functions

### Code Style
- Black formatter (line length 100)
- isort for import sorting
- flake8 for linting
- mypy for type checking

## Future Enhancements (TODO)

### Planned Features
1. GPU acceleration (CuPy/JAX)
2. DTW result caching (maybe...)
3. Parallel batch processing
4. Approximate DTW algorithms
5. Interactive visualization
6. Web API interface
7. PyPI packaging (for Python PIP download)

### Optimization Opportunities
1. Numba on other hot paths (match calculation, weight interpolation)
2. Sparse weight matrices
3. Early stopping heuristics
4. Hierarchical clustering preprocessing
5. Incremental learning mode

View docs/PERFORMANCE_OPTIMIZATIONS.md for more details!

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)