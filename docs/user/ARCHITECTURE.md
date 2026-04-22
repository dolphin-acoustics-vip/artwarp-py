# ARTwarp-py Architecture

This document describes the internal architecture of ARTwarp-py.

Although it's not necessary for end-users to fully understand this document, it might be a useful entry-point for potential developers (in which case, view docs/dev/).

## Overview

ARTwarp-py is organized into modular components that separate concerns and enable
easy testing, maintenance, and extension.

```
artwarp-py/
├── run.sh                # interactive launcher (menu: Train / Plot / Predict / Export)
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

If this is a bit counter-intuitive, I suggest you go on Neetcode; they have great resources explaining 1D and 2D Dynamic Programming! 

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

**Weight Update Algorithm** (when `compare_warped` is False, matching MATLAB `compareWarped == 0`):
1. Warp input to match weight length using DTW result
2. Update content: `new_weight = old_weight + lr * (warped_input - old_weight)`
3. Calculate new length: `new_length = old_length + lr * (input_length - old_length)`
4. Unwarp and interpolate to new length
5. Store in weight matrix

When `compare_warped` is True (MATLAB `compareWarped == 1`), the implementation follows `ART_Update_Weights.m`: fixed reference length (no unwarp pass) and content update from the warped input as in MATLAB.

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
- `recat_single_categories` (bool, default False): MATLAB `recatSingleCats` (CLI `--recat-single-categories`)
- `compare_warped` (bool, default False): MATLAB `compareWarped` (CLI `--compare-warped`)
- `deprioritize_lone_category_search` (bool, default False): **Not in MATLAB `stable`.** Optional search-order tweak from merged experimental PR (`martion2007/delete_unused_categories`): when a sample is the only contour in its category, try other categories before its current one in the resonance loop (CLI `--deprioritize-lone-category-search`).
- `purge_empty_categories` (bool, default False): **Not in MATLAB `stable`.** Optional post-iteration cleanup from the same PR: remove weight columns with **zero** assigned contours and reindex labels (CLI `--purge-empty-categories`). Distinct from MATLAB’s **`ARTwarp_Recat_Single_Cats`** (Kai / `recatSingleCats`), which only removes a column after a **successful** lone-contour reassignment.

Resampling is **not** a network parameter: the **`train`** CLI applies resampling by default (MATLAB `resample=1`); `--no-resample` turns it off. With resampling on, the CLI loads contours with `return_tempres=True`, calls `resample_contours(contours, tempres_list, sample_interval)`, then passes contours to `ARTwarp(...).fit(contours, names)`.

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
- Pickle files (`.pkl`) for complete results (includes `category_parent_names`)
- CSV files for category assignments
- Individual CSV files for reference contours
- CSV provenance metadata (`metadata.csv`) — UUID per reference contour + parent contour names; written automatically alongside reference contours when using `--export-refs` or `export_reference_contour_metadata()`

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

**CLI** (`train` command): resampling is **on by default** (`--no-resample` to match MATLAB `resample=0`). `--sample-interval SEC` defaults to **0.01** (MATLAB `sampleInterval`). `--tempres SEC` (default 0.01) fills missing per-contour temporal resolution when resampling. These options are preprocessing only, not `ARTwarp` constructor fields.

### 8. Numba check (`utils/numba_check.py`)

**Purpose**: Report whether Numba is installed (for DTW performance optimizations) and, when using the CLI, optionally offer to install it via pip/conda.

**Key functions**: `numba_available()`, `report_numba_status()` (status only), `check_numba(offer_install=True)` (status + optional install prompt). The package calls `report_numba_status()` on import so API users (e.g. in Jupyter) see the Numba status. The CLI calls `check_numba(offer_install=stdin.isatty())` at startup so interactive runs can prompt to install Numba if missing.

**Entry points**: `run.sh` (interactive launcher) and `artwarp-py` / `python -m artwarp.cli.main` (CLI) both trigger the Numba check when run; see **Quick Start** in the main README.

## Tracking MATLAB `stable` v2.0 [NEW]

The reference MATLAB implementation is the sibling directory **`../artwarp`** tracked on branch **`stable`**. Keep that checkout **up to date** (`git pull` there) when you need the latest upstream behaviour; artwarp-py does not auto-sync to remote. Training behaviour in artwarp-py is implemented to match **`ARTwarp_Run_Categorisation.m`**, **`ARTwarp_cli_mode.m`** (CLI defaults), **`ARTwarp_Update_Weights.m`**, **`ARTwarp_Recat_Single_Cats.m`**, **`ARTwarp_Average_Weights.m`**, and related `.m` files as they exist on **`stable`** at your pulled revision.

**Not identical run-for-run to MATLAB:** sample order uses different RNG (`sort(randn(n,1))` vs NumPy). The **`ARTwarp`** Python class still defaults **`bias=0.0`** (typical GUI); the **`train`** CLI defaults **`bias=1e-6`** and **resampling on**, matching **`ARTwarp_cli_mode.m`**.

## Alignment with MATLAB ARTwarp

This implementation is a rewrite of [artwarp](https://github.com/dolphin-acoustics-vip/artwarp); the following keeps behavior aligned.

| Aspect | MATLAB | Python |
|--------|--------|--------|
| **Category indices** | 1-based (1 .. numCategories) | 0-based (0 .. num_categories-1) internally; use `one_based_categories=True` in `export_category_assignments` for .csv |
| **Reference contour export** | SaveRefContours.m: `refContour_1.csv`, `%7.1f` per line | `export_reference_contours(..., prefix="refContour")` (default), same numeric format |
| **Reference contour provenance** | `REFCONTOURS(i).id` (UUID), `REFCONTOURS(i).parent_ids` (input IDs); saved in `.ctr` | `TrainingResults.category_parent_names`; `export_reference_contour_metadata()` → `metadata.csv` with `category`, `ref_contour_id`, `parent_contour_name` |
| **Valid weight positions** | `find(weight > 0)` for activation, match, update | `(weight > 0) & np.isfinite(weight)` in art.py and weights.py |
| **DTW length ratio** | Reject when `max(m,n)/(min(m,n)-1) >= warpFactorLevel` | Same; check runs first (before short-contour branch) |
| **Load saved run** | "Load Categorization" (.mat with NET, DATA) | `load_mat_categorization(filepath)` in `artwarp.io.loaders` |
| **Resample option** | GUI / CLI `ARTwarp_cli_mode`: resample default **on**, `sampleInterval` default **0.01** s | `resample_contours(...)` in `artwarp.utils`; CLI defaults resample **on** (`--no-resample` to disable), `--sample-interval` default **0.01**, `--tempres` default **0.01** |
| **recatSingleCats** | Optional after each iteration: `ARTwarp_Recat_Single_Cats.m` | `recat_single_categories=True` or CLI `--recat-single-categories` (default off, same as MATLAB unchecked) |
| **compareWarped** | Optional: `compareWarped` in `ARTwarp_Update_Weights`; then `ARTwarp_Average_Weights.m` each iteration | `compare_warped=True` or CLI `--compare-warped` (default off) |
| **.ctr / .csv / .txt** | Load_Data, Load_CSV_Data, Load_TabDelim_Data | `load_contours(..., file_format='ctr'|'csv'|'txt')`, same contour and tempres; optional `return_tempres=True` for resampling |

Algorithm steps (warp, unwarp, activate, match, update_weights, add_new_category, training loop and convergence) match the MATLAB logic. Optional post-iteration passes (`recatSingleCats`, `compareWarped`) match `ARTwarp_Run_Categorisation.m` order: inner sample loop, then recat if enabled, then average weights if `compareWarped`, then (if enabled) `purge_empty_categories` via `purge_empty_category_columns()` in `weights.py`, then convergence on inner-loop reclassifications only. The in-loop **`deprioritize_lone_category_search`** step runs only when that flag is on (non-MATLAB).

### Feature parity with MATLAB artwarp

| MATLAB | Python | Notes |
|--------|--------|--------|
| warp.m, unwarp.m | core/dtw.py | Same algorithm, length ratio and constraints aligned |
| ARTwarp_Calculate_Match, _Activate_Categories, _Add_New_Category, _Update_Weights | core/art.py, core/weights.py | Same logic; valid = weight > 0 |
| ARTwarp_Run_Categorisation | core/network.py `fit()` | Same loop and convergence; optional `ARTwarp_Recat_Single_Cats` / `ARTwarp_Average_Weights` when flags set; resample via `resample_contours()` before `fit()` |
| ARTwarp_Recat_Single_Cats | `_recat_single_categories()` | When `recat_single_categories` / `--recat-single-categories` |
| ARTwarp_Average_Weights | `average_weights()` in weights.py | When `compare_warped` / `--compare-warped` (after recat block in MATLAB order) |
| *(none on MATLAB `stable`)* | `purge_empty_category_columns()` in `weights.py` | Optional: `purge_empty_categories` / `--purge-empty-categories` after recat + compare-warped |
| *(none on MATLAB `stable`)* | `deprioritize_lone_category_search` in `network.py` | Optional: `--deprioritize-lone-category-search` inside sample loop |
| ARTwarp_Update_Weights (compare_warped) | `update_weights(..., compare_warped=...)` | Matches `ART_Update_Weights.m` branches |
| Load_Data, Load_CSV_Data, Load_TabDelim_Data | io/loaders.py `load_contours()` | .ctr, .csv, .txt; same contour/tempres handling |
| SaveRefContours.m | io/exporters.py `export_reference_contours()` | refContour_1.csv, %7.1f |
| REFCONTOURS struct (id, parent_ids) | `TrainingResults.category_parent_names`; `export_reference_contour_metadata()` | CSV instead of .ctr; UUID per prototype, parent names list |
| Load Categorization (.mat) | io/loaders.py `load_mat_categorization()` | NET + optional DATA |
| Resample option (GUI / CLI) | utils/resample.py `resample_contours()`; CLI resample **on** by default | Same formula; `--no-resample` disables; `--sample-interval` / `--tempres` match `ARTwarp_cli_mode.m` defaults |
| Get_Parameters, Create_Figure, Plot_Net, Plot_Net2 | CLI + API + visualization/ | No GUI; params via constructor/CLI; plots via `plot_*` |
| ARTwarp_Assess_Net | — | Not implemented (species misclassification diagnostic) |
| ARTwarp_Test_Net (assign best, no vigilance) | `predict()` uses vigilance | Can add `assign_best_only` if needed |
| TempRes3.m (CSV→.ctr) | — | Unnecessary; we load CSV directly |
| DiscoveryCurves.m | visualization/plotting.py `plot_discovery_curve()` | Cumulative categories vs sample order (discovery curve) |

The **core algorithm and I/O** are complete and aligned. The only missing “ARTwarp” behaviors are: optional Assess_Net-style diagnostic, optional Test_Net “assign best category only” mode, and the standalone helper scripts (RenameFiles, whistplot_RefContours). DiscoveryCurves.m is implemented as `plot_discovery_curve()`.

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
    categories: NDArray[np.float64]              # category assignments
    matches: NDArray[np.float64]                 # match values
    weight_matrix: NDArray[np.float64]           # final weights
    num_categories: int                          # number of categories created
    num_iterations: int                          # iterations performed
    converged: bool                              # convergence status
    iteration_history: List[tuple]               # per-iteration statistics
    training_time: float                         # total training time
    category_parent_names: Dict[int, List[str]]  # provenance: category → contour names
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