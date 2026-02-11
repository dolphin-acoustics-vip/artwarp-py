# ARTwarp-py Test Results

**Date**: February, 2026  
**Version**: 2.0.2  
**Environment**: Linux, Python 3.14.2 (miniconda sig-process)  
**Status**: ALL TESTS PASSING :D

---

## Test Summary

```
===================================================================== test session starts ======================================================================
platform linux -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /home/pedroggbm/Documents/vp4038-dolphin-acoustics/ARTWarp/artwarp-py
configfile: pyproject.toml
plugins: cov-7.0.0
collected 136 items
===================================================================== 136 passed in 6.39s ======================================================================
```

### Overall Statistics
- **Total Tests**: 136
- **Passed**: 136 (100%)
- **Failed**: 0
- **Skipped**: 0
- **Duration**: ~6 seconds

---

## Test Breakdown by Module

### 1. ART Neural Network Tests (`test_art.py`)
**Status**: 17/17 passed

#### Activate Categories (4 tests)
- `test_empty_weight_matrix` - Empty weight matrix handling
- `test_single_category_perfect_match` - Single category activation
- `test_bias_effect` - Bias parameter effect on activations
- `test_multiple_categories` - Multiple category activation

#### Calculate Match (5 tests)
- `test_identical_contours` - Perfect match calculation
- `test_different_contours` - Different contour matching
- `test_empty_contours` - Empty contour edge case
- `test_mismatched_lengths` - Length mismatch handling
- `test_match_formula` - Match formula correctness

#### Sort Categories (4 tests)
- `test_descending_order` - Correct sorting by activation
- `test_empty_activations` - Empty activation array
- `test_single_activation` - Single category sorting
- `test_duplicate_activations` - Duplicate values handling

#### Check Resonance (4 tests)
- `test_match_above_vigilance` - Above vigilance threshold
- `test_match_below_vigilance` - Below vigilance threshold
- `test_match_equals_vigilance` - At vigilance threshold
- `test_boundary_conditions` - Boundary value testing

---

### 2. Dynamic Time Warping Tests (`test_dtw.py`)
**Status**: 28/28 passed

#### Similarity Matrix (4 tests)
- `test_identical_contours` - Identical contour similarity
- `test_different_lengths` - Different length handling
- `test_similarity_calculation` - Similarity formula correctness
- `test_vectorization` - Vectorized computation

#### DTW Algorithm (6 tests)
- `test_identical_contours_perfect_match` - Perfect match scenario
- `test_length_ratio_constraint` - Length ratio enforcement
- `test_simple_time_stretch` - Time stretching handling
- `test_warp_factor_level_effect` - Warp factor flexibility
- `test_single_element_contours` - Single-element edge case
- `test_different_frequency_ranges` - Frequency range handling

#### Python Path DTW (9 tests) — coverage of dtw.py when Numba disabled
- `test_identical_contours_python_path` - Identical via Python DP
- `test_length_ratio_reject_python_path` - Length ratio rejection
- `test_simple_time_stretch_python_path` - Time stretch via Python DP
- `test_short_contours_simple_alignment_python_path` - Short contours m==n diagonal
- `test_short_contours_m_shorter_python_path` - Short contours m < n
- `test_short_contours_n_shorter_python_path` - Short contours n < m (clip branch)
- `test_full_python_dp_stage` - Full Python DP (early + general stage)
- `test_python_dp_warp_factor_2` - Python DP with warp_factor_level=2
- `test_python_dp_identical_longer` - Longer identical contours (general stage)

#### Unwarp Function (4 tests)
- `test_identity_warp` - Identity mapping
- `test_compression_warp` - Time compression
- `test_expansion_warp` - Time expansion
- `test_empty_warp` - Empty array handling

#### Python Path Unwarp (3 tests) — coverage when Numba disabled
- `test_identity_warp_python_path` - Identity via Python unwarp
- `test_compression_warp_python_path` - Compression via Python unwarp
- `test_expansion_warp_python_path` - Expansion via Python unwarp

#### Performance (2 tests)
- `test_large_contours` - Large contour stress test (500+ points)
- `test_many_small_warps` - Multiple warp operations

---

### 3. Loaders Tests (`test_loaders.py`)
**Status**: 22/22 passed

#### Load CTR File (6 tests)
- `test_load_fcontour` - Load .ctr with 'fcontour' variable
- `test_load_fcontour_scalar_becomes_1d` - Single scalar fcontour becomes 1D array
- `test_load_freq_contour_drops_last` - Load with 'freqContour' (MATLAB: drop last element)
- `test_load_ctr_with_tempres_and_ctrlength` - .ctr with tempres and ctrlength
- `test_load_ctr_ctrlength_inferred_from_tempres` - ctrlength inferred from tempres
- `test_load_ctr_no_fcontour_or_freq_raises` - Missing fcontour/freqContour raises

#### Load CSV/TXT (4 tests)
- `test_load_csv_basic` - Load CSV with frequency column
- `test_load_csv_frequency_column_out_of_range_raises` - Invalid column index
- `test_load_txt_tab_delimited` - Load tab-delimited file
- `test_load_txt_frequency_column_out_of_range_raises` - Invalid column index

#### Load Contours (7 tests)
- `test_load_contours_csv_directory` - Load from directory of CSV files
- `test_load_contours_with_return_tempres` - return_tempres=True returns third element
- `test_load_contours_directory_not_found_raises` - FileNotFoundError
- `test_load_contours_not_a_directory_raises` - NotADirectoryError
- `test_load_contours_unknown_format_raises` - Unknown format ValueError
- `test_load_contours_no_files_raises` - No files found ValueError
- `test_load_contours_auto_detects_extensions` - file_format='auto'

#### Load MAT Categorisation Extended (2 tests)
- `test_load_mat_data_empty_returns_net_only` - DATA present but empty
- `test_load_mat_weight_1d_reshaped_to_2d` - 1-d weight reshaped to (n, 1)

---

### 4. MATLAB Compatibility Tests (`test_matlab_compat.py`)
**Status**: 9/9 passed

(Load NET only, missing NET, file not found, export one-based/zero-based, resample, ref contour prefix.)

---

### 5. Network Tests (`test_network.py`)
**Status**: 18/18 passed

#### Initialization (6 tests)
- `test_default_initialization` - Default parameter values
- `test_custom_parameters` - Custom parameter setting
- `test_invalid_vigilance` - Vigilance validation
- `test_invalid_learning_rate` - Learning rate validation
- `test_invalid_bias` - Bias validation
- `test_random_seed_reproducibility` - Reproducible results

#### Training (7 tests)
- `test_simple_training` - Basic training workflow
- `test_identical_contours_one_category` - Single category formation
- `test_max_categories_limit` - Maximum category enforcement
- `test_convergence_detection` - Convergence criteria
- `test_empty_contours_list` - Empty input handling
- `test_contour_names` - Contour name tracking
- `test_training_results_structure` - Results structure correctness

#### Prediction (3 tests)
- `test_predict_after_training` - Post-training prediction
- `test_predict_before_training_error` - Pre-training error handling
- `test_predict_dissimilar_contour` - Dissimilar contour rejection

#### Training Results (2 tests)
- `test_get_category_sizes` - Category size calculation
- `test_get_uncategorized_count` - Uncategorized count

---

### 6. Validation Tests (`test_validation.py`)
**Status**: 25/25 passed

#### Validate Contour (8 tests)
- `test_valid_contour_passes` - Valid 1D positive numeric array
- `test_not_ndarray_raises` - Non-numpy array raises
- `test_2d_raises` - 2D array raises
- `test_empty_raises` - Empty array raises
- `test_non_numeric_dtype_raises` - Non-numeric dtype raises
- `test_nan_raises` - NaN values raise
- `test_inf_raises` - Infinite values raise
- `test_non_positive_raises` - Zero or negative values raise

#### Validate Contours (4 tests)
- `test_valid_list_passes` - Valid list of contours
- `test_not_list_raises` - Non-list raises
- `test_empty_list_raises` - Empty list raises
- `test_invalid_contour_at_index_raises` - Invalid contour re-raises with index

#### Validate Parameters (13 tests)
- `test_valid_parameters_pass` - All valid passes
- Vigilance / learning_rate / bias / max_categories / max_iterations / warp_factor_level boundary and type checks

---

### 7. Visualization Tests (`test_visualization.py`)
**Status**: 19/19 passed

#### Visualization Functions (14 tests)
- `test_plot_training_summary` - Multi-panel summary plot
- `test_plot_reference_contours` - Reference contour grid
- `test_plot_reference_contours_empty` - Empty matrix handling
- `test_plot_category_distribution` - Category bar chart
- `test_plot_convergence_history` - Convergence line plot
- `test_plot_convergence_history_no_data` - No history handling
- `test_plot_contours_by_category` - Per-category plots
- `test_plot_contours_by_category_empty` - Empty category handling
- `test_plot_match_distribution` - Match histogram
- `test_plot_match_distribution_no_categories` - No categories case
- `test_create_results_report` - Comprehensive report
- `test_figure_save_functionality` - File saving
- `test_custom_figure_sizes` - Size customization
- `test_dpi_specification` - DPI control

#### Edge Cases (4 tests)
- `test_single_category` - Single category visualization
- `test_many_categories` - Many categories handling
- `test_with_uncategorized_samples` - Uncategorized samples
- `test_very_many_categories_renders` - Training summary, category distribution, and reference contours with >25 categories (no legend, thinned x-axis labels)

---

## Bug Fixes Applied

### Issue 1: All-NaN Slice Error
**Problem**: DTW algorithm encountered "All-NaN slice" error when Itakura parallelogram constraints left no valid paths.

**Solution**: Added checks for all-NaN arrays before calling `np.nanargmax()` in three locations:
- Early stage alignment
- Main alignment
- General stage alignment

**Files Modified**: `src/artwarp/core/dtw.py`

### Issue 2: Single-Element Contours
**Problem**: Single-element contours (length=1) caused division by zero in length ratio calculation.

**Solution**: Added special case handling for single-element contours, returning exact similarity directly.

**Files Modified**: `src/artwarp/core/dtw.py`

### Issue 3: Very Short Contours
**Problem**: Contours with length < 2*warp_factor_level had overly restrictive Itakura constraints.

**Solution**: Added simple alignment strategy for very short contours with reasonable length differences.

**Files Modified**: `src/artwarp/core/dtw.py`

### Issue 4: Test Expectations
**Problem**: Test expected 66.67% similarity but got 50.0% (mathematically correct) with simple alignment.

**Solution**: Corrected test expectations to match mathematical formula: min/max = 0.5 → 50%.

**Files Modified**: `tests/unit/test_dtw.py`

---

## Test Coverage

Coverage is configured in `pyproject.toml`: `**/__init__.py`, `tests/*`, and `src/artwarp/cli/main.py` are omitted from measurement; only code under `src` is measured. The CLI is omitted so that overall coverage reflects library code; CLI behaviour is exercised indirectly via integration. To regenerate the HTML report (requires `pytest-cov`):

```bash
python -m pytest tests/ --cov=artwarp --cov-report=html
```

Then open `htmlcov/index.html` in a browser.

### Core Modules
- **DTW Algorithm**: High coverage (all paths tested)
- **ART Network**: High coverage (all operations tested)
- **Weight Management**: Covered via network tests
- **Network Training**: High coverage
- **Visualization**: High coverage
- **Validation** (`utils/validation.py`): Covered by `test_validation.py` (validate_contour, validate_contours, validate_parameters)
- **Loaders** (`io/loaders.py`): Covered by `test_loaders.py` and `test_matlab_compat.py` (load_ctr_file, load_csv_file, load_txt_file, load_contours, load_mat_categorisation)

### Edge Cases Tested
- Empty inputs
- Single-element arrays
- Very short contours
- Very long contours (500+ points)
- Length ratio violations
- Identical contours
- Completely different contours
- Maximum category limits
- Convergence scenarios
- Uncategorized samples

### Error Handling Tested
- Invalid parameters (vigilance, learning rate, bias)
- Prediction before training
- Empty contour lists
- Mismatched contour lengths
- All-NaN scenarios
- Division by zero protection

---

## Continuous Integration

### GitHub Actions CI

The repository uses **GitHub Actions** (`.github/workflows/ci.yml`) for PR and push checks:

- **Triggers**: Push and pull requests to `main` / `master`.
- **Test job**: Runs on Python 3.9, 3.10, 3.11, 3.12 with `pip install -e ".[dev,accelerate]"`, then:
  - `pytest tests/` with coverage
  - **Coverage gate**: `--cov-fail-under=80` — PRs **must** keep overall coverage ≥ 80% or the check fails.
- **Lint job**: Black, isort, Flake8, Mypy on `src/artwarp/` and `tests/`.
- **Artifact**: On PRs, the workflow uploads the `htmlcov/` report as an artifact so you can download and inspect coverage.

**Best practice**: Require the CI workflow to pass (and optionally “Require status checks to pass before merging”) in the branch protection rules for `main`/`master` so only green PRs are merged.

### Test Command (local)
```bash
python -m pytest tests/ -v --tb=short
```

### Coverage Command (local)
```bash
python -m pytest tests/ --cov=artwarp --cov-report=html --cov-fail-under=80
```

### Linting (local)
```bash
black --check src/artwarp/ tests/
isort --check-only src/artwarp/ tests/
flake8 src/artwarp/ --max-line-length=100 --extend-ignore=E203,W503
mypy src/artwarp/ --ignore-missing-imports
```

---

## Compatibility Matrix

| Python Version | NumPy | SciPy | Pandas | Matplotlib | Status |
|----------------|-------|-------|--------|------------|--------|
| 3.13.11 | 2.4.1 | 1.17.0 | 3.0.0 | 3.10.8 | Verified |
| 3.11.x | 1.23+ | 1.9+ | 1.5+ | 3.6+ | Expected |
| 3.10.x | 1.21+ | 1.7+ | 1.3+ | 3.5+ | Expected |
| 3.9.x | 1.20+ | 1.7+ | 1.3+ | 3.4+ | Expected |
| 3.8.x | 1.20+ | 1.7+ | 1.3+ | 3.4+ | ⚠️ Should work |

---

## Test Execution Environment

### System Information
```
OS: Linux 6.18.5-arch1-1
Architecture: x86_64
Python: 3.14.2
Pytest: 9.0.2
Virtual Environment: venv_artwarp | sig-process (conda)
```

### Installed Packages
```
numpy==2.4.1
scipy==1.17.0
pandas==3.0.0
matplotlib==3.10.8
pytest==9.0.2
pytest-cov==7.0.0
black==26.1.0
mypy==1.19.1
flake8==7.3.0
isort==7.0.0
```

---

## Conclusion

**ARTwarp-py v2.0.3 passes all 136 tests with 100% success rate.**

## Author Note

Although this specific internal documentation .md file shouldn't be regularly updated because it's tedious to add/remove tests and update correspondingly, please keep this somewhat up-to-date every couple months. 

I would highly advise you simply prompting an assisting tool to, based on CLI output of test runs, modify @TEST_RESULTS.md.

Pretty please.

Thank you.

---

_Test execution completed: February, 2026_  
_All tests passed: 136/136 (100%)_  
_Status: READY FOR USE_

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)

