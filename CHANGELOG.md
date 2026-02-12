# Changelog

All notable changes to ARTwarp-py will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.3] - 2026-02-09

### Added

- **CLI resample options**: `train` command now supports `--resample`, `--sample-interval SEC` (default 0.02), and `--tempres SEC` (default 0.01), matching the MATLAB ARTwarp "Resample Contours" option. When `--resample` is set, contours are resampled to the given sampling interval before training using `artwarp.utils.resample.resample_contours`.
- **load_contours return_tempres**: `load_contours(..., return_tempres=True)` returns a third value `tempres_list` (seconds per point per contour, or None when unknown), for use with resampling.

## [2.0.2] - 2026-02-09

### Added

- **Load MATLAB categorisation**: `load_mat_categorisation(filepath)` in `artwarp.io.loaders` loads .mat files saved by MATLAB ARTwarp (NET and optionally DATA), returning weight matrix, parameters, and contours/categories when present. Enables "Load Categorisation" workflow from Python.
- **1-based export options**: `export_category_assignments(..., one_based_categories=True)` and `export_reference_contours(..., one_based_filenames=True)` for MATLAB-style output (default for reference contour filenames remains 1-based).
- **Regression tests**: `test_length_ratio_equals_warp_factor_rejected`, `test_match_uses_only_positive_weights`, and `tests/unit/test_matlab_compat.py` for MATLAB alignment and .mat loading.

### Changed

- **DTW length ratio**: Reject when `length_ratio >= warp_factor_level` (was `>`) to match MATLAB `warp.m`. Length-ratio check now runs before the short-contour branch.
- **Valid weight mask**: Activation, match, and weight extraction use `weight > 0` (and finite) for valid positions to match MATLAB `find(weightVector > 0)`.

### Fixed

- DTW rejecting correctly when length ratio equals `warp_factor_level`; test updated for `test_warp_factor_level_effect` so contour lengths keep ratio below the smaller warp level.

## [2.0.1] - 2026-01-29

### Added

- **Professional Visualization Module**: Complete visualization system with 7 plotting functions
  - `plot_training_summary()`: Comprehensive multi-panel overview
  - `plot_reference_contours()`: Category prototype visualization
  - `plot_category_distribution()`: Sample distribution analysis
  - `plot_convergence_history()`: Training convergence monitoring
  - `plot_contours_by_category()`: Per-category detailed view
  - `plot_match_distribution()`: Statistical match analysis
  - `create_results_report()`: Automated comprehensive report generation

- **Visualization Features**:
  - Publication-ready figures with 300 DPI default
  - Professional scientific plotting standards
  - Customizable figure sizes and colors
  - Multiple output formats (PNG, PDF, SVG, EPS)
  - Interactive and batch processing modes
  - Comprehensive documentation (VISUALIZATION.md)

- **Examples and Tests**:
  - `examples/visualization_example.py`: Complete demonstration
  - 75+ visualization test cases in `test_visualization.py`

### Changed

- **Dependencies**: matplotlib now included as core dependency
- **Documentation**: Updated all docs to include visualization sections
- **Setup**: Simplified extras_require with new 'all' option

### Fixed

- None (new feature addition)

## [2.0.0] - 2026-01-29

### Added

- Complete Python reimplementation of ARTwarp algorithm
- High-performance DTW implementation with vectorization
- Modular architecture with clear separation of concerns
- Command-line interface for easy usage
- Comprehensive test suite with unit and integration tests
- Support for multiple input formats (.ctr, .csv, .txt)
- Export functionality for results and reference contours
- Type hints throughout codebase
- Detailed API documentation
- Installation and architecture guides
- Example scripts demonstrating usage

### Performance Improvements

- 10-100x faster DTW computation through vectorization
- Elimination of GUI overhead during training
- Optional Numba JIT compilation support
- Efficient NumPy operations throughout

### Differences from MATLAB Version

- Pure Python implementation (no MATLAB required)
- Command-line interface instead of GUI
- Programmatic API for integration into pipelines
- Better memory efficiency
- Faster execution on most datasets
- Modern Python packaging and distribution

### Maintained Compatibility

- Mathematical equivalence with original MATLAB algorithm
- Same DTW algorithm with Itakura parallelogram constraints
- Identical ART network operations
- Same parameter meanings and ranges
- Compatible output formats

## Original MATLAB Versions

This is a complete rewrite of ARTwarp, originally developed in MATLAB.
For the original MATLAB implementation history, see:
https://github.com/dolphin-acoustics-vip/artwarp

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)