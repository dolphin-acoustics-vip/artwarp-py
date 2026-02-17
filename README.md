# ARTwarp-py

A high-performance Python implementation of ARTwarp for automated categorization of tonal animal sounds.

## Overview

ARTwarp-py is a complete rewrite of the original MATLAB ARTwarp software, combining Dynamic Time Warping (DTW) for contour similarity measurement with an Adaptive Resonance Theory (ART) neural network for unsupervised clustering.

This implementation provides performance improvements while maintaining mathematical equivalence with the original algorithm.

## Features

- **Performance**: Optimized algorithms using NumPy, with optional Numba JIT compilation
- **Multiple Input Formats**: Support for .ctr (MATLAB), .csv, and .txt contour files; load MATLAB categorisation (.mat) with `load_mat_categorisation()`
- **Visualizations**: Publication-ready plots and rigorous reports
- **Parallel Processing Potential**: Multi-core support for batch processing (TODO)
- **Command-Line Interface**: Easy-to-use CLI for programmatic access
- **Testing**: Full test suite ensuring equivalence with MATLAB implementation
- **Modern Python**: Type hints, code coverage, proper documentation, and best practices

## Performance Improvements

Compared to the original MATLAB implementation:
- **Faster DTW computation** through vectorization and optional JIT compilation (benchmarks TODO)
- **Parallel category activation** across multiple CPU cores
- **Efficient caching** of repeated DTW calculations
- **Reduced memory footprint** through sparse matrix operations
- **Batch processing potential** without GUI overhead (TODO)

## Installation

```bash
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- pandas >= 1.3.0 (for CSV/data handling)
- matplotlib >= 3.4.0 (for visualization)
- Numba >= 0.54.0 (optional, for JIT acceleration)

## Quick Start

Or look at QUICK_REFERENCE.md!

Please read INSTALLATION.md for environment setup.

### Command Line

Activate your virtual environment first (e.g. `source venv/bin/activate` or, in Fish, `source venv/bin/activate.fish`), then:

```bash
# train on a directory of contour files
artwarp-py train --input-dir ./contours --output results.pkl \
    --vigilance 85 --learning-rate 0.1 --max-iterations 50

# optionally export reference contours and category assignments
artwarp-py train --input-dir ./contours --output results.pkl --export-refs --export-categories

# resample contours to uniform temporal resolution (like MATLAB resample option)
artwarp-py train --input-dir ./contours --output results.pkl --resample --sample-interval 0.02

# full command
artwarp-py train --input-dir ./contours --output results.pkl --resample --sample-interval 0.02 --vigilance 85 --learning-rate 0.1 --max-iterations 50 --export-refs --export-categories

# generate visualizations from Pickle (.pkl) file
artwarp-py plot --results results.pkl --input-dir ./contours --output-dir ./report
```

### Python API

```python
from artwarp import ARTwarp, load_contours

# load contour data
contours, names = load_contours('path/to/contour/directory', file_format='csv')

# init network
network = ARTwarp(
    vigilance=85.0,
    learning_rate=0.1,
    bias=0.0,
    max_categories=50,
    max_iterations=50,
    warp_factor_level=3
)

# train
results = network.fit(contours, contour_names=names)

# get category assignments
categories = results.categories
reference_contours = results.weight_matrix

# visualize results
from artwarp.visualization import plot_training_summary
plot_training_summary(results)
```

## Architecture

The codebase is organized into modular components:

```
artwarp/
├── core/
│   ├── dtw.py          # Dynamic Time Warping implementation
│   ├── art.py          # ART neural network
│   ├── network.py      # main ARTwarp algorithm
│   └── weights.py      # weight update and management
├── io/
│   ├── loaders.py      # data loading utilities
│   └── exporters.py    # results export
├── visualization/
│   └── plotting.py     # plotting functions
├── utils/
│   ├── validation.py   # input validation
│   └── resample.py     # contour resampling (MATLAB-aligned)
└── cli/
    └── main.py         # command-line interface
```

## Algorithm Details

### Dynamic Time Warping

The DTW implementation uses an optimized dynamic programming approach with Itakura parallelogram constraints:
- Vectorized similarity matrix computation
- Efficient path backtracing
- Optional Numba JIT compilation for speedup

### ART Neural Network

The Adaptive Resonance Theory implementation follows the original ARTwarp design:
- Bottom-up activation (similarity-based category activation)
- Top-down matching (vigilance-based category validation)
- Dynamic category creation
- Incremental weight updates with learning rate

## MATLAB compatibility

- Load a .mat file saved by MATLAB ARTwarp (e.g. after "Run Categorisation"): `from artwarp.io import load_mat_categorisation; data = load_mat_categorisation("ARTwarp85FINAL.mat")` to get `weight_matrix`, `num_categories`, and optional `contours`/`categories`/`contour_names`.
- Export category assignments or reference contours with 1-based indices/filenames via `one_based_categories=True` or `one_based_filenames=True` (see API.md).

## Testing

Run the test suite (use the venv’s Python so dependencies are found):

```bash
python -m pytest tests/ -v
```

With coverage (CI requires ≥80%): `python -m pytest tests/ --cov=artwarp --cov-report=html --cov-fail-under=80`

**CI**: GitHub Actions runs tests (Python 3.9–3.12), coverage gate (80%), and lint (Black, isort, Flake8, Mypy) on push and pull requests. See `docs/TEST_RESULTS.md` for details.

## Citation

If you use this software in your research, please cite:

1. This Python implementation: [DOI to be assigned, hopefully soon]

2. Buck, J. R. & Tyack, P. L. 1993. A quantitative measure of similarity for 
Tursiops truncatus signature whistles. Journal of the Acoustical Society of 
America, 94, 2497-2506. https://doi.org/10.1121/1.407385 

3. Deecke, V. B., Ford, J. K. B. & Spong, P. 1999. Quantifying complex patterns 
of bioacoustic variation: Use of a neural network to compare killer whale 
(Orcinus orca) dialects. Journal of the Acoustical Society of America, 105, 
2499-2507. https://doi.org/10.1121/1.426853 

4. Deecke, V. B. & Janik, V. M. 2006. Automated categorization of bioacoustic
signals: Avoiding perceptual pitfalls. Journal of the Acoustical Society of
America, 119, 645-653. https://doi.org/10.1121/1.2139067

## License

ARTwarp is distributed under the terms of the GNU Lesser General Public
License, version 3, as published by the Free Software Foundation. For
details, please refer to the LICENSE file in the root directory of the
ARTwarp distribution or see https://www.gnu.org/licenses/lgpl.

## Original MATLAB Implementation

This is a complete rewrite of the original ARTwarp MATLAB software available at:
https://github.com/dolphin-acoustics-vip/artwarp

## Publishing PIP & PyPI

Instructions: 
- https://packaging.python.org/en/latest/tutorials/packaging-projects/
- (Useful source) https://stackoverflow.com/questions/56129825/publishing-modules-to-pip-and-pypi

## Authors

- Python Implementation: Pedro Gronda Garrigues [2026]
- Original MATLAB: Volker Deecke & Vincent Janik [2006]