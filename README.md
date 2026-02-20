<p align="center">
  <img src="img/artwarp_py_banner.png" alt="ARTwarp-py" width="100%"/>
</p>

# ARTwarp-py

**A high-performance Python implementation of ARTwarp for automated categorization of tonal animal sounds.**

## Overview

ARTwarp-py is a complete rewrite of the original MATLAB ARTwarp software, combining **Dynamic Time Warping (DTW)** for contour similarity with an **Adaptive Resonance Theory (ART)** neural network for unsupervised clustering, with performance improvements while staying mathematically equivalent to the original algorithm.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Architecture](#architecture)
- [Algorithm Details](#algorithm-details)
- [MATLAB Compatibility](#matlab-compatibility)
- [Testing](#testing)
- [Citation](#citation)
- [License & Authors](#license--authors)

---

## Features

| Area | Description |
|------|-------------|
| **Performance** | Optimized algorithms using NumPy, with optional Numba JIT compilation |
| **Input formats** | `.ctr` (MATLAB), `.csv`, `.txt`; load MATLAB categorisation (`.mat`) via `load_mat_categorisation()` |
| **Visualization** | Publication-worthy plots and reports |
| **CLI** | Command-line interface for training, plotting, and batch workflows |
| **Quality** | Type hints, tests, coverage, and documentation |
| **Future** | Parallel processing and batch pipelines (TODO) |

---

## Installation

**Requirements:** Python 3.8+, NumPy ≥1.20, SciPy ≥1.7, pandas ≥1.3, matplotlib ≥3.4, and optionally Numba ≥0.54 for JIT.

```bash
pip install -e .
```

For environment setup and virtualenv details, see **[user/INSTALLATION.md](docs/user/INSTALLATION.md)** (end-user) and **[dev/ENVIRONMENT_SETUP.md](docs/dev/ENVIRONMENT_SETUP.md)** (developers) within **[docs/](docs/)**.

---

## Quick Start

See **[docs/user/QUICK_REFERENCE.md](docs/user/QUICK_REFERENCE.md)** for a condensed cheat sheet.

### Command line

Activate your virtual environment first (e.g. `source venv/bin/activate` or, in Fish, `source venv/bin/activate.fish`), then:

Basic commands:

```bash
# train on a directory of contour files
artwarp-py train --input-dir ./contours --output results.pkl \
    --vigilance 85 --learning-rate 0.1 --max-iterations 50

# export reference contours and category assignments
artwarp-py train --input-dir ./contours --output results.pkl --export-refs --export-categories
```

Resample with sample interval (seconds) [default = 0.02s]:


```bash
# resample contours to uniform temporal resolution (like MATLAB resample option)
artwarp-py train --input-dir ./contours --output results.pkl --resample --sample-interval 0.02
```

Altogether (resampling & exporting reference contours / category assignments):

```bash
# full command
artwarp-py train --input-dir ./contours --output results.pkl --resample --sample-interval 0.02 --vigilance 85 --learning-rate 0.1 --max-iterations 50 --export-refs --export-categories
```

Gener visualizations:

```bash
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

---

## Documentation

| Document | Description |
|----------|-------------|
| [**docs/user/QUICK_REFERENCE.md**](docs/user/QUICK_REFERENCE.md) | (User Guide) Short reference for CLI and common tasks |
| [**docs/user/INSTALLATION.md**](docs/user/INSTALLATION.md) | (User Guide) Install and environment setup |
| [**docs/user/API.md**](docs/user/API.md) | (User Guide) Public API (loaders, exporters, options) |
| [**docs/user/ARCHITECTURE.md**](docs/user/ARCHITECTURE.md) | (User Guide) Code layout and design |
| [**docs/user/VISUALIZATION.md**](docs/user/VISUALIZATION.md) | (User Guide) Plotting and report generation |
| [**CHANGELOG.md**](CHANGELOG.md) | Version history |
| [**docs/README.md**](docs/README.md) | Docs index and overview |
| [**docs/dev/ENVIRONMENT_SETUP.md**](docs/dev/ENVIRONMENT_SETUP.md) | (Developers) Detailed environment and tooling |
| [**docs/dev/PROJECT_SUMMARY.md**](docs/dev/PROJECT_SUMMARY.md) | (Developers) Project summary and goals |
| [**docs/dev/PERFORMANCE_OPTIMIZATIONS.md**](docs/dev/PERFORMANCE_OPTIMIZATIONS.md) | (Developers) Performance notes and benchmarks |
| [**docs/dev/TEST_RESULTS.md**](docs/dev/TEST_RESULTS.md) | (Developers) CI, test matrix, and coverage |

---

## Architecture

```
artwarp/
├── core/
│   ├── dtw.py          # Dynamic Time Warping
│   ├── art.py          # ART neural network
│   ├── network.py      # Main ARTwarp algorithm
│   └── weights.py      # Weight update and management
├── io/
│   ├── loaders.py      # Data loading (.ctr, .csv, .txt, .mat)
│   └── exporters.py    # Results export
├── visualization/
│   └── plotting.py     # Plotting and reports
├── utils/
│   ├── validation.py   # Input validation
│   └── resample.py     # Contour resampling (MATLAB-aligned)
└── cli/
    └── main.py         # Command-line interface
```

---

## Algorithm Details

### Dynamic Time Warping

- Optimized DP with Itakura parallelogram constraints  
- Vectorized similarity matrix and efficient path backtracing  
- Optional Numba JIT for speed

### ART neural network

- Bottom-up activation (similarity-based) and top-down matching (vigilance)  
- Dynamic category creation and incremental weight updates  

---

## MATLAB compatibility

- **Load MATLAB results:**  
  `from artwarp.io import load_mat_categorisation`  
  `data = load_mat_categorisation("ARTwarp85FINAL.mat")`  
  → `weight_matrix`, `num_categories`, and optionally `contours` / `categories` / `contour_names`.
- **Export for MATLAB:** Use `one_based_categories=True` or `one_based_filenames=True` (see [API.md](docs/user/API.md)).

---

## Testing

Run the test suite (use the venv’s Python so dependencies are found):

```bash
python -m pytest tests/ -v
```

With coverage (CI requires ≥80%):

```bash
python -m pytest tests/ --cov=artwarp --cov-report=html --cov-fail-under=80
```

**CI:** GitHub Actions runs tests (Python 3.9–3.12), coverage gate, and lint (Black, isort, Flake8, Mypy). See [docs/dev/TEST_RESULTS.md](docs/dev/TEST_RESULTS.md) for details.

---

## Citation

If you use this software in your research, please cite:

1. **This implementation:** [DOI to be assigned]
2. **Buck, J. R. & Tyack, P. L. (1993).** A quantitative measure of similarity for *Tursiops truncatus* signature whistles. *J. Acoust. Soc. Am.* **94**, 2497–2506. https://doi.org/10.1121/1.407385
3. **Deecke, V. B., Ford, J. K. B. & Spong, P. (1999).** Quantifying complex patterns of bioacoustic variation: Use of a neural network to compare killer whale (*Orcinus orca*) dialects. *J. Acoust. Soc. Am.* **105**, 2499–2507. https://doi.org/10.1121/1.426853
4. **Deecke, V. B. & Janik, V. M. (2006).** Automated categorization of bioacoustic signals: Avoiding perceptual pitfalls. *J. Acoust. Soc. Am.* **119**, 645–653. https://doi.org/10.1121/1.2139067

---

## License & Authors

ARTwarp is distributed under the **GNU Lesser General Public License v3**. See [LICENSE](LICENSE) or https://www.gnu.org/licenses/lgpl.

- **Python implementation:** Pedro Gronda Garrigues (2026)  
- **Original MATLAB:** Volker Deecke & Vincent Janik (2006)  
- **Original MATLAB repo:** https://github.com/dolphin-acoustics-vip/artwarp  

ARTwarp is distributed under the terms of the GNU Lesser General Public
License, version 3, as published by the Free Software Foundation. For
details, please refer to the LICENSE file in the root directory of the
ARTwarp distribution or see https://www.gnu.org/licenses/lgpl.

## Original MATLAB Implementation

This is a complete rewrite of the original ARTwarp MATLAB software available at:
https://github.com/dolphin-acoustics-vip/artwarp

**Publishing (PyPI/pip):** [Python packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) · [Stack Overflow: publishing to PyPI](https://stackoverflow.com/questions/56129825/publishing-modules-to-pip-and-pypi)

The following are steps (specific to @PedroGGBM) to upload the PyPI package:

```bash
# to generate distribution archives (build/)
conda activate sig-process # or your corresponding venv (e.g., conda, venv, uv, etc)
cd <base_directory>
python -m pip install --upgrade setuptools wheel build # ensure latest version of PyPA's build installed
python -m build

# to upload distribution archive
python -m pip install --upgrade twine
python -m twine upload --repository testpypi dist/* # this is to test it
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-package-YOUR-USERNAME-HERE # to test install on test server for PyPI

### !!! Once registered in PyPI and own official API key
python -m twine upload dist/* # this is to upload it
```
