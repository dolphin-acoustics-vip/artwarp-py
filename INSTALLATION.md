# ARTwarp-py Installation Guide

This is a detailed guide, which serves as a simplification of the developer docs/ENVIRONMENT_SETUP.md

If you are a developer, please read all of docs/!!! 

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Options

### Option 1: Install from Source

1. Clone the repository:
```bash
git clone https://github.com/dolphin-acoustics-vip/artwarp-py.git
cd artwarp-py
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e .
```

This installs ARTwarp-py in "editable" mode, meaning changes to the source code
are immediately reflected without reinstalling.

### Option 2: Install with Optional Dependencies

For best performance, install with acceleration support:
```bash
pip install -e ".[accelerate]"
```

For visualization capabilities:
```bash
pip install -e ".[viz]"
```

For development tools:
```bash
pip install -e ".[dev]"
```

Install everything:
```bash
pip install -e ".[accelerate,viz,dev]"
```

### Option 3: Install from PyPI (when available)

```bash
pip install artwarp-py
```

## Environment Management

### Using venv (Recommended)

For better isolation, use a virtual environment (as seen above):

```bash
# create and activate
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# install
pip install -e .[dev]

# deactivate when done
deactivate
```

### Using conda

Alternatively, use conda:

```bash
# create and activate
conda create -n artwarp python=3.11 -y
conda activate artwarp

# install
pip install -e .[dev]

# deactivate when done
conda deactivate
```

## Verify Installation

### Quick Test

```bash
# check version
python3 -c "import artwarp; print(f'ARTwarp v{artwarp.__version__} installed!')"

# quick functionality test
python3 -c "
from artwarp import ARTwarp
import numpy as np
contours = [np.array([100.0, 200.0, 300.0]) for _ in range(3)]
network = ARTwarp(vigilance=85.0, verbose=False)
results = network.fit(contours)
print(f'Test passed! Created {results.num_categories} category(ies)')
"
```

### Run Command-Line Tool

```bash
artwarp-py --help
```

## Running Tests

If you installed with dev dependencies, run tests with the **same Python as your venv** so that numpy and other deps are found:

```bash
# recommended: use the venv's Python to run pytest (avoids "no numpy" errors)
python -m pytest tests/ -v
```

If you type `pytest tests/ -v` and get `ModuleNotFoundError: No module named 'numpy'`, see the troubleshooting section below.

## Running Examples

```bash
python3 examples/simple_example.py
python3 examples/visualization_example.py
```

## Troubleshooting

### pytest says "No module named 'numpy'" (but the venv is active)

Your shell is likely running a **different** `pytest` (e.g. system or another env), which uses a Python that doesn't have the venv's packages.

**Fix:** run pytest via the venv's Python so the same interpreter that has numpy is used:

```bash
python -m pytest tests/ -v
```

**Check which Python is used:**

```bash
which python    # should be something like .../artwarp-py/.venv/bin/python
which pytest    # should be inside the same .venv/bin if you want to use bare "pytest"
pip list        # should show numpy, scipy, artwarp, etc.
```

If `which pytest` is outside your project (e.g. `/usr/bin/pytest`), use `python -m pytest` every time, or run `pip install -e ".[dev]"` in the venv so the venv's `pytest` is used when you type `pytest`!

This will avoid any annoying dependency error messages.

### ImportError for NumPy/SciPy

If you get import errors for NumPy or SciPy, make sure they are installed **in the active environment**:

```bash
pip install numpy scipy pandas
# or reinstall the project: pip install -e .
```

### Numba Installation Issues

Numba (optional acceleration) may have installation issues on some systems.
If you encounter problems, ARTwarp-py will work fine without it, just slower:

```bash
pip install -e .  # without accelerate extras
```

### Permission Errors

If you get permission errors during installation:

1. Use a virtual environment (recommended)
2. Or install for your user only: `pip install --user -e .`

## System Requirements

### Minimum Requirements

- RAM: 2GB (for small datasets <100 contours)
- CPU: Any modern processor
- Storage: 100MB for code and dependencies

### Recommended for Large Datasets

- RAM: 8GB+ (for datasets with 1000+ contours)
- CPU: Multi-core processor for parallel operations
- Storage: Additional space for results and cache

## Dependencies

Core dependencies (automatically installed):
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0 (for visualization)

Optional dependencies:
- numba >= 0.54.0 (for JIT acceleration)

Development dependencies:
- pytest >= 7.0.0
- pytest-cov >= 3.0.0
- black, mypy, flake8, isort (code quality tools)

## Next Steps

After installation:

1. **Quick Start**: See `QUICK_REFERENCE.md` for common usage patterns
2. **API Documentation**: See `API.md` for complete API reference
3. **Visualization**: See `VISUALIZATION.md` for plotting guide
4. **Examples**: Check `examples/` directory for complete demos
5. **Architecture**: See `ARCHITECTURE.md` for design details
6. **Detailed Docs**: See `docs/` directory for additional documentation (for developers)

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)