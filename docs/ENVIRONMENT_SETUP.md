# Environment Setup Guide

## Python Virtual Environments for ARTwarp-py

This guide shows how to set up both **venv** (standard Python virtual environment) and **conda** environments for ARTwarp-py.

I would personally recommend using conda (i.e. miniconda3) for better cross-package version support.

---

## Option 1: venv (Standard Python)

### Create and Activate venv

```bash
cd artwarp-py

# create virtual env
python3 -m venv venv_artwarp

# activate on Linux/macOS
source venv_artwarp/bin/activate       # bash shell
source venv_artwarp/bin/activate.fish  # fish shell

# activate on Windows
venv_artwarp\Scripts\activate
```

### Install Dependencies

```bash
# upgrade pip
pip install --upgrade pip

# install artwarp-py with all dependencies (including dev tools)
pip install -e .[dev]

# or install just the runtime dependencies
pip install -e .
```

### Verify Installation

```bash
# check installation
python -c "import artwarp; print(artwarp.__version__)"

# run tests
python -m pytest tests/ -v

# run example
python examples/simple_example.py
```

### Deactivate

```bash
deactivate
```

---

## Option 2: conda Environment

### Create and Activate Conda Environment

```bash
cd artwarp-py

# create conda environment with Python 3.11 (or 3.9, 3.10)
conda create -n artwarp python=3.11 -y

# activate environment
conda activate artwarp
```

### Install Dependencies

```bash
# install artwarp-py with dependencies
pip install -e .[dev]

# alternative: Install dependencies via conda first (optional)
conda install numpy scipy pandas matplotlib -y
pip install -e .
```

### Verify Installation

```bash
# check installation
python -c "import artwarp; print(artwarp.__version__)"

# run tests
python -m pytest tests/ -v
```

### Deactivate

```bash
conda deactivate
```

---

## Dependency Overview

### Core Dependencies (Required)
- **numpy** >= 1.20.0 - Numerical computing
- **scipy** >= 1.7.0 - Scientific computing (for .mat file loading)
- **pandas** >= 1.3.0 - Data manipulation (for CSV/TXT loading)
- **matplotlib** >= 3.4.0 - Visualization

### Optional Dependencies (Recommended)
- **numba** >= 0.54.0 - JIT compilation for performance acceleration

### Development Dependencies
- **pytest** >= 7.0.0 - Testing framework
- **pytest-cov** >= 3.0.0 - Code coverage
- **black** >= 22.0.0 - Code formatter
- **mypy** >= 0.950 - Type checking
- **flake8** >= 4.0.0 - Linting
- **isort** >= 5.10.0 - Import sorting

---

## Installation Methods

### Method 1: Editable Install (Development)

```bash
# install in editable mode with dev dependencies
pip install -e .[dev]
```

**Benefits:**
- Changes to source code are immediately reflected
- Includes all development tools

### Method 2: Regular Install (Production)

```bash
# install normally
pip install .
```

**Benefits:**
- Standard installation
- Smaller footprint (no dev dependencies)
- Suitable for production use

### Method 3: Install from Requirements File

```bash
# install from requirements.txt
pip install -r requirements.txt

# then install package
pip install -e .
```

---

## Verifying Your Setup

### 1. Check Python Version

```bash
python --version
# should be Python 3.8 or higher
```

### 2. Check Installed Packages

```bash
pip list | grep -E "numpy|scipy|pandas|matplotlib|artwarp"
```

### 3. Run Quick Test

```bash
python -c "
from artwarp import ARTwarp, load_contours
import numpy as np
print('ARTwarp successfully imported!')

# Quick functionality test
contours = [np.array([100.0, 200.0, 300.0]) for _ in range(3)]
network = ARTwarp(vigilance=85.0, verbose=False)
results = network.fit(contours)
print(f'Test passed! Created {results.num_categories} category(ies)')
"
```

### 4. Run Full Test Suite

```bash
python -m pytest tests/ -v
# Should show X passed tests
```

### 5. Run Example Scripts

```bash
# basic example
python examples/simple_example.py

# visualization example (requires matplotlib)
python examples/visualization_example.py
```

---

## Troubleshooting

### Issue: "pip: command not found"

**Solution:**
```bash
# on Linux/macOS
python3 -m pip install --upgrade pip

# on Windows
python -m pip install --upgrade pip
```

### Issue: "Module 'numpy' not found"

**Solution:**
```bash
pip install numpy scipy pandas matplotlib
```

### Issue: "pytest: command not found"

**Solution:**
```bash
pip install pytest
# or use: python -m pytest tests/
```

### Issue: matplotlib backend errors

**Solution:**
```bash
# for headless servers, set non-interactive backend
export MPLBACKEND=Agg
python examples/visualization_example.py
```

### Issue: Permission denied during installation

This would occur if, say, you're on CSC PC Lab computers

**Solution:**
```bash
# use --user flag for user-level install
pip install --user -e .

# or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Issue: Conda environment conflicts

**Solution:**
```bash
# remove and recreate environment
conda remove -n artwarp --all -y
conda create -n artwarp python=3.11 -y
conda activate artwarp
pip install -e .[dev]
```

---

## Environment Information

### Tested Configurations

| Platform | Python | Status |
|----------|--------|--------|
| Linux (Arch) | 3.13.11 | ✅ All X tests passing |
| Linux | 3.11.x | ✅ Expected to work |
| Linux | 3.10.x | ✅ Expected to work |
| Linux | 3.9.x | ✅ Expected to work |
| macOS | 3.9+ | ⚠️ Not tested (should work) |
| Windows | 3.9+ | ⚠️ Not tested (should work) |

### Current Test Environment (Verified)

```
OS: Linux 6.18.5-arch1-1
Python: 3.13.11
Conda: 25.11.0
Virtual Environment: venv_artwarp
Test Results: 79/79 passed (100%) (as of 28/02/2026)
```

---

## Quick Start Summary

### For Development:

```bash
# clone/navigate to repo
cd artwarp-py

# create venv
python3 -m venv venv_artwarp
source venv_artwarp/bin/activate

# install with dev dependencies
pip install -e .[dev]

# verify
python -m pytest tests/ -v
```

### For Production Use:

```bash
# create venv
python3 -m venv venv
source venv/bin/activate

# install
pip install .

# verify
python -c "import artwarp; print('OK')"
```

---

## Additional Resources

- **Installation Guide**: See `INSTALLATION.md` for detailed instructions
- **API Documentation**: See `API.md` for complete API reference
- **Visualization Guide**: See `VISUALIZATION.md` for plotting instructions
- **Examples**: Check `examples/` directory for usage examples
- **Tests**: Review `tests/unit/` for test examples

---

## Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Verify Python version (3.8+)
3. Try creating a fresh virtual environment
4. Ensure all dependencies are installed
5. Run the test suite to verify installation

For further assistance, consult GitHub Issues, or shoot me a message -> pgg6@st-andrews.ac.uk (university) / pgrondagarrigues@gmail.com (personal)

---

_Last Updated: January 29, 2026_  
_Tested Environment: Linux 6.18.5, Python 3.13.11_

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)