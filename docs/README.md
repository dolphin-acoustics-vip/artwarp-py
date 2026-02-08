# ARTwarp-py INTERNAL Documentation

This directory contains detailed internal documentation for developers and maintainers.

## Contents

### Test Documentation
- **TEST_RESULTS.md** - Complete test report with all X tests documented
  - Test execution results
  - Bug fixes applied during testing
  - Performance benchmarks
  - Compatibility matrix

### Project Documentation
- **PROJECT_SUMMARY.md** - Rigorous project overview
  - Complete code statistics
  - Feature comparison with MATLAB
  - Translation details
  - Performance analysis (TODO)

- **PERFORMANCE_OPTIMIZATIONS.md** - Numba-based DTW and unwarp optimizations
  - Why the Python version was slower (no JIT, inner-loop allocation, etc.)
  - What was done: `_dtw_core_numba`, `_unwarp_numba`, integration in `dynamic_time_warp` and `unwarp`
  - How it was implemented (nopython, preallocated arrays, fixed-size loops, no list building)
  - What was not changed (algorithm, number of DTW calls, ART/network layer)
  - How to get the speedup (install Numba)

### Environment Setup
- **ENVIRONMENT_SETUP.md** - Detailed env configuration guide
  - Both venv and conda instructions
  - Troubleshooting guide
  - Tested configurations
  - Dependency details

## For End Users [!!!]

If you're looking for user-facing documentation, see the main directory:

- **README.md**          - Project overview and quick start
- **INSTALLATION.md**    - Installation instructions
- **API.md**             - Complete API reference
- **VISUALIZATION.md**   - Visualization guide
- **QUICK_REFERENCE.md** - Quick reference cheat sheet
- **ARCHITECTURE.md**    - System architecture
- **CHANGELOG.md**       - Version history

## For Developers [!!!]

This `docs/` directory contains:

1. **Testing Information**
   - Test results
   - Test coverage details (coverage omits `__init__.py` and tests; see `pyproject.toml`)
   - Bug fix documentation

2. **Project Metrics**
   - Lines of code statistics
   - Module breakdown
   - Performance comparisons (TODO)

3. **Setup Details**
   - Detailed environment configuration
   - Multiple installation methods
   - Platform-specific notes

## Directory Structure

```
artwarp-py/
├── README.md                          # main project documentation
├── INSTALLATION.md                    # setup instructions
├── API.md                             # API reference
├── VISUALIZATION.md                   # user visualization guide
├── QUICK_REFERENCE.md                 # cheat sheet
├── ARCHITECTURE.md                    # system design
├── CHANGELOG.md                       # version history
├── docs/                              # internal documentation (this dir)
│   ├── README.md                      # this file
│   ├── TEST_RESULTS.md                # test documentation
│   ├── PROJECT_SUMMARY.md             # complete project summary
│   ├── PERFORMANCE_OPTIMIZATIONS.md   # optimizations for JIT compiler
│   └── ENVIRONMENT_SETUP.md           # detailed environment guide
├── src/                               # source code
├── tests/                             # test suite
└── examples/                          # usage examples
```

## Usage

These documents are primarily for:
- Developers contributing to the project
- Maintainers reviewing code quality
- Users interested in implementation details
- Researchers evaluating the translation from MATLAB

Most users should start with the main `README.md` and `QUICK_REFERENCE.md` in the parent directory.

Please read all the docs corresponding to either end-user or developers (or both). Please.

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)
