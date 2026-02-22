# ARTwarp-py INTERNAL Documentation

This directory contains detailed internal documentation for end-users, developers, and maintainers.

## Contents

### Test Documentation
- **docs/dev/TEST_RESULTS.md** - Complete test report with all X tests documented
  - Test execution results
  - Bug fixes applied during testing
  - Performance benchmarks
  - Compatibility matrix

### Project Documentation
- **docs/dev/PROJECT_SUMMARY.md** - Rigorous project overview
  - Complete code statistics
  - Feature comparison with MATLAB
  - Translation details
  - Performance analysis (TODO)

- **docs/dev/PERFORMANCE_OPTIMIZATIONS.md** - Numba-based DTW and unwarp optimizations
  - Why the Python version was slower (no JIT, inner-loop allocation, etc.)
  - What was done: `_dtw_core_numba`, `_unwarp_numba`, integration in `dynamic_time_warp` and `unwarp`
  - How it was implemented (nopython, preallocated arrays, fixed-size loops, no list building)
  - What was not changed (algorithm, number of DTW calls, ART/network layer)
  - How to get the speedup (install Numba)

### Environment Setup
- **docs/dev/ENVIRONMENT_SETUP.md** - Detailed env configuration guide
  - Both venv and conda instructions
  - Troubleshooting guide
  - Tested configurations
  - Dependency details

## For End Users [!!!]

If you're looking for user-facing documentation, see the docs/user/ directory:

- **README.md**                    - Project overview and quick start
- **CHANGELOG.md**                 - Version history
- **docs/user/INSTALLATION.md**    - Installation instructions
- **docs/user/API.md**             - Complete API reference
- **docs/user/VISUALIZATION.md**   - Visualization guide
- **docs/user/QUICK_REFERENCE.md** - Quick reference cheat sheet
- **docs/user/ARCHITECTURE.md**    - System architecture

## For Developers [!!!]

This `docs/dev/` directory contains:

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
├── CHANGELOG.md                       # version history
├── run.sh                             # interactive launcher (Train / Plot / Predict / Export)
├── docs/                              # internal documentation (this dir)
│   ├── README.md                          # this file
│   ├── dev/ 
│   │   ├── ENVIRONMENT_SETUP.md           # detailed environment guide
│   │   ├── TEST_RESULTS.md                # test documentation
│   │   ├── PROJECT_SUMMARY.md             # complete project summary
│   │   └── PERFORMANCE_OPTIMIZATIONS.md   # optimizations for JIT compiler
│   └── user/
│       ├── INSTALLATION.md                # setup instructions
│       ├── API.md                         # API reference
│       ├── VISUALIZATION.md               # user visualization guide
│       ├── QUICK_REFERENCE.md             # cheat sheet
│       └── ARCHITECTURE.md                # system design
├── src/                               # source code
├── tests/                             # test suite
├── img/                               # README banner
└── examples/                          # usage examples
```

## Usage

These documents are primarily for:
- Developers contributing to the project
- Maintainers reviewing code quality
- Users interested in implementation details
- Researchers evaluating the translation from MATLAB

Most users should start with the main `README.md` in the parent directory and `docs/user/QUICK_REFERENCE.md` in the `docs/user` directory.

Please read all the docs corresponding to either end-user (`docs/user/`) or developers (`docs/dev/`), or both. Please :)

## GitHub Wiki

The **GitHub Wiki** for this repo is synced from this `docs/` directory. When you push changes under `docs/**` to `main`, the workflow `.github/workflows/sync-wiki.yml` copies content into the wiki:

| In this repo | Wiki page |
|--------------|-----------|
| `docs/README.md` | **Home** |
| `docs/user/ARCHITECTURE.md` | User-ARCHITECTURE |
| `docs/user/INSTALLATION.md` | User-INSTALLATION |
| `docs/user/API.md` | User-API |
| `docs/user/VISUALIZATION.md` | User-VISUALIZATION |
| `docs/user/QUICK_REFERENCE.md` | User-QUICK_REFERENCE |
| `docs/dev/PROJECT_SUMMARY.md` | Dev-PROJECT_SUMMARY |
| `docs/dev/ENVIRONMENT_SETUP.md` | Dev-ENVIRONMENT_SETUP |
| `docs/dev/TEST_RESULTS.md` | Dev-TEST_RESULTS |
| `docs/dev/PERFORMANCE_OPTIMIZATIONS.md` | Dev-PERFORMANCE_OPTIMIZATIONS |

---

@author: Pedro Gronda Garrigues
         @PedroGGBM (GitHub)
