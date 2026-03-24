# Performance Optimizations (DTW and Unwarp)

This document describes the Numba-based optimizations added to the core DTW and unwarp code so that the Python implementation can match or exceed MATLAB’s speed. It specifies **what** was changed and **how** it was implemented.

---

## 1. Why the Python Version Was Slower

Before optimization, the main bottlenecks were:

1. **No JIT on the hot path** – The DP loops in `dynamic_time_warp` ran as interpreted Python; MATLAB’s engine compiles/JIT-compiles similar loops.
2. **Heavy inner-loop work** – For each DP cell we built lists (`valid_indices`), called `np.array(prev_values)`, and used `np.nanargmax`, causing many small allocations and Python overhead.
3. **Per-sample, per-category DTW** – Training does one DTW per (sample, category) per iteration; slow per-call cost was multiplied by that number.
4. **Only the similarity matrix was vectorized** – The cumulative matrix, path, and constraints were loop-based in Python.
5. **Unwarp** – Used `np.where(warp_function == i)` over the full array for each output index, which is O(m) per index.

---

## 2. What Was Done

### 2.1 Numba JIT for the DTW core

- **Where:** `src/artwarp/core/dtw.py`
- **What:** A separate, Numba-compiled function `_dtw_core_numba(M, m, n, wfl)` was added that performs the entire DP (early stage, general stage, backtrace) and returns `(normalized_similarity, warp_function)`.
- **How:**
  - The function is decorated with `@jit(nopython=True, cache=True)` so it compiles to native code and results are cached on disk.
  - It takes the **precomputed** similarity matrix `M` (from the existing vectorized `compute_similarity_matrix`) and dimensions `m`, `n`, and warp factor level `wfl`.
  - **No list allocations:** Instead of building `valid_indices` and `prev_values` per cell, the inner logic loops over a fixed range `step in range(step_start, wfl + 1)`, computes `j_prev = j - step`, and compares `N[i-1, j_prev]` directly to a running `best_val` / `best_step`. No lists or temporary arrays.
  - **Preallocated arrays:** `N`, `p`, and `k` are created once at the start with `np.full` / `np.zeros` / `np.ones` and filled in place.
  - **Same algorithm:** Early-stage and general-stage loop bounds and the “disqualify vertical step” condition (based on `k[i-1, j] >= wfl` and the `z == 1` case) match the original Python/ MATLAB logic; only the implementation is nopython-friendly.
  - Backtrace is a single loop from `(m-1, n-1)` backward, reading `p[i, j]` and updating `j`; the warping function is written into a preallocated `warp_func` array.
- **Integration:** In `dynamic_time_warp`, after the length-ratio check we compute `M = compute_similarity_matrix(u1, u2)`. If `NUMBA_AVAILABLE` is true, we return `_dtw_core_numba(M, m, n, warp_factor_level)` and never run the Python DP. Otherwise the original Python DP (and short-contour branch, if present) is used.

### 2.2 Numba JIT for unwarp

- **Where:** `src/artwarp/core/dtw.py`
- **What:** A Numba-compiled function `_unwarp_numba(warp_function)` was added that inverts the warping function without using `np.where` over the full array.
- **How:**
  - Decorated with `@jit(nopython=True, cache=True)`.
  - For each output index `i` in `0 .. warp_function[-1]`, a **single pass** over `warp_function` finds the first index where `warp_function[idx] == i`; if none, the value is forward-filled from the previous index. So work per output index is O(m) in the worst case but with tight loops and no Python/NumPy call overhead.
  - The public `unwarp()` checks `NUMBA_AVAILABLE` and, when true, returns `_unwarp_numba(warp_function)`; otherwise it keeps the original implementation using `np.where`.
- **Why it helps:** Unwarp is called inside `weights.update_weights()` for every weight update, so making it nopython reduces overhead in a hot path.

### 2.3 Optional Numba dependency

- **Where:** Top of `src/artwarp/core/dtw.py`
- **What:** `from numba import jit` is wrapped in a try/except. If Numba is missing, `NUMBA_AVAILABLE` is False and a no-op `jit` decorator is used so that code that decorates functions with `@jit(...)` still runs (uncompiled).
- **How:** When Numba is not installed, `dynamic_time_warp` and `unwarp` use the original Python implementations only; no JIT is applied. When Numba is installed, the fast paths above are used automatically.

---

## 3. Summary Table

| Component            | Change                                                                 | Effect                                                                 |
|----------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|
| DTW DP               | New `_dtw_core_numba(M, m, n, wfl)` with `@jit(nopython=True, cache=True)` | DP runs as compiled code; no per-cell list/array allocation           |
| DTW entry point      | `dynamic_time_warp` calls `_dtw_core_numba` when Numba is available    | Hot path uses JIT without changing the public API                      |
| Unwarp               | New `_unwarp_numba(warp_function)` with `@jit(nopython=True, cache=True)`  | Unwarp runs as compiled code; no `np.where` over full array per index  |
| Unwarp entry point   | `unwarp()` calls `_unwarp_numba` when Numba is available               | Same API; fast path when Numba is present                             |
| Similarity matrix    | Unchanged                                                              | Already vectorized; still computed once before calling the Numba core |

---

## 4. What Was Not Changed

- **Number of DTW calls:** Training still does one DTW per (sample, category) per iteration; we did not add batching or reduce calls.
- **Algorithm:** The DP recurrence, Itakura constraints, and backtrace logic are unchanged; only the implementation is JIT-compiled and allocation-free in the inner loops.
- **ART/network layer:** `activate_categories` and the training loop in `network.py` remain in Python; they only call into the now-optimized `dynamic_time_warp` and `unwarp`.
- **Weights update:** The rest of `update_weights` (interpolation, etc.) is still NumPy/Python; only the unwarp call inside it uses the Numba path when available.

---

## 5. How to Get Speedup

- **Install Numba:** e.g. `pip install numba` or use the `[accelerate]` extra if the project defines it. With Numba available, the first run of `dynamic_time_warp` / `_dtw_core_numba` and `unwarp` / `_unwarp_numba` will trigger JIT compilation; subsequent runs use the cached compiled code.
- **Numba check:** When you run the CLI (`artwarp-py` or `./run.sh`), the launcher reports whether Numba is installed and, in an interactive terminal, can offer to install it automatically (pip/conda). See `src/artwarp/utils/numba_check.py` (`report_numba_status`, `check_numba`). The Python API prints a one-line Numba status on `import artwarp`.
- **No API or parameter changes** are required; the same `dynamic_time_warp` and `unwarp` interfaces are used by `art.py` and `weights.py`.

---

## 6. Future Optimisation Opportunities

### 6.1 Multi-core CPU parallelism (recommended!)

The most practical near-term speedup is **parallelising across contours**.  During
each training iteration, the DTW score between the incoming contour and every
existing category prototype is computed sequentially.  These per-category calls
are fully independent and can be distributed across CPU cores.

Suggested approach:

```python
# conceptual sketch (!!! not yet implemented)
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
    scores = list(pool.map(dtw_vs_prototype, contours))
```

`joblib` (already available via scikit-learn) is another drop-in option:

```python
from joblib import Parallel, delayed
scores = Parallel(n_jobs=-1)(delayed(dynamic_time_warp)(c, proto) for proto in prototypes)
```

Expected benefit: **near-linear speedup** with the number of physical cores, with
no algorithm changes.  This is the highest-value, lowest-risk extension.

> **Implementation note:** Numba-JIT functions cannot be pickled by the default
> Python `multiprocessing` spawn/fork mechanism.  The workaround is to
> serialise the *inputs* (NumPy arrays), not the compiled function, and call the
> Numba function inside each worker -> Numba re-uses the cached AOT binary per
> process.

---

### 6.2 GPU acceleration (advanced, non-trivial)

The DTW dynamic-programming recurrence is inherently sequential, each cell
depends on the previous row — so a single DTW computation **cannot** be
straightforwardly parallelised on a GPU.  However, two GPU-applicable patterns
exist:

#### Anti-diagonal wavefront parallelism (intra-DTW)

Cells on the same anti-diagonal of the DP matrix are mutually independent.  A
GPU kernel can compute all cells of diagonal `d` simultaneously before advancing
to `d+1`.  This gives limited intra-DTW parallelism at the cost of more complex
memory access patterns and kernel launch overhead.

**When it helps:** only when contours are long enough (roughly > 1 000 points)
that the diagonal width justifies the overhead.  For typical bioacoustic contours
(100–600 points) the gain is likely marginal.

#### Batch-DTW across category prototypes (inter-DTW)

Computing DTW between one query contour and all `N` category prototypes is
embarrassingly parallel (seriously...).  A single GPU dispatch could evaluate all `N`
comparisons simultaneously.

Relevant libraries:

| Library | Notes |
|---|---|
| [`cuML` / RAPIDS](https://rapids.ai/) | GPU-accelerated time-series distances; requires CUDA |
| [PyTorch soft-DTW](https://github.com/Sleepwalking/pytorch-softdtw) | Differentiable DTW on CUDA; not identical to hard-DTW |
| Custom CUDA / Triton kernel | Full control; significant engineering effort |

**Practical trade-offs for this project:**

| Factor | Impact |
|---|---|
| Category count (100–200) | Small `N` → limited parallelism; GPU underutilised |
| Contour length (100–600 pts) | Short sequences → kernel launch overhead dominates |
| Itakura constraint | Irregular DP band complicates GPU memory access |
| Engineering cost | Substantial rewrite of `dtw.py` |
| MATLAB parity requirement | Any GPU variant must produce bit-identical results |

**Recommendation:** Multi-core CPU parallelism (§ 6.1) should be implemented
first.  GPU acceleration is worth revisiting only if dataset sizes grow
substantially (e.g., > 10 000 contours, > 500 categories, or contours > 1 000
points).

---

## 7. References

- **Core implementation:** `src/artwarp/core/dtw.py` (functions `_dtw_core_numba`, `_unwarp_numba`, and the `NUMBA_AVAILABLE` branch in `dynamic_time_warp` and `unwarp`).
- **Numba availability and install prompt:** `src/artwarp/utils/numba_check.py` (`numba_available`, `report_numba_status`, `check_numba`); used by the CLI and on package import.
- **MATLAB alignment:** `docs/CORE_MATLAB_VERIFICATION.md` (confirms the DP and unwarp logic match the MATLAB core).
- **Parallelism (§ 6.1):** [`joblib` docs](https://joblib.readthedocs.io/en/stable/), Python [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html).
- **GPU DTW (§ 6.2):** [RAPIDS cuML](https://rapids.ai/), [pytorch-softdtw](https://github.com/Sleepwalking/pytorch-softdtw), [Anti-diagonal GPU DTW (Salvador & Chan, 2007)](https://doi.org/10.1007/s10115-006-0030-0).
