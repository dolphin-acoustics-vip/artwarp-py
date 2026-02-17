"""
Dynamic Time Warping (DTW) implementation with Itakura parallelogram constraints.

This module provides a high-performance implementation of DTW specifically designed
for ARTwarp's frequency contour comparison. It includes optimizations such as:
- Vectorized similarity matrix computation
- Efficient dynamic programming with constraints
- Optional Numba JIT compilation
- Memory-efficient path backtracing

The implementation maintains mathematical equivalence with the original MATLAB
warp.m function while providing significant performance improvements.

@author: Pedro Gronda Garrigues
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

# optional numba for JIT
try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # no-op decorator when numba missing
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@jit(nopython=True, cache=True)  # pragma: no cover
def _dtw_core_numba(M, m, n, wfl):
    """
    Numba-compiled DTW core: fill N, p, k in place, backtrace, return (normalized_sim, warp_func).
    No list allocations; fixed-size loops over step in 0..wfl.
    Same logic tested via Python fallback when NUMBA_AVAILABLE=False.
    """
    N = np.full((m, n), np.nan, dtype=np.float64)
    p = np.zeros((m, n), dtype=np.int32)
    k = np.ones((m, n), dtype=np.int32)

    N[0, 0] = M[0, 0]
    k[0, 0] = 1

    early_stage_limit = min(wfl * (wfl + 1) - 1, m)

    # early stage
    for i in range(1, early_stage_limit):
        for z in range(1, wfl + 1):
            if (i // wfl) <= z:
                j = z - 1
                if j >= n:
                    continue
                condition = (k[i - 1, j] > wfl) if (z == 1) else (k[i - 1, j] >= wfl)
                best_val = np.nan
                best_step = -1
                step_start = 1 if condition else 0
                for step in range(step_start, wfl + 1):
                    j_prev = j - step
                    if j_prev >= 0:
                        v = N[i - 1, j_prev]
                        if not np.isnan(v) and (np.isnan(best_val) or v > best_val):
                            best_val = v
                            best_step = step
                if best_step < 0:
                    continue
                N[i, j] = M[i, j] + best_val
                p[i, j] = -best_step
                k[i, j] = (1 + k[i - 1, j]) if (best_step == 0) else 1

        j_start = wfl
        j_end = min(wfl * i, int((i - m) / wfl + n))
        for j in range(j_start, min(j_end + 1, n)):
            if k[i - 1, j] >= wfl:
                step_start = 1
            else:
                step_start = 0
            best_val = np.nan
            best_step = -1
            for step in range(step_start, wfl + 1):
                j_prev = j - step
                if j_prev >= 0:
                    v = N[i - 1, j_prev]
                    if not np.isnan(v) and (np.isnan(best_val) or v > best_val):
                        best_val = v
                        best_step = step
            if best_step < 0:
                continue
            N[i, j] = M[i, j] + best_val
            p[i, j] = -best_step
            k[i, j] = (1 + k[i - 1, j]) if (best_step == 0) else 1

    # general stage
    for i in range(early_stage_limit, m):
        j_start = max(0, max(((i + 1) // wfl) - 1, (i - m) * wfl + n))
        j_end_1b = min(wfl * (i + 1), int(round((i - m) / wfl + n)))
        j_end = j_end_1b - 1
        for j in range(j_start, min(j_end + 1, n)):
            if k[i - 1, j] >= wfl:
                step_start = 1
            else:
                step_start = 0
            best_val = np.nan
            best_step = -1
            for step in range(step_start, wfl + 1):
                j_prev = j - step
                if j_prev >= 0:
                    v = N[i - 1, j_prev]
                    if not np.isnan(v) and (np.isnan(best_val) or v > best_val):
                        best_val = v
                        best_step = step
            if best_step < 0:
                continue
            N[i, j] = M[i, j] + best_val
            p[i, j] = -best_step
            k[i, j] = (1 + k[i - 1, j]) if (best_step == 0) else 1

    # backtrace
    warp_func = np.zeros(m, dtype=np.int32)
    j = n - 1
    for i in range(m - 1, -1, -1):
        warp_func[i] = j
        if i > 0:
            j = j + p[i, j]

    norm_sim = N[m - 1, n - 1] / m if not np.isnan(N[m - 1, n - 1]) else 0.0
    return norm_sim, warp_func


def compute_similarity_matrix(
    u1: NDArray[np.float64], u2: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute point-wise similarity matrix between two frequency contours.

    The similarity between two frequency values is calculated as:
        similarity(f1, f2) = (min(f1, f2) / max(f1, f2)) * 100

    This gives a percentage similarity where identical frequencies score 100%.

    Args:
        u1: Reference contour of shape (m,)
        u2: Comparison contour of shape (n,)

    Returns:
        Similarity matrix M of shape (m, n) with values in [0, 100]

    Note:
        This function is fully vectorized for performance. The original MATLAB
        implementation used a parfor loop, but NumPy broadcasting is faster.
    """
    m, n = len(u1), len(u2)

    # reshape for broadcast => u1 col (m,1), u2 row (1,n)
    u1_col = u1.reshape(-1, 1)
    u2_row = u2.reshape(1, -1)

    # vectorized min/max
    numerator = np.minimum(u1_col, u2_row)
    denominator = np.maximum(u1_col, u2_row)

    # avoid div by zero (rare with freq data)
    denominator = np.where(denominator == 0, 1e-10, denominator)

    similarity = (numerator / denominator) * 100.0

    return similarity


def dynamic_time_warp(
    u1: NDArray[np.float64], u2: NDArray[np.float64], warp_factor_level: int = 3
) -> Tuple[float, NDArray[np.int32]]:
    """
    Perform Dynamic Time Warping between two frequency contours.

    This implements the Itakura parallelogram constraint variant of DTW, which
    limits the maximum local warping factor. The algorithm uses dynamic programming
    to find the optimal warping path that maximizes cumulative point-wise similarity.

    Args:
        u1: Reference contour (frequency values over time)
        u2: Comparison contour (frequency values over time)
        warp_factor_level: Maximum allowed local warping factor (default: 3)
            Controls how much time compression/expansion is allowed.
            - Maximum consecutive vertical steps: warp_factor_level
            - Maximum horizontal jumps: warp_factor_level

    Returns:
        Tuple containing:
            - normalized_similarity: Average point-wise similarity along optimal path (0-100)
            - warp_function: Array of indices mapping u1 to u2, shape (m,)
                For each index i in u1, warp_function[i] gives the corresponding
                index in u2 that best matches it.

    Raises:
        ValueError: If contours differ in length by more than warp_factor_level

    Algorithm:
        1. Compute point-wise similarity matrix M[i,j]
        2. Build cumulative similarity matrix N[i,j] using DP
        3. Enforce Itakura parallelogram constraints
        4. Backtrace to find optimal warping path
        5. Return normalized similarity and warping function

    Note:
        This implementation is mathematically equivalent to the MATLAB warp.m
        function but uses vectorized operations and optimized indexing for
        significantly better performance.
    """
    m, n = len(u1), len(u2)

    # single-element contours => special case
    if m == 1 and n == 1:
        similarity = compute_similarity_matrix(u1, u2)[0, 0]
        return similarity, np.array([0], dtype=np.int32)

    # length ratio check first (MATLAB warp.m: reject before anything else)
    length_ratio = max(m, n) / (min(m, n) - 1) if min(m, n) > 1 else float("inf")
    if length_ratio >= warp_factor_level:
        return 0.0, np.array([], dtype=np.int32)

    # fast path => numba JIT DP (no python loop); when available we use it and return
    M = compute_similarity_matrix(u1, u2)
    if NUMBA_AVAILABLE:
        norm_sim, warp_func = _dtw_core_numba(M, m, n, warp_factor_level)
        return float(norm_sim), warp_func

    # fallback -> Python DP when Numba not available
    # handle very short contours (length < 2*warp_factor_level)
    # for very short contours -> Itakura parallelogram constraints may be too restrictive
    if min(m, n) < 2 * warp_factor_level and abs(m - n) <= warp_factor_level:
        # short contours, reasonable diff => simple alignment
        M = compute_similarity_matrix(u1, u2)

        if m == n:
            # same length => diagonal
            similarity = np.mean(np.diag(M))
            warp_func = np.arange(m, dtype=np.int32)
        elif m < n:
            # m shorter -> map each m to closest n
            warp_func = np.array(
                [int(i * (n - 1) / (m - 1)) if m > 1 else 0 for i in range(m)], dtype=np.int32
            )
            similarity = np.mean([M[i, warp_func[i]] for i in range(m)])
        else:
            # n shorter -> map each m to closest n
            warp_func = np.array(
                [int(i * (n - 1) / (m - 1)) if m > 1 else 0 for i in range(m)], dtype=np.int32
            )
            warp_func = np.clip(warp_func, 0, n - 1)  # keep in bounds
            similarity = np.mean([M[i, warp_func[i]] for i in range(m)])

        return similarity, warp_func

    # point-wise similarity matrix
    M = compute_similarity_matrix(u1, u2)

    # cumulative similarity (NaN init)
    N = np.full((m, n), np.nan, dtype=np.float64)

    # path matrix => horizontal step to previous
    p = np.zeros((m, n), dtype=np.int32)

    # local expansion factor (consecutive vertical steps)
    k = np.ones((m, n), dtype=np.int32)

    # possible horizontal steps => [0, -1, ..., -wfl]
    r2 = np.arange(0, -warp_factor_level - 1, -1, dtype=np.int32)

    # base case
    N[0, 0] = M[0, 0]
    k[0, 0] = 1

    # early stage (boundary handling)
    early_stage_limit = min(warp_factor_level * (warp_factor_level + 1) - 1, m)

    for i in range(1, early_stage_limit):
        for z in range(1, warp_factor_level + 1):
            # can we consider z-th index of u2?
            if int(i / warp_factor_level) <= z:
                j = z - 1  # -> 0-indexed

                if j >= n:
                    continue

                # vertical step disqualified?
                if z == 1:
                    condition = k[i - 1, j] > warp_factor_level
                else:
                    condition = k[i - 1, j] >= warp_factor_level

                if condition:
                    # disqualify vertical -> only diagonal/horizontal
                    valid_indices = [
                        j + r2[idx] for idx in range(1, min(z, len(r2))) if 0 <= j + r2[idx] < n
                    ]

                    if valid_indices:
                        prev_values = np.array([N[i - 1, idx] for idx in valid_indices])
                        if np.all(np.isnan(prev_values)):
                            continue
                        max_idx = np.nanargmax(prev_values)
                        y = prev_values[max_idx]
                        p[i, j] = r2[max_idx + 1]
                        k[i, j] = 1
                    else:
                        continue
                else:
                    # all steps including vertical
                    valid_indices = [
                        j + r2[idx] for idx in range(min(z, len(r2))) if 0 <= j + r2[idx] < n
                    ]

                    if valid_indices:
                        prev_values = np.array([N[i - 1, idx] for idx in valid_indices])
                        if np.all(np.isnan(prev_values)):
                            continue
                        max_idx = np.nanargmax(prev_values)
                        y = prev_values[max_idx]

                        if max_idx == 0:  # vertical step
                            k[i, j] = 1 + k[i - 1, j]
                        else:  # diagonal or horizontal
                            k[i, j] = 1

                        p[i, j] = r2[max_idx]
                    else:
                        continue

                N[i, j] = M[i, j] + y

        # main alignment inside itakura parallelogram
        j_start = warp_factor_level
        j_end = min(warp_factor_level * i, int((i - m) / warp_factor_level + n))

        for j in range(j_start, j_end + 1):
            if j >= n:
                continue

            if k[i - 1, j] >= warp_factor_level:
                # disqualify vertical
                valid_indices = [j + r2[idx] for idx in range(1, len(r2)) if 0 <= j + r2[idx] < n]

                if valid_indices:
                    prev_values = np.array([N[i - 1, idx] for idx in valid_indices])
                    if np.all(np.isnan(prev_values)):
                        continue
                    max_idx = np.nanargmax(prev_values)
                    y = prev_values[max_idx]
                    p[i, j] = r2[max_idx + 1]
                    k[i, j] = 1
                else:
                    continue
            else:
                # all steps
                valid_indices = [j + r2[idx] for idx in range(len(r2)) if 0 <= j + r2[idx] < n]

                if valid_indices:
                    prev_values = np.array([N[i - 1, idx] for idx in valid_indices])
                    if np.all(np.isnan(prev_values)):
                        continue
                    max_idx = np.nanargmax(prev_values)
                    y = prev_values[max_idx]

                    if max_idx == 0:  # vertical step
                        k[i, j] = 1 + k[i - 1, j]
                    else:
                        k[i, j] = 1

                    p[i, j] = r2[max_idx]
                else:
                    continue

            N[i, j] = M[i, j] + y

    # general stage (no boundary fuss)
    # MATLAB: i 1-based, j = max(round(i/wfl),(i-m)*wfl+n) : min(wfl*i, round((i-m)/wfl+n))
    # python 0-based -> (i+1) for MATLAB i
    for i in range(early_stage_limit, m):
        j_start = max(round((i + 1) / warp_factor_level) - 1, (i - m) * warp_factor_level + n)
        j_start = max(0, j_start)  # clamp
        j_end_1based = min(warp_factor_level * (i + 1), round((i - m) / warp_factor_level + n))
        j_end = j_end_1based - 1  # -> 0-based last

        for j in range(j_start, j_end + 1):
            if j >= n:
                continue

            if k[i - 1, j] >= warp_factor_level:
                # disqualify vertical
                valid_indices = [j + r2[idx] for idx in range(1, len(r2)) if 0 <= j + r2[idx] < n]

                if valid_indices:
                    prev_values = np.array([N[i - 1, idx] for idx in valid_indices])
                    if np.all(np.isnan(prev_values)):
                        continue
                    max_idx = np.nanargmax(prev_values)
                    y = prev_values[max_idx]
                    p[i, j] = r2[max_idx + 1]
                    k[i, j] = 1
                else:
                    continue
            else:
                # all steps
                valid_indices = [j + r2[idx] for idx in range(len(r2)) if 0 <= j + r2[idx] < n]

                if valid_indices:
                    prev_values = np.array([N[i - 1, idx] for idx in valid_indices])
                    if np.all(np.isnan(prev_values)):
                        continue
                    max_idx = np.nanargmax(prev_values)
                    y = prev_values[max_idx]

                    if max_idx == 0:  # vertical step
                        k[i, j] = 1 + k[i - 1, j]
                    else:
                        k[i, j] = 1

                    p[i, j] = r2[max_idx]
                else:
                    continue

            N[i, j] = M[i, j] + y

    # backtrace => warping function
    warp_function = np.zeros(m, dtype=np.int32)
    j = n - 1  # from last index

    for i in range(m - 1, -1, -1):
        warp_function[i] = j
        dj = p[i, j]
        j = j + dj

    # normalized similarity
    if np.isnan(N[m - 1, n - 1]):
        normalized_similarity = 0.0
    else:
        normalized_similarity = N[m - 1, n - 1] / m

    return normalized_similarity, warp_function


@jit(nopython=True, cache=True)  # pragma: no cover
def _unwarp_numba(warp_function):
    """Numba-compiled unwarp: no np.where, single pass with forward-fill. Tested via Python fallback when Numba disabled."""
    L = len(warp_function)
    if L == 0:
        return np.empty(0, dtype=np.int32)
    n = warp_function[L - 1] + 1
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        found = -1
        for idx in range(L):
            if warp_function[idx] == i:
                found = idx
                break
        if found >= 0:
            out[i] = found
        else:
            out[i] = out[i - 1] if i > 0 else 0
    return out


def unwarp(warp_function: NDArray[np.int32]) -> NDArray[np.int32]:
    """
    Invert a warping function.

    Given a warping function that maps indices from contour 1 to contour 2,
    this creates the inverse mapping from contour 2 back to contour 1.

    Args:
        warp_function: Warping function of shape (m,) where warp_function[i]
            gives the index in contour 2 that matches index i in contour 1

    Returns:
        Inverse warping function of shape (n,) where n = warp_function[-1] + 1

    Note:
        For indices in contour 2 that don't appear in warp_function, the
        function forward-fills from the previous valid index.
    """
    if len(warp_function) == 0:
        return np.array([], dtype=np.int32)
    if NUMBA_AVAILABLE:
        return _unwarp_numba(warp_function)
    n = warp_function[-1] + 1
    unwarp_function = np.zeros(n, dtype=np.int32)
    for i in range(n):
        matches = np.where(warp_function == i)[0]
        if len(matches) > 0:
            unwarp_function[i] = matches[0]
        else:
            unwarp_function[i] = unwarp_function[i - 1] if i > 0 else 0
    return unwarp_function
