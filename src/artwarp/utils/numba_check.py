"""
Numba availability check and optional install prompt.

Provides a small verifier that reports whether Numba is installed (for DTW
performance optimizations) and, if not, can offer to install it via pip or conda.

@author: Pedro Gronda Garrigues
"""

import subprocess
import sys
from typing import Optional


def numba_available() -> bool:
    """Return True if Numba can be imported, False otherwise."""
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def _pip_available() -> bool:
    """Check if pip is available (silently)."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _conda_available() -> bool:
    """Check if conda is available (silently)."""
    try:
        return subprocess.run(
            ["conda", "--version"],
            capture_output=True,
            timeout=5,
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def report_numba_status(stream: Optional[object] = None) -> bool:
    """
    Print whether Numba is installed (one line). No install prompt.
    Use this when using the API (e.g. in a Jupyter notebook) so users see the status.
    """
    if stream is None:
        stream = sys.stderr
    if numba_available():
        print("Numba: installed (performance optimizations enabled).", file=stream)
        return True
    print(
        "Numba: not installed. DTW performance optimizations will not be in effect. "
        "Install with: pip install numba",
        file=stream,
    )
    return False


def check_numba(
    offer_install: bool = True,
    stream: Optional[object] = None,
) -> bool:
    """
    Report whether Numba is installed and optionally offer to install it.

    - If Numba is available -> print a short status line to stream.
    - If not, print a warning that performance optimizations will not be in effect;
      when offer_install is True and stdin is a TTY, prompt to install via pip/conda
      (checks for pip and conda silently) and run the install if the user agrees;

    Args:
        offer_install: If True and not in a non-interactive context, offer to install.
        stream: Where to print messages (default: sys.stderr).

    Returns:
        True if Numba is available after this call, False otherwise.
    """
    if stream is None:
        stream = sys.stderr

    if numba_available():
        print("Numba: installed (performance optimizations enabled).", file=stream)
        return True

    print(
        "Numba: not installed. DTW and related performance optimizations will not be in effect.",
        file=stream,
    )
    print(
        "  Install for a significant speed boost: pip install numba  (or conda install numba)",
        file=stream,
    )

    if not offer_install or not sys.stdin.isatty():
        return False

    # check for pip/conda silently
    has_pip = _pip_available()
    has_conda = _conda_available()

    if not has_pip and not has_conda:
        print("  (pip and conda not detected; install Numba manually.)", file=stream)
        return False

    try:
        prompt_msg = "Install Numba now? [y/N] "
        print(prompt_msg, end="", file=stream)
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("", file=stream)
        return False

    if answer not in ("y", "yes"):
        return False

    if has_pip:
        cmd = [sys.executable, "-m", "pip", "install", "numba"]
        print(f"  Running: {' '.join(cmd)}", file=stream)
        try:
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("  Numba installed successfully. Restart the CLI for it to take effect.", file=stream)
                return True
            print("  Install failed. Try manually: pip install numba", file=stream)
        except Exception as e:
            print(f"  Install failed: {e}. Try manually: pip install numba", file=stream)
        return False

    if has_conda:
        cmd = ["conda", "install", "-y", "numba"]
        print(f"  Running: {' '.join(cmd)}", file=stream)
        try:
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("  Numba installed successfully. Restart the CLI for it to take effect.", file=stream)
                return True
            print("  Install failed. Try manually: conda install numba", file=stream)
        except Exception as e:
            print(f"  Install failed: {e}. Try manually: conda install numba", file=stream)
    return False
