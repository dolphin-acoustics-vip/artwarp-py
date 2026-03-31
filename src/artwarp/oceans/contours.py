"""
OCEANS → frequency contour pipeline.

Downloads selection WAV files from OCEANS, extracts a frequency contour
(peak frequency per time frame via spectrogram analysis) from each WAV, and
saves one CSV file per selection.  The resulting CSV directory is ready to be
consumed by ``artwarp-py train --input-dir <dir> --format csv``.

Contour extraction uses ``scipy.signal.spectrogram`` + argmax peak-tracking
within a configurable frequency band.  OCEANS WAV files use a 500 kHz sample
rate, so automatic ``nperseg`` selection (targeting ≈50 Hz/bin resolution) and
band-limiting to the whistle frequency range are both critical for correct
peak-tracking.

Pipeline summary::

    OCEANS API
      → species  → encounters  → recordings  → selections
      → download WAV per selection
      → scipy spectrogram  → peak_freq per frame (within freq band)
      → save as single-column CSV  (one row = one time frame, value in Hz)
    → artwarp-py train --input-dir <output_dir>

Quick start::

    from artwarp.oceans.contours import fetch_contours_to_dir

    n = fetch_contours_to_dir(
        output_dir="./contours_ocean",
        max_per_species=50,
    )
    print(f"{n} contour CSVs ready for training")

@author: Pedro Gronda Garrigues
"""

import io
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from artwarp.oceans.api import DEFAULT_SPECIES_IDS, OceansAPIError, OceansClient

# scipy is a core artwarp-py dependency, so it is always available.
try:
    from scipy.io import wavfile
    from scipy.signal import spectrogram as scipy_spectrogram

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def _require_scipy() -> None:
    if not _HAS_SCIPY:
        raise ImportError(
            "scipy is required for WAV→contour extraction.\n"
            "Install with: pip install scipy"
        )


# ------------------------------------------------------------------
# Contour extraction
# ------------------------------------------------------------------

# target frequency resolution when auto-selecting nperseg
_TARGET_HZ_PER_BIN: float = 50.0


def _auto_nperseg(sample_rate: int) -> int:
    """Return the smallest power-of-two nperseg that achieves ≤ _TARGET_HZ_PER_BIN Hz/bin."""
    raw = sample_rate / _TARGET_HZ_PER_BIN
    return int(2 ** math.ceil(math.log2(max(raw, 64))))


def extract_contour_from_wav_bytes(
    wav_bytes: bytes,
    nperseg: Optional[int] = None,
    peak_quantile: float = 0.9,
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
) -> NDArray[np.float64]:
    """
    Extract a frequency contour from raw WAV bytes.

    Reads WAV data from memory (no disk I/O), computes a short-time
    spectrogram, and returns the peak frequency (Hz) at each time frame,
    optionally restricted to a frequency band [*freq_low*, *freq_high*].

    OCEANS WAV files are recorded at 500 kHz.  When *nperseg* is ``None``
    (the default), the segment length is chosen automatically to achieve
    ≈50 Hz/bin frequency resolution, which is essential for resolving
    dolphin whistles (4–12 kHz) correctly.

    Frames whose peak power falls below *peak_quantile* of all frame-peak
    powers are set to the median peak frequency (de-noising heuristic).

    Parameters
    ----------
    wav_bytes : bytes
        Raw WAV file data (as downloaded from OCEANS).
    nperseg : int or None
        Number of samples per spectrogram segment.  ``None`` (default) enables
        automatic selection targeting ≈50 Hz/bin.  Larger values give finer
        frequency resolution but coarser time resolution.
    peak_quantile : float
        Frames with peak power below this quantile of all peaks are
        replaced by the overall median peak frequency.  Set to ``0.0`` to
        disable de-noising.
    freq_low : float or None
        Lower bound (Hz) for peak tracking.  Spectrogram bins below this
        frequency are ignored.  Defaults to 0 Hz (no lower bound).
    freq_high : float or None
        Upper bound (Hz) for peak tracking.  Spectrogram bins above this
        frequency are ignored.  Defaults to Nyquist (no upper bound).

    Returns
    -------
    NDArray[np.float64]
        1D array of peak frequency values (Hz), one per time frame.
        Length equals the number of spectrogram time frames.

    Raises
    ------
    ImportError
        If scipy is not installed.
    ValueError
        If *wav_bytes* cannot be read as a WAV file.
    """
    _require_scipy()

    try:
        sample_rate, sig = wavfile.read(io.BytesIO(wav_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot parse WAV data: {exc}") from exc

    # Convert to mono float64 in [-1, 1].
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if np.issubdtype(sig.dtype, np.integer):
        sig = sig.astype(np.float64) / float(np.iinfo(sig.dtype).max)
    else:
        sig = sig.astype(np.float64)

    seg = nperseg if nperseg is not None else _auto_nperseg(sample_rate)
    freqs, _times, Sxx = scipy_spectrogram(sig, fs=sample_rate, nperseg=seg)

    # restrict peak search to the requested frequency band
    f_lo = freq_low if freq_low is not None else 0.0
    f_hi = freq_high if freq_high is not None else float(freqs[-1])
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band_mask):
        # fallback: use the full spectrum if the band mask selects nothing
        band_mask = np.ones(len(freqs), dtype=bool)

    Sxx_band = Sxx[band_mask, :]
    freqs_band = freqs[band_mask]
    peak_bins = np.argmax(Sxx_band, axis=0)
    contour_hz = freqs_band[peak_bins].astype(np.float64)

    # optional: suppress noisy low-power frames
    if peak_quantile > 0.0:
        frame_peak_power = Sxx_band[peak_bins, np.arange(len(peak_bins))]
        threshold = np.quantile(frame_peak_power, peak_quantile)
        noisy = frame_peak_power < threshold
        if np.any(~noisy):
            contour_hz[noisy] = float(np.median(contour_hz[~noisy]))

    return contour_hz


def extract_contour_from_wav_file(
    wav_path: Path,
    nperseg: Optional[int] = None,
    peak_quantile: float = 0.9,
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
) -> NDArray[np.float64]:
    """
    Extract a frequency contour from a WAV file on disk.

    Convenience wrapper around :func:`extract_contour_from_wav_bytes`
    that reads from a file path.

    Parameters
    ----------
    wav_path : Path
        Path to a WAV audio file.
    nperseg : int or None
        Spectrogram segment length (see :func:`extract_contour_from_wav_bytes`).
    peak_quantile : float
        Noise-suppression quantile (see :func:`extract_contour_from_wav_bytes`).
    freq_low : float or None
        Lower frequency bound for peak tracking (Hz).
    freq_high : float or None
        Upper frequency bound for peak tracking (Hz).

    Returns
    -------
    NDArray[np.float64]
        1D array of peak frequency values (Hz).
    """
    return extract_contour_from_wav_bytes(
        wav_path.read_bytes(),
        nperseg=nperseg,
        peak_quantile=peak_quantile,
        freq_low=freq_low,
        freq_high=freq_high,
    )


# ------------------------------------------------------------------
# High-level pipeline
# ------------------------------------------------------------------


def fetch_contours_to_dir(
    output_dir: str,
    species_ids: Optional[List[str]] = None,
    max_per_species: Optional[int] = None,
    client: Optional[OceansClient] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    access_token: Optional[str] = None,
    nperseg: Optional[int] = None,
    peak_quantile: float = 0.9,
    freq_low: Optional[float] = None,
    freq_high: Optional[float] = None,
    verbose: bool = True,
) -> int:
    """
    Download OCEANS selections and extract frequency contours to CSV files.

    For each species → encounter → recording → selection:
    1. Download the selection WAV from OCEANS.
    2. Extract a frequency contour (peak Hz per spectrogram frame).
    3. Save as a single-column CSV in *output_dir*.

    The output directory can be passed directly to ``artwarp-py train``::

        artwarp-py train --input-dir <output_dir> --format csv

    Frequency bounds are derived automatically from each selection's
    ``selection_table`` (``low_frequency`` / ``high_frequency``) when
    *freq_low* and *freq_high* are not specified.  This ensures peak-tracking
    stays within the annotated whistle band even for high-sample-rate recordings
    (OCEANS WAVs use 500 kHz).

    Parameters
    ----------
    output_dir : str
        Directory to write contour CSV files.  Created if it does not exist.
    species_ids : list of str, optional
        OCEANS species UUIDs to fetch.  Defaults to
        :data:`artwarp.oceans.api.DEFAULT_SPECIES_IDS` (bottlenose dolphin).
    max_per_species : int, optional
        Maximum number of contour CSVs to write per species.  Useful for
        quick tests.  ``None`` (default) fetches everything available.
    client : OceansClient, optional
        Pre-constructed API client.  If ``None``, one is created from
        *username*, *password*, *access_token*, or environment variables.
    username : str, optional
        OCEANS account e-mail (overridden by ``OCEAN_USERNAME`` env var).
    password : str, optional
        OCEANS account password (overridden by ``OCEAN_PASSWORD`` env var).
    access_token : str, optional
        Pre-obtained JWT bearer token (overridden by ``OCEAN_ACCESS_TOKEN`` env var).
    nperseg : int or None
        Spectrogram segment length.  ``None`` (default) auto-selects to achieve
        ≈50 Hz/bin based on the WAV sample rate.
    peak_quantile : float
        Noise-suppression quantile for contour extraction (default 0.9).
    freq_low : float or None
        Global lower frequency bound (Hz) for peak tracking.  When ``None``
        (default), each selection uses its ``selection_table.low_frequency``
        value from OCEANS metadata (with a small safety margin).
    freq_high : float or None
        Global upper frequency bound (Hz) for peak tracking.  When ``None``
        (default), each selection uses its ``selection_table.high_frequency``
        value from OCEANS metadata (with a small safety margin).
    verbose : bool
        Print progress to stdout (default ``True``).

    Returns
    -------
    int
        Total number of contour CSV files written.

    Raises
    ------
    ImportError
        If *requests* or *scipy* are not installed.
    ValueError
        If no credentials can be resolved.
    """
    _require_scipy()

    if client is None:
        client = OceansClient(
            username=username,
            password=password,
            access_token=access_token,
            verbose=verbose,
        )

    species_ids = species_ids or DEFAULT_SPECIES_IDS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_written = 0

    for species_id in species_ids:
        species_count = 0
        if verbose:
            print(f"\n  Species: {species_id}")

        try:
            encounter_list = client.get_encounters(species_id=species_id)
        except OceansAPIError as exc:
            print(f"    Warning: could not fetch encounters — {exc}", file=sys.stderr)
            continue

        if verbose:
            print(f"  Encounters: {len(encounter_list)}")

        done = False
        for encounter in encounter_list:
            if done:
                break
            enc_id = encounter.get("id", "")

            try:
                recordings = client.get_recordings(encounter_id=enc_id)
            except OceansAPIError as exc:
                print(f"    Warning: could not fetch recordings — {exc}", file=sys.stderr)
                continue

            for recording in recordings:
                if done:
                    break
                rec_id = recording.get("id", "")

                try:
                    selections = client.get_selections(recording_id=rec_id)
                except OceansAPIError as exc:
                    print(f"    Warning: could not fetch selections — {exc}", file=sys.stderr)
                    continue

                for sel in selections:
                    if max_per_species is not None and species_count >= max_per_species:
                        done = True
                        break

                    file_id = sel.get("selection_file_id")
                    sel_id = sel.get("id", "unknown")

                    if not file_id:
                        if verbose:
                            print(f"    Skipping selection {sel_id}: no selection_file_id")
                        continue

                    # derive per-selection frequency bounds from OCEANS metadata
                    # when the caller has not provided explicit global bounds
                    sel_table: Dict[str, Any] = sel.get("selection_table") or {}
                    eff_freq_low = freq_low
                    eff_freq_high = freq_high
                    if eff_freq_low is None:
                        raw_lo = sel_table.get("low_frequency")
                        if raw_lo is not None:
                            # apply a small margin below the annotated lower bound
                            eff_freq_low = max(0.0, float(raw_lo) * 0.9)
                    if eff_freq_high is None:
                        raw_hi = sel_table.get("high_frequency")
                        if raw_hi is not None:
                            eff_freq_high = float(raw_hi) * 1.1

                    # download WAV and extract contour
                    try:
                        _filename, wav_bytes = client.download_wav(file_id)
                        contour = extract_contour_from_wav_bytes(
                            wav_bytes,
                            nperseg=nperseg,
                            peak_quantile=peak_quantile,
                            freq_low=eff_freq_low,
                            freq_high=eff_freq_high,
                        )
                    except (OceansAPIError, ValueError, RuntimeError) as exc:
                        print(f"    Warning: skipping {sel_id}: {exc}", file=sys.stderr)
                        continue

                    # save CSV: one frequency value per line
                    safe_species = species_id.replace("-", "")[:16]
                    csv_name = f"{safe_species}_{sel_id}.csv"
                    csv_path = out / csv_name
                    np.savetxt(csv_path, contour, fmt="%.4f", delimiter=",")

                    species_count += 1
                    total_written += 1

                    if verbose:
                        print(
                            f"    [{species_count}] {csv_name}  "
                            f"({len(contour)} frames, "
                            f"{contour.min():.0f}–{contour.max():.0f} Hz)"
                        )

        if verbose:
            print(f"  Written for this species: {species_count}")

    if verbose:
        print(f"\n  Total contour CSVs written: {total_written}")
        print(f"  Output directory: {out.resolve()}")

    return total_written


# ------------------------------------------------------------------
# Stats helper
# ------------------------------------------------------------------


def count_available_selections(
    species_ids: Optional[List[str]] = None,
    client: Optional[OceansClient] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    access_token: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Count how many selections are available in OCEANS without downloading.

    Useful for estimating how large a fetch will be before committing to it.

    Parameters
    ----------
    species_ids : list of str, optional
        Species UUIDs to count.  Defaults to DEFAULT_SPECIES_IDS.
    client : OceansClient, optional
        Pre-constructed client.
    username, password, access_token : str, optional
        Credentials (see :func:`fetch_contours_to_dir`).
    verbose : bool
        Print counts per species as they are retrieved.

    Returns
    -------
    dict
        ``{"species_id": int, ..., "total": int}``
        Maps each species UUID to its selection count plus a "total" key.
    """
    if client is None:
        client = OceansClient(
            username=username, password=password, access_token=access_token, verbose=False
        )

    species_ids = species_ids or DEFAULT_SPECIES_IDS
    result: Dict[str, Any] = {}
    grand_total = 0

    for species_id in species_ids:
        count = 0
        try:
            encounters = client.get_encounters(species_id=species_id)
            for enc in encounters:
                recordings = client.get_recordings(encounter_id=enc["id"])
                for rec in recordings:
                    sels = client.get_selections(recording_id=rec["id"])
                    count += sum(1 for s in sels if s.get("selection_file_id"))
        except OceansAPIError as exc:
            print(f"  Warning: error counting selections for {species_id}: {exc}", file=sys.stderr)

        result[species_id] = count
        grand_total += count
        if verbose:
            print(f"  {species_id}: {count} selections with WAV files")

    result["total"] = grand_total
    if verbose:
        print(f"  Total: {grand_total}")
    return result
