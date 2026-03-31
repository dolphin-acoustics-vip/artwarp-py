# OCEANS Integration Guide

**OCEANS** (Odontocete Call Environment and Archival Network) is the University of St Andrews
dolphin acoustics database, developed by **James Sullivan**:
[github.com/dolphin-acoustics-vip/database-management-system](https://github.com/dolphin-acoustics-vip/database-management-system)

The `artwarp.oceans` subpackage bridges OCEANS and artwarp-py, letting you fetch real dolphin
whistle selections directly into the training pipeline without leaving the terminal.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Credentials & Secret Management](#2-credentials--secret-management)
3. [Quick Start](#3-quick-start)
4. [Full Pipeline](#4-full-pipeline)
5. [Python API Reference](#5-python-api-reference)
6. [CLI Reference](#6-cli-reference)
7. [Interactive Launcher (`oceans.sh`)](#7-interactive-launcher-oceanssh)
8. [run.sh Integration](#8-runsh-integration)
9. [Contour Extraction Parameters](#9-contour-extraction-parameters)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Install the `requests` package (not bundled with the base artwarp-py install because
the OCEANS integration is optional):

```bash
> pip install artwarp-py[oceans]   # preferred —> installs requests alongside artwarp-py
# or
> pip install requests             # if artwarp-py is already installed
```

`scipy` (already a core artwarp-py dependency) handles the WAV → contour extraction.

To verify your installation:

```python
from artwarp.oceans import OceansClient
print("OCEANS integration ready")
```

---

## 2. Credentials & Secret Management

> **Never commit credentials to source control.**

OCEANS uses token-based bearer authentication. Tokens are valid for approximately
30 minutes and are obtained by logging in with your OCEANS account.

### Environment variables (recommended)

Set these in your shell before running artwarp-py. A good place is a `.env` file
that is **listed in `.gitignore`**:

```bash
> export OCEAN_USERNAME="your@email.ac.uk"
> export OCEAN_PASSWORD="your_password"
```

Or, if you already have a token:

```bash
> export OCEAN_ACCESS_TOKEN="eyJ..."   # pre-obtained JWT — takes priority
```

### Server selection

```bash
# production server (requires OCEANS API privileges on your account)
> export OCEAN_BASE_URL="https://research.st-andrews.ac.uk/ocean/api"   # default

# test server (no API privileges required — good for development)
> export OCEAN_BASE_URL="https://rescomp-test-2.st-andrews.ac.uk/ocean/api"
```

### Priority order

The library resolves credentials in this order (highest → lowest priority):

1. `access_token` argument to `OceansClient()`
2. `OCEAN_ACCESS_TOKEN` environment variable
3. `username` / `password` arguments to `OceansClient()`
4. `OCEAN_USERNAME` / `OCEAN_PASSWORD` environment variables
5. Interactive prompt (only in interactive terminals)

---

## 3. Quick Start

### Python (3 lines)

```python
from artwarp.oceans import fetch_contours_to_dir

# reads OCEAN_USERNAME / OCEAN_PASSWORD from environment
n = fetch_contours_to_dir("./contours_ocean", max_per_species=20)
print(f"Wrote {n} contour CSVs — ready for training")
```

### CLI (one command)

```bash
artwarp-py oceans fetch --output-dir ./contours_ocean --max-per-species 20
```

### Interactive launcher

```bash
./oceans.sh     # guided menu: Fetch / Count
# or via the main launcher:
./run.sh        # choose option 5 — OCEANS
```

---

## 4. Full Pipeline

```
OCEANS API
  species ──→ encounters ──→ recordings ──→ selections
                                              │
                                    download WAV per selection
                                              │
                                    scipy spectrogram
                                    + peak-frequency tracking
                                              │
                                    save one CSV per selection
                                              │
artwarp-py train --input-dir ./contours_ocean --format csv
```

End-to-end example:

```python
from artwarp.oceans import fetch_contours_to_dir
from artwarp import ARTwarp, load_contours

# 1. Fetch contours from OCEANS
n = fetch_contours_to_dir("./contours_ocean", max_per_species=50)

# 2. Load and train
contours, names = load_contours("./contours_ocean", file_format="csv")
results = ARTwarp(vigilance=85.0).fit(contours)

print(f"Trained on {len(contours)} contours → {results.num_categories} categories")
```

---

## 5. Python API Reference

### `OceansClient`

Low-level authenticated REST client.

```python
from artwarp.oceans import OceansClient

client = OceansClient(
    base_url=None,        # override OCEAN_BASE_URL env var
    username=None,        # override OCEAN_USERNAME env var
    password=None,        # override OCEAN_PASSWORD env var
    access_token=None,    # pre-obtained JWT — skips login
    verbose=False,        # print progress messages
)
```

**Authentication**

```python
token = client.login("your@email.ac.uk", "your_password")
# OceansAuthError raised on wrong credentials
# Token is cached; client re-authenticates automatically on 401
```

**Metadata endpoints**

```python
# all encounters for a species
encounters = client.get_encounters(species_id="3ebfce8d-769b-11ef-9a56-0050568e393c")

# all recordings for an encounter
recordings = client.get_recordings(encounter_id=encounters[0]["id"])

# all selections for a recording (each has a WAV file attached)
selections = client.get_selections(recording_id=recordings[0]["id"])

# Example selection dict keys:
#   "id"                  — UUID of the selection
#   "recording_id"        — parent recording UUID
#   "selection_file_id"   — file UUID to pass to download_wav()
#   "start_time"          — start time in the recording (seconds)
#   "end_time"            — end time in the recording (seconds)
```

**File download**

```python
# download WAV audio for a selection
filename, wav_bytes = client.download_wav(selections[0]["selection_file_id"])

# download pre-computed spectrogram PNG
filename, png_bytes = client.download_spectrogram(selections[0]["id"])
```

### `fetch_contours_to_dir`

High-level pipeline function.

```python
from artwarp.oceans import fetch_contours_to_dir

n = fetch_contours_to_dir(
    output_dir="./contours_ocean",      # created if it doesn't exist
    species_ids=None,                   # list of UUIDs; None = default (bottlenose dolphin)
    max_per_species=None,               # None = fetch all
    client=None,                        # pass a pre-built OceansClient, or None to auto-create
    username=None,                      # passed to OceansClient if client is None
    password=None,
    access_token=None,
    nperseg=256,                        # spectrogram segment length
    peak_quantile=0.9,                  # noise-suppression quantile
    verbose=True,
)
```

Returns the number of CSV files written.

### `count_available_selections`

Browse without downloading:

```python
from artwarp.oceans.contours import count_available_selections

stats = count_available_selections()
# {'3ebfce8d-...': 142, '3ebfcfab-...': 78, 'total': 220}
print(f"Total selections available: {stats['total']}")
```

### `DEFAULT_SPECIES_IDS`

The built-in species UUIDs (bottlenose dolphin, *Tursiops truncatus*) that are
used when no `species_ids` argument is passed:

```python
from artwarp.oceans import DEFAULT_SPECIES_IDS
print(DEFAULT_SPECIES_IDS)
```

### Exception types

| Exception | When raised |
|---|---|
| `OceansAuthError` | Wrong credentials, expired token, or login refused |
| `OceansAPIError` | Non-200 HTTP response from any metadata or file endpoint |
| `ValueError` | Malformed WAV data during contour extraction |
| `ImportError` | `requests` not installed (`pip install artwarp-py[oceans]`) |

---

## 6. CLI Reference

### `artwarp-py oceans fetch`

```
artwarp-py oceans fetch \
  --output-dir DIR            Output directory for CSV files (required)
  [--species-id UUID]         Species UUID (repeatable; default: built-in)
  [--max-per-species N]       Cap per-species download (default: all)
  [--nperseg N]               Spectrogram segment length (default: 256)
  [--peak-quantile Q]         Noise-suppression quantile 0–1 (default: 0.9)
  [-q / --quiet]              Suppress progress output
```

Example:

```bash
> export OCEAN_USERNAME="your@email.ac.uk"
> export OCEAN_PASSWORD="your_password"

> artwarp-py oceans fetch \
   --output-dir ./contours_ocean \
   --max-per-species 50 \
   --nperseg 256
```

After fetching, train immediately:

```bash
> artwarp-py train \
   --input-dir ./contours_ocean \
   --format csv \
   --output results.pkl
```

### `artwarp-py oceans count`

Count available selections without downloading:

```bash
> artwarp-py oceans count
# or for specific species:
> artwarp-py oceans count --species-id UUID1 --species-id UUID2
```

---

## 7. Interactive Launcher (`oceans.sh`)

`oceans.sh` is a dedicated interactive Bash launcher for the OCEANS pipeline,
in the same style as `run.sh`:

```bash
> ./oceans.sh          # interactive guided menu
> ./oceans.sh fetch -o ./contours_ocean --max-per-species 50
> ./oceans.sh count
```

The interactive menu guides you through:
- Credential status check (shows env-var status without revealing values)
- Output directory selection
- Species UUID input (custom or default)
- Per-species limit
- Spectrogram extraction parameters
- Confirmation before running

---

## 8. `run.sh` Integration

The main `run.sh` launcher includes OCEANS as **option 5** in its menu:

```
1) Train   — Train network on contour directory, save model (.pkl)
2) Plot    — Generate visualization report from results (.pkl)
3) Predict — Predict categories for new contours using trained model
4) Export  — Export reference contours / category assignments from .pkl
5) OCEANS  — Fetch dolphin call data from OCEANS and extract contours
6) Quit
```

Choose option 5 to access a streamlined OCEANS fetch dialog directly from
the main launcher.

---

## 9. Contour Extraction Parameters

Each WAV file is processed by `scipy.signal.spectrogram` followed by peak-frequency
tracking:

| Parameter | Default | Effect |
|---|---|---|
| `nperseg` | 256 | Spectrogram window size in samples. Larger → finer frequency resolution, coarser time resolution. |
| `peak_quantile` | 0.9 | Frames with peak power below this quantile are replaced by the overall median peak frequency. Set to 0.0 to disable. |

**Choosing `nperseg`:**
- Short calls (< 0.5 s), high pitch variation → try 128 or 64 for better time resolution.
- Long, stable tonal calls → 256 or 512 for sharper frequency peaks.
- Rule of thumb: `nperseg / sample_rate` gives the window duration in seconds.

---

## 10. Troubleshooting

### "No module named 'requests'"

Install the optional OCEANS dependency:

```bash
> pip install artwarp-py[oceans]
```

### "OCEANS authentication failed (400)"

- Double-check username and password (note: passwords may include special characters such as trailing periods).
- Ensure no leading/trailing whitespace in environment variables.
- Try the test server: `export OCEAN_BASE_URL="https://rescomp-test-2.st-andrews.ac.uk/ocean/api"`

### "Token expired or invalid, re-authenticating"

Tokens are valid for ~30 minutes. The client automatically re-authenticates once
on a 401 response. If your script runs longer than 30 minutes, this is normal.

### "No contour CSV files were written"

1. Check that your account has API privileges on the production server.
2. Use the test server for development.
3. Run `artwarp-py oceans count` to verify selections are available.
4. Pass `verbose=True` (default) to see per-selection progress.

### "Cannot parse WAV data"

Some selections may have missing or corrupted WAV files. The pipeline skips these
with a warning and continues — this is expected behaviour.

---

*See also:* `docs/user/QUICK_REFERENCE.md` for the cheat-sheet,
`docs/dev/OCEANS_DEV.md` for internal implementation notes.

---

@author: Pedro Gronda Garrigues  
OCEANS database: James Sullivan (https://github.com/dolphin-acoustics-vip/database-management-system)
