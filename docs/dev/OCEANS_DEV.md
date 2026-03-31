# OCEANS Integration — Developer Notes

Internal design, architecture decisions, and maintenance guide for the
`artwarp.oceans` subpackage.

For the end-user guide, see `docs/user/OCEANS.md`.

OCEANS is developed by **James Sullivan**:
[github.com/dolphin-acoustics-vip/database-management-system](https://github.com/dolphin-acoustics-vip/database-management-system)

---

## Contents

1. [Package Structure](#1-package-structure)
2. [API Endpoints](#2-api-endpoints)
3. [Authentication Design](#3-authentication-design)
4. [OceansClient Internals](#4-oceansclient-internals)
5. [Contour Extraction Pipeline](#5-contour-extraction-pipeline)
6. [CLI Integration](#6-cli-integration)
7. [Optional Dependency Strategy](#7-optional-dependency-strategy)
8. [Testing](#8-testing)
9. [Known Limitations](#9-known-limitations)
10. [Extending the Integration](#10-extending-the-integration)

---

## 1. Package Structure

```
src/artwarp/oceans/
├── __init__.py     Public surface: OceansClient, fetch_contours_to_dir,
│                  OceansAuthError, OceansAPIError, DEFAULT_SPECIES_IDS
├── auth.py         Credential resolution (env vars, prompt, explicit args)
├── api.py          Low-level REST client (OceansClient) + exception classes
├── contours.py     WAV → contour pipeline + high-level fetch function
└── cli.py          argparse helpers + command functions for the CLI
```

The CLI layer (`cli.py`) is separate from the business logic so that importing
`artwarp.oceans.api` or `artwarp.oceans.contours` never drags in `argparse`.

---

## 2. API Endpoints

The following endpoints have been verified against both the production and test
OCEANS servers.

### Authentication

```
POST  /auth/login/?username=<email>&password=<pass>
      → 200 {"access_token": "<jwt>"}
      → 400 {"message": "Incorrect password"}
      → 401 if account has insufficient privileges
```

Tokens expire after approximately **30 minutes**.

### Metadata

All metadata endpoints require `Authorization: Bearer <token>`.

```
GET   /metadata/encounters/
      ?species_id=<uuid>    → list of encounters for a species
      ?id=<uuid>            → single encounter

GET   /metadata/recordings/
      ?encounter_id=<uuid>  → list of recordings for an encounter
      ?id=<uuid>            → single recording

GET   /metadata/selections/
      ?recording_id=<uuid>  → list of selections for a recording
      ?id=<uuid>            → single selection

Response schema (all three endpoints return lists):
  [
    {
      "id":                 "<uuid>",
      "recording_id":       "<uuid>",      // selections only
      "encounter_id":       "<uuid>",      // recordings only
      "species_id":         "<uuid>",      // encounters only
      "selection_file_id":  "<uuid>",      // selections: file_id to pass to /filespace/file/
      "start_time":         <float>,       // selections: start time in recording (s)
      "end_time":           <float>,       // selections: end time in recording (s)
      ...
    }
  ]
```

### File access

```
GET   /filespace/file/?id=<file_id>
      Content-Type: audio/wav
      Content-Disposition: attachment; filename="<name>.wav"
      → raw WAV bytes (streamed)

GET   /filespace/spectrogram/?selection_id=<selection_id>
      Content-Type: image/png
      Content-Disposition: attachment; filename="<name>.png"
      → raw PNG bytes (pre-computed spectrogram)
```

The `file_id` for a WAV is the `selection_file_id` field on a selection dict,
**not** the selection's own `id`.

---

## 3. Authentication Design

`auth.py` is a pure utility module with no side-effects at import time. It never
stores or logs credentials.

**Credential resolution priority** (`resolve_auth`):
1. Explicit `access_token` argument
2. `OCEAN_ACCESS_TOKEN` env var
3. Explicit `username` + `password` arguments
4. `OCEAN_USERNAME` + `OCEAN_PASSWORD` env vars
5. `prompt_credentials()` — only invoked if `sys.stdin.isatty()` is True; raises
   `ValueError` in non-interactive environments (e.g. CI, notebooks)

**Why not store tokens to disk?** Tokens expire in ~30 minutes and the OCEANS
server does not provide a refresh endpoint. Persisting them would add complexity
for minimal gain and risks stale-token errors. Basically, not worth it :P

---

## 4. OceansClient Internals

`OceansClient` wraps a `requests.Session` (persistent TCP connection, shared
headers). Authentication is lazy — `_ensure_token()` is called only on the first
authenticated request.

**Automatic re-auth on 401:**

```python
def _get(self, endpoint, params=None, stream=False):
    self._ensure_token()
    response = self._session.get(url, headers=self._auth_header(), ...)
    if response.status_code == 401:
        self._token = None
        self._initial_token = None
        self._ensure_token()           # re-authenticate once
        response = self._session.get(...)  # retry
    return response
```

This handles the common case of a long-running script where the token expires
mid-run. Only one retry is attempted; if the second request also returns 401,
the error propagates to the caller.

**Streaming downloads:** `download_wav` and `download_spectrogram` use
`stream=True` + `iter_content()` to avoid loading very large WAV files
entirely into memory before processing.

**Timeouts:** `_CONNECT_TIMEOUT = 15 s`, `_READ_TIMEOUT = 120 s`. These are
conservative values suitable for production use; increase `_READ_TIMEOUT` for
very large WAV files if needed.

---

## 5. Contour Extraction Pipeline

`extract_contour_from_wav_bytes` (in `contours.py`):

1. Read WAV from `io.BytesIO` (no disk I/O, works on bytes from network).
2. Convert to mono float64 in [−1, 1] (average channels if stereo; normalise
   integer PCM by `iinfo.max`).
3. Compute short-time spectrogram via `scipy.signal.spectrogram`.
4. Find `argmax` across frequency axis → peak frequency bin per time frame.
5. Map bin indices to Hz using the `freqs` array returned by scipy.
6. Optional de-noising: replace frames whose peak power is below
   `peak_quantile` of all frame peaks with the median peak frequency.

**Why argmax peak tracking?** It is simple, fast, and works well for tonal
calls where one frequency component dominates. For broadband or multi-harmonic
calls, a more sophisticated tracker (e.g. quadratic interpolation around the
peak, or a Viterbi path through the spectrogram) would give better results;
see the "Extending" section below!

**No disk staging of WAV files:** WAV bytes are downloaded into memory,
processed, and discarded. This avoids cluttering the user's filesystem with
temporary files and makes the pipeline usable on machines with limited disk space.

---

## 6. CLI Integration

`cli.py` exports two functions consumed by `artwarp/cli/main.py`:

- `add_oceans_parser(subparsers)` — registers the `oceans` command group.
- `command_oceans(args)` — dispatches to `command_oceans_fetch` or
  `command_oceans_count`.

The integration in `main.py` is wrapped in a try/except ImportError so that the
CLI works even without `requests` installed, it just shows a helpful stub:

```python
try:
    from artwarp.oceans.cli import add_oceans_parser
    add_oceans_parser(subparsers)
except ImportError:
    # registers a stub with install instructions
```

---

## 7. Optional Dependency Strategy

`requests` is an optional dependency declared in `pyproject.toml`:

```toml
[project.optional-dependencies]
oceans = ["requests>=2.25.0"]
```

At import time, `api.py` does:

```python
try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
```

Any function that needs `requests` calls `_require_requests()` which raises a
clear `ImportError` with install instructions. This avoids confusing
`AttributeError: module 'artwarp.oceans.api' has no attribute '_requests'`
errors and lets downstream code (`cli.py`, `contours.py`) fail gracefully.

`scipy` is already a mandatory artwarp-py dependency, so `contours.py` can
import it unconditionally.

---

## 8. Testing

Tests live in `tests/unit/test_oceans.py` (50 tests, all offline / mocked).

**No real network calls in tests.** All HTTP interactions are replaced with
`unittest.mock` patches on `requests.Session.get` / `.post`.

Test categories:
- `TestGetBaseUrl`, `TestGetTokenFromEnv`, `TestGetCredentialsFromEnv`: `auth.py` utilities
- `TestPromptCredentials`, `TestResolveAuth`: credential resolution logic
- `TestOceansClientLogin`: 401/400/500 login scenarios
- `TestOceansClientMetadata`: encounters, recordings, selections (list + dict responses, 401 retry)
- `TestOceansClientFileDownload`: WAV and spectrogram download + error cases
- `TestExtractContourFromWavBytes`: spectrogram extraction (mono/stereo, de-noising, invalid input)
- `TestFetchContoursToDir`: full pipeline (mock client, max_per_species, missing file_id, API errors)
- `TestAddOceansParser`, `TestCommandOceansFetch`, etc.: CLI parser and command dispatch
- `TestOceansPublicAPI`: public surface of `artwarp.oceans.__init__`

To run just the OCEANS tests:

```bash
pytest tests/unit/test_oceans.py -v
```

**Adding tests for new endpoints:** create a `_client_with_token()` helper that
constructs an `OceansClient` in `__new__`-bypass mode (skipping `__init__` to
avoid needing the real `requests` import path), then patch `client._session.get`.

---

## 9. Known Limitations

- **No species listing endpoint.** The OCEANS API does not expose a
  `/metadata/species/` endpoint in the current (v1.2.1) version; the default
  species UUIDs are hard-coded from prior API testing.
- **No contour (`.ctr`) download.** OCEANS stores Raven Pro contour files for
  some selections, but there is currently no public endpoint to fetch them
  directly. The WAV-based extraction in `contours.py` is the supported path.
- **Token expiry.** Tokens expire in ~30 minutes with no refresh endpoint. For
  very large fetches that take longer, the client will automatically re-authenticate
  once per expired token; if credentials are not stored in env vars (e.g. only
  a token was provided), re-auth will fail.
- **Rate limiting.** The production server may rate-limit aggressive parallel
  requests. The current implementation is sequential (one request at a time).

---

## 10. Extending the Integration

### Adding a new endpoint

1. Add a method to `OceansClient` in `api.py`, following the pattern of
   `get_encounters` / `get_recordings` / `get_selections`.
2. Document it in `docs/user/OCEANS.md` under **Python API Reference**.
3. Add corresponding tests in `tests/unit/test_oceans.py`.

### Better contour extraction

Replace or augment `extract_contour_from_wav_bytes` in `contours.py`:
- The function signature and return type (`NDArray[np.float64]`) should stay the same.
- Consider Viterbi path tracking (via `librosa.sequence.viterbi_discriminative`) for
  multi-harmonic or noisy signals.
- Consider `soundfile` as a WAV reader (supports more formats than `scipy.io.wavfile`).

### Parallel fetching

`fetch_contours_to_dir` is currently sequential. For large datasets, a
`concurrent.futures.ThreadPoolExecutor` around the per-selection WAV download
and extraction loop would significantly reduce wall-clock time. The
`OceansClient` session is not thread-safe; use one client per worker thread.

---

@author: Pedro Gronda Garrigues  
OCEANS database: James Sullivan (https://github.com/dolphin-acoustics-vip/database-management-system)
