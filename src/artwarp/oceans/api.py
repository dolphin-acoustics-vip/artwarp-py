"""
Low-level OCEANS REST API client.

OCEANS (Odontocete Call Environment and Archival Network) is the University of
St Andrews dolphin acoustics database, developed by James Sullivan:
  https://github.com/dolphin-acoustics-vip/database-management-system

API Endpoints
=============

Authentication
--------------
POST  /auth/login/?username=<email>&password=<pass>
      Response: {"access_token": "<jwt>"}
      Tokens expire after ~30 minutes.

Metadata  (all require Authorization: Bearer <token>)
---------
GET   /metadata/encounters/
      Query params:
        species_id=<uuid>   → encounters for a species
        id=<uuid>           → single encounter by ID
      Response: list of encounter dicts
        {"id": "...", "name": "...", "species_id": "...", ...}

GET   /metadata/recordings/
      Query params:
        encounter_id=<uuid> → recordings for an encounter
        id=<uuid>           → single recording by ID
      Response: list of recording dicts
        {"id": "...", "encounter_id": "...", ...}

GET   /metadata/selections/
      Query params:
        recording_id=<uuid> → selections for a recording
        id=<uuid>           → single selection by ID
      Response: list of selection dicts
        {"id": "...", "recording_id": "...", "selection_file_id": "...",
         "start_time": ..., "end_time": ..., ...}

File access  (all require Authorization: Bearer <token>)
-----------
GET   /filespace/file/?id=<file_id>
      Content-Type:        audio/wav
      Content-Disposition: attachment; filename="<name>.wav"
      → Raw WAV bytes for the selection (use selection["selection_file_id"]).

GET   /filespace/spectrogram/?selection_id=<selection_id>
      Content-Type:        image/png
      Content-Disposition: attachment; filename="<name>.png"
      → Pre-computed spectrogram PNG for a selection.

Usage example::

    from artwarp.oceans.api import OceansClient

    client = OceansClient()                           # reads env vars
    encounters = client.get_encounters(species_id="3ebfce8d-...")
    for enc in encounters:
        recordings = client.get_recordings(encounter_id=enc["id"])
        for rec in recordings:
            selections = client.get_selections(recording_id=rec["id"])
            for sel in selections:
                filename, wav_bytes = client.download_wav(sel["selection_file_id"])
                # wav_bytes is raw PCM audio

@author: Pedro Gronda Garrigues
"""

from typing import Any, Dict, List, Optional, Tuple

from artwarp.oceans.auth import get_base_url, resolve_auth

# Guard against missing requests at import time — keep error message friendly.
try:
    import requests as _requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# Known species IDs in the production OCEANS database (verified via live API).
# Bottlenose dolphin (Tursiops truncatus).
DEFAULT_SPECIES_IDS: List[str] = [
    "3ebfce8d-769b-11ef-9a56-0050568e393c",
    "3ebfcfab-769b-11ef-9a56-0050568e393c",
]

# Network timeouts (seconds).
_CONNECT_TIMEOUT = 15
_READ_TIMEOUT = 120  # WAV downloads can be large


def _require_requests() -> None:
    """Raise ImportError with install instructions if requests is missing."""
    if not _HAS_REQUESTS:
        raise ImportError(
            "The 'requests' package is required for OCEANS integration.\n"
            "Install it with:  pip install artwarp-py[oceans]\n"
            "or directly:      pip install requests"
        )


class OceansAuthError(Exception):
    """Raised when OCEANS authentication fails (wrong credentials or expired token)."""


class OceansAPIError(Exception):
    """Raised when an OCEANS API call returns an unexpected error response."""


class OceansClient:
    """
    Thread-safe* REST client for the OCEANS API.

    Credentials are resolved lazily on the first authenticated request.
    If a request returns 401 (Unauthorized), the client re-authenticates once
    automatically and retries.

    (*) Thread safety: each client instance maintains its own session and token.
    Do not share a single instance across threads without external locking.

    Parameters
    ----------
    base_url : str, optional
        Override the API base URL. Defaults to OCEAN_BASE_URL env var, then
        the production server. Use the test server URL for development:
        ``https://rescomp-test-2.st-andrews.ac.uk/ocean/api``
    username : str, optional
        OCEANS account e-mail. Overridden by OCEAN_USERNAME env var if not set.
    password : str, optional
        OCEANS account password. Overridden by OCEAN_PASSWORD env var if not set.
    access_token : str, optional
        A pre-obtained JWT bearer token. Overrides username/password.
        Equivalent to setting OCEAN_ACCESS_TOKEN env var.
    verbose : bool
        Print progress messages (default: False).

    Examples
    --------
    Credentials from environment variables::

        # export OCEAN_USERNAME="..." OCEAN_PASSWORD="..." first
        client = OceansClient()

    Explicit credentials (avoid hard-coding in scripts)::

        import os
        client = OceansClient(
            username=os.environ["OCEAN_USERNAME"],
            password=os.environ["OCEAN_PASSWORD"],
        )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        access_token: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        _require_requests()
        self._base_url = (base_url or get_base_url()).rstrip("/")
        self._username = username
        self._password = password
        self._initial_token = access_token
        self.verbose = verbose

        self._token: Optional[str] = None
        self._session = _requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [OCEANS] {msg}")

    def _auth_header(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _ensure_token(self) -> None:
        """Authenticate if we do not yet have a token."""
        if self._token:
            return

        tok, u, p = resolve_auth(
            username=self._username,
            password=self._password,
            access_token=self._initial_token,
        )

        if tok:
            self._token = tok
            self._log("Using pre-obtained access token.")
        else:
            self._token = self.login(u, p)  # type: ignore[arg-type]

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> "_requests.Response":
        """Authenticated GET with automatic one-shot re-auth on 401."""
        self._ensure_token()
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        response = self._session.get(
            url,
            params=params,
            headers=self._auth_header(),
            stream=stream,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
        if response.status_code == 401:
            # token may have expired -> re-authenticate once and retry
            self._log("Token expired or invalid, re-authenticating...")
            self._token = None
            self._initial_token = None  # force re-login
            self._ensure_token()
            response = self._session.get(
                url,
                params=params,
                headers=self._auth_header(),
                stream=stream,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
        return response

    # ------------------------------------------------------------------
    # Public API — Authentication
    # ------------------------------------------------------------------

    def login(self, username: str, password: str) -> str:
        """
        Authenticate with OCEANS and return a bearer token.

        Parameters
        ----------
        username : str
            OCEANS account e-mail address.
        password : str
            OCEANS account password.

        Returns
        -------
        str
            JWT bearer token (valid for ~30 minutes).

        Raises
        ------
        OceansAuthError
            If the server rejects the credentials.
        OceansAPIError
            On unexpected HTTP errors.
        """
        _require_requests()
        url = f"{self._base_url}/auth/login/"
        self._log(f"Authenticating as {username!r} ...")
        response = self._session.post(
            url,
            params={"username": username, "password": password},
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )

        if response.status_code == 200:
            token = response.json().get("access_token", "")
            if not token:
                raise OceansAuthError("Login succeeded but response contained no access_token.")
            preview = token[:10] + f"...({len(token) - 10} chars)"
            self._log(f"Token obtained: {preview}")
            self._token = token
            return token

        if response.status_code in (400, 401, 403):
            raise OceansAuthError(
                f"OCEANS authentication failed ({response.status_code}): {response.text.strip()}"
            )
        raise OceansAPIError(
            f"Unexpected response from /auth/login/ ({response.status_code}): {response.text[:200]}"
        )

    # ------------------------------------------------------------------
    # Public API — Metadata
    # ------------------------------------------------------------------

    def get_encounters(
        self,
        species_id: Optional[str] = None,
        encounter_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve encounter records from OCEANS.

        Pass exactly one of *species_id* or *encounter_id*.

        Parameters
        ----------
        species_id : str, optional
            UUID of a species — returns all encounters for that species.
        encounter_id : str, optional
            UUID of a specific encounter — returns that encounter.

        Returns
        -------
        list of dict
            Each dict contains at minimum ``{"id": "<uuid>", ...}``.

        Raises
        ------
        OceansAPIError
            On non-200 responses.
        """
        params: Dict[str, str] = {}
        if species_id:
            params["species_id"] = species_id
        elif encounter_id:
            params["id"] = encounter_id

        resp = self._get("/metadata/encounters/", params=params)
        if resp.status_code != 200:
            raise OceansAPIError(
                f"GET /metadata/encounters/ returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    def get_recordings(
        self,
        encounter_id: Optional[str] = None,
        recording_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recording records from OCEANS.

        Parameters
        ----------
        encounter_id : str, optional
            UUID of an encounter — returns all recordings for that encounter.
        recording_id : str, optional
            UUID of a specific recording.

        Returns
        -------
        list of dict
        """
        params: Dict[str, str] = {}
        if encounter_id:
            params["encounter_id"] = encounter_id
        elif recording_id:
            params["id"] = recording_id

        resp = self._get("/metadata/recordings/", params=params)
        if resp.status_code != 200:
            raise OceansAPIError(
                f"GET /metadata/recordings/ returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    def get_selections(
        self,
        recording_id: Optional[str] = None,
        selection_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve selection records from OCEANS.

        Each selection represents a manually annotated call within a recording.
        The ``selection_file_id`` field is the file ID to pass to
        :meth:`download_wav`.

        Parameters
        ----------
        recording_id : str, optional
            UUID of a recording — returns all selections for that recording.
        selection_id : str, optional
            UUID of a specific selection.

        Returns
        -------
        list of dict
            Each dict includes ``{"id": "...", "selection_file_id": "...", ...}``.
        """
        params: Dict[str, str] = {}
        if recording_id:
            params["recording_id"] = recording_id
        elif selection_id:
            params["id"] = selection_id

        resp = self._get("/metadata/selections/", params=params)
        if resp.status_code != 200:
            raise OceansAPIError(
                f"GET /metadata/selections/ returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    # ------------------------------------------------------------------
    # Public API — File access
    # ------------------------------------------------------------------

    def download_wav(self, file_id: str) -> Tuple[str, bytes]:
        """
        Download a WAV audio file for a selection.

        Parameters
        ----------
        file_id : str
            The ``selection_file_id`` field from a selection dict.

        Returns
        -------
        tuple of (filename, wav_bytes)
            *filename* is the original filename from the server.
            *wav_bytes* is the raw PCM/WAV data.

        Raises
        ------
        OceansAPIError
            If the response is not a WAV file or the download fails.
        """
        resp = self._get("/filespace/file/", params={"id": file_id}, stream=True)
        if resp.status_code != 200:
            raise OceansAPIError(
                f"GET /filespace/file/ returned {resp.status_code} for file_id={file_id!r}"
            )

        content_type = resp.headers.get("Content-Type", "")
        if "audio/wav" not in content_type and "audio/x-wav" not in content_type:
            raise OceansAPIError(
                f"Expected audio/wav from /filespace/file/ but got {content_type!r} "
                f"for file_id={file_id!r}"
            )

        filename = _extract_filename(resp.headers.get("Content-Disposition", ""), file_id + ".wav")
        wav_bytes = b"".join(resp.iter_content(chunk_size=8192))
        self._log(f"Downloaded WAV '{filename}' ({len(wav_bytes):,} bytes)")
        return filename, wav_bytes

    def download_spectrogram(self, selection_id: str) -> Tuple[str, bytes]:
        """
        Download the pre-computed spectrogram PNG for a selection.

        Parameters
        ----------
        selection_id : str
            The ``id`` field from a selection dict.

        Returns
        -------
        tuple of (filename, png_bytes)
            *filename* is the original filename from the server.
            *png_bytes* is the raw PNG image data.

        Raises
        ------
        OceansAPIError
            If the response is not a PNG or the download fails.
        """
        resp = self._get(
            "/filespace/spectrogram/", params={"selection_id": selection_id}, stream=True
        )
        if resp.status_code != 200:
            raise OceansAPIError(
                f"GET /filespace/spectrogram/ returned {resp.status_code} "
                f"for selection_id={selection_id!r}"
            )

        content_type = resp.headers.get("Content-Type", "")
        if "image/png" not in content_type:
            raise OceansAPIError(
                f"Expected image/png from /filespace/spectrogram/ but got {content_type!r} "
                f"for selection_id={selection_id!r}"
            )

        filename = _extract_filename(
            resp.headers.get("Content-Disposition", ""), selection_id + ".png"
        )
        png_bytes = b"".join(resp.iter_content(chunk_size=8192))
        self._log(f"Downloaded spectrogram '{filename}' ({len(png_bytes):,} bytes)")
        return filename, png_bytes


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _extract_filename(content_disposition: str, fallback: str) -> str:
    """Extract filename from a Content-Disposition header value."""
    if "filename=" in content_disposition:
        return content_disposition.split("filename=")[1].strip().strip('"').strip("'")
    return fallback
