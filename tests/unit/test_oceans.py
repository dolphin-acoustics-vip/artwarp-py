"""
Unit tests for the artwarp.oceans subpackage.

All network calls are mocked — these tests never hit the real OCEANS API.

Test coverage:
  - artwarp.oceans.auth   : credential resolution, prompt behaviour
  - artwarp.oceans.api    : OceansClient (login, metadata endpoints, file download)
  - artwarp.oceans.contours: extract_contour_from_wav_bytes, fetch_contours_to_dir
  - artwarp.oceans.cli    : argument parser construction and command dispatch

@author: Pedro Gronda Garrigues
"""

import io
import os
import struct
import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers — minimal in-memory WAV builder
# ---------------------------------------------------------------------------

def _make_wav_bytes(num_samples: int = 512, sample_rate: int = 44100) -> bytes:
    """Return a minimal valid WAV file as bytes (mono, 16-bit, sine wave)."""
    t = np.linspace(0, num_samples / sample_rate, num_samples)
    sig = (np.sin(2 * np.pi * 5000 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    # RIFF header
    data_bytes = sig.tobytes()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data_bytes)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))      # chunk size
    buf.write(struct.pack("<H", 1))       # PCM
    buf.write(struct.pack("<H", 1))       # mono
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))
    buf.write(struct.pack("<H", 2))       # block align
    buf.write(struct.pack("<H", 16))      # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data_bytes)))
    buf.write(data_bytes)
    return buf.getvalue()


def _mock_response(
    status_code: int = 200,
    json_data: Any = None,
    content: bytes = b"",
    headers: dict = None,
) -> MagicMock:
    """Build a minimal mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = str(json_data or "")
    resp.content = content
    resp.headers = headers or {}
    resp.iter_content = lambda chunk_size=8192: iter([content])
    return resp


# ---------------------------------------------------------------------------
# auth.py tests
# ---------------------------------------------------------------------------

class TestGetBaseUrl(unittest.TestCase):
    def test_default_production(self):
        from artwarp.oceans.auth import DEFAULT_BASE_URL, get_base_url
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCEAN_BASE_URL", None)
            self.assertEqual(get_base_url(), DEFAULT_BASE_URL)

    def test_env_override_strips_slash(self):
        from artwarp.oceans.auth import get_base_url
        with patch.dict(os.environ, {"OCEAN_BASE_URL": "https://test.example.com/api/"}):
            self.assertEqual(get_base_url(), "https://test.example.com/api")


class TestGetTokenFromEnv(unittest.TestCase):
    def test_returns_token_when_set(self):
        from artwarp.oceans.auth import get_token_from_env
        with patch.dict(os.environ, {"OCEAN_ACCESS_TOKEN": "mytoken123"}):
            self.assertEqual(get_token_from_env(), "mytoken123")

    def test_returns_none_when_empty(self):
        from artwarp.oceans.auth import get_token_from_env
        with patch.dict(os.environ, {"OCEAN_ACCESS_TOKEN": "  "}):
            self.assertIsNone(get_token_from_env())

    def test_returns_none_when_unset(self):
        from artwarp.oceans.auth import get_token_from_env
        env = {k: v for k, v in os.environ.items() if k != "OCEAN_ACCESS_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(get_token_from_env())


class TestGetCredentialsFromEnv(unittest.TestCase):
    def test_both_set(self):
        from artwarp.oceans.auth import get_credentials_from_env
        with patch.dict(os.environ, {"OCEAN_USERNAME": "u@ex.com", "OCEAN_PASSWORD": "pass"}):
            u, p = get_credentials_from_env()
            self.assertEqual(u, "u@ex.com")
            self.assertEqual(p, "pass")

    def test_both_unset(self):
        from artwarp.oceans.auth import get_credentials_from_env
        env = {k: v for k, v in os.environ.items()
               if k not in ("OCEAN_USERNAME", "OCEAN_PASSWORD")}
        with patch.dict(os.environ, env, clear=True):
            u, p = get_credentials_from_env()
            self.assertIsNone(u)
            self.assertIsNone(p)


class TestPromptCredentials(unittest.TestCase):
    def test_raises_when_not_tty(self):
        from artwarp.oceans.auth import prompt_credentials
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with self.assertRaises(ValueError):
                prompt_credentials()

    def test_prompts_when_tty(self):
        from artwarp.oceans.auth import prompt_credentials
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", return_value="u@ex.com"):
                with patch("getpass.getpass", return_value="secret"):
                    u, p = prompt_credentials()
                    self.assertEqual(u, "u@ex.com")
                    self.assertEqual(p, "secret")

    def test_raises_on_empty_username(self):
        from artwarp.oceans.auth import prompt_credentials
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", return_value=""):
                with self.assertRaises(ValueError, msg="Username cannot be empty"):
                    prompt_credentials()


class TestResolveAuth(unittest.TestCase):
    def test_explicit_token_wins(self):
        from artwarp.oceans.auth import resolve_auth
        tok, u, p = resolve_auth(access_token="explicit_token")
        self.assertEqual(tok, "explicit_token")
        self.assertIsNone(u)
        self.assertIsNone(p)

    def test_env_token_wins_over_credentials(self):
        from artwarp.oceans.auth import resolve_auth
        with patch.dict(os.environ, {"OCEAN_ACCESS_TOKEN": "env_token"}):
            tok, u, p = resolve_auth(username="u", password="p")
            self.assertEqual(tok, "env_token")

    def test_explicit_credentials(self):
        from artwarp.oceans.auth import resolve_auth
        env = {k: v for k, v in os.environ.items()
               if k not in ("OCEAN_ACCESS_TOKEN", "OCEAN_USERNAME", "OCEAN_PASSWORD")}
        with patch.dict(os.environ, env, clear=True):
            tok, u, p = resolve_auth(username="user@ex.com", password="pw")
            self.assertIsNone(tok)
            self.assertEqual(u, "user@ex.com")
            self.assertEqual(p, "pw")

    def test_env_credentials(self):
        from artwarp.oceans.auth import resolve_auth
        env = {k: v for k, v in os.environ.items() if k != "OCEAN_ACCESS_TOKEN"}
        env["OCEAN_USERNAME"] = "env@ex.com"
        env["OCEAN_PASSWORD"] = "envpw"
        with patch.dict(os.environ, env, clear=True):
            tok, u, p = resolve_auth()
            self.assertIsNone(tok)
            self.assertEqual(u, "env@ex.com")
            self.assertEqual(p, "envpw")


# ---------------------------------------------------------------------------
# api.py tests — OceansClient
# ---------------------------------------------------------------------------

class TestOceansClientLogin(unittest.TestCase):
    def _make_client(self, **kwargs):
        from artwarp.oceans.api import OceansClient
        # Provide an env token so __init__ does not try to resolve creds.
        return OceansClient(access_token="dummy", **kwargs)

    def test_login_success(self):
        from artwarp.oceans.api import OceansClient
        client = OceansClient.__new__(OceansClient)
        client._base_url = "https://example.com/api"
        client.verbose = False
        client._token = None
        import requests as _r
        client._session = _r.Session()

        resp = _mock_response(200, {"access_token": "tok123"})
        with patch("requests.Session.post", return_value=resp):
            token = client.login("user@ex.com", "pw")
        self.assertEqual(token, "tok123")
        self.assertEqual(client._token, "tok123")

    def test_login_wrong_password(self):
        from artwarp.oceans.api import OceansAuthError, OceansClient
        client = OceansClient.__new__(OceansClient)
        client._base_url = "https://example.com/api"
        client.verbose = False
        client._token = None
        import requests as _r
        client._session = _r.Session()

        resp = _mock_response(400, {"message": "Incorrect password"})
        resp.text = '{"message": "Incorrect password"}'
        with patch("requests.Session.post", return_value=resp):
            with self.assertRaises(OceansAuthError):
                client.login("user@ex.com", "wrongpw")

    def test_login_unexpected_error(self):
        from artwarp.oceans.api import OceansAPIError, OceansClient
        client = OceansClient.__new__(OceansClient)
        client._base_url = "https://example.com/api"
        client.verbose = False
        client._token = None
        import requests as _r
        client._session = _r.Session()

        resp = _mock_response(500)
        resp.text = "Internal Server Error"
        with patch("requests.Session.post", return_value=resp):
            with self.assertRaises(OceansAPIError):
                client.login("user@ex.com", "pw")


class TestOceansClientMetadata(unittest.TestCase):
    def _client_with_token(self):
        from artwarp.oceans.api import OceansClient
        client = OceansClient.__new__(OceansClient)
        client._base_url = "https://example.com/api"
        client._token = "tok123"
        client.verbose = False
        client._username = None
        client._password = None
        client._initial_token = "tok123"
        import requests as _r
        client._session = _r.Session()
        return client

    def test_get_encounters_list(self):
        client = self._client_with_token()
        data = [{"id": "enc-1"}, {"id": "enc-2"}]
        resp = _mock_response(200, data)
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_encounters(species_id="sp-uuid")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "enc-1")

    def test_get_encounters_dict_response_wrapped(self):
        client = self._client_with_token()
        resp = _mock_response(200, {"id": "enc-single"})
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_encounters(encounter_id="enc-single")
        self.assertEqual(result, [{"id": "enc-single"}])

    def test_get_encounters_api_error(self):
        from artwarp.oceans.api import OceansAPIError
        client = self._client_with_token()
        resp = _mock_response(500)
        resp.text = "Server error"
        with patch.object(client._session, "get", return_value=resp):
            with self.assertRaises(OceansAPIError):
                client.get_encounters(species_id="sp")

    def test_get_recordings(self):
        client = self._client_with_token()
        data = [{"id": "rec-1"}]
        resp = _mock_response(200, data)
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_recordings(encounter_id="enc-1")
        self.assertEqual(result[0]["id"], "rec-1")

    def test_get_selections(self):
        client = self._client_with_token()
        data = [{"id": "sel-1", "selection_file_id": "fid-1"}]
        resp = _mock_response(200, data)
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_selections(recording_id="rec-1")
        self.assertEqual(result[0]["selection_file_id"], "fid-1")

    def test_401_triggers_reauth(self):
        """A 401 response should cause the client to re-authenticate."""
        from artwarp.oceans.api import OceansAPIError
        client = self._client_with_token()

        resp_401 = _mock_response(401)
        resp_401.text = "Unauthorized"

        # after re-auth attempt -> another 401 should propagate as OceansAPIError
        with patch.object(client._session, "get", return_value=resp_401):
            with patch.object(client, "_ensure_token", side_effect=lambda: None):
                # two consecutive 401s -> OceansAPIError on second attempt
                with self.assertRaises(OceansAPIError):
                    client.get_encounters(species_id="sp")


class TestOceansClientFileDownload(unittest.TestCase):
    def _client_with_token(self):
        from artwarp.oceans.api import OceansClient
        client = OceansClient.__new__(OceansClient)
        client._base_url = "https://example.com/api"
        client._token = "tok123"
        client.verbose = False
        client._username = None
        client._password = None
        client._initial_token = "tok123"
        import requests as _r
        client._session = _r.Session()
        return client

    def test_download_wav_success(self):
        client = self._client_with_token()
        wav = _make_wav_bytes()
        resp = _mock_response(
            200,
            content=wav,
            headers={
                "Content-Type": "audio/wav",
                "Content-Disposition": 'attachment; filename="test.wav"',
            },
        )
        with patch.object(client._session, "get", return_value=resp):
            filename, data = client.download_wav("fid-1")
        self.assertEqual(filename, "test.wav")
        self.assertEqual(data, wav)

    def test_download_wav_wrong_content_type(self):
        from artwarp.oceans.api import OceansAPIError
        client = self._client_with_token()
        resp = _mock_response(
            200,
            content=b"not a wav",
            headers={"Content-Type": "text/html"},
        )
        with patch.object(client._session, "get", return_value=resp):
            with self.assertRaises(OceansAPIError):
                client.download_wav("fid-bad")

    def test_download_spectrogram_success(self):
        client = self._client_with_token()
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # minimal PNG-like bytes
        resp = _mock_response(
            200,
            content=png,
            headers={
                "Content-Type": "image/png",
                "Content-Disposition": 'attachment; filename="spec.png"',
            },
        )
        with patch.object(client._session, "get", return_value=resp):
            filename, data = client.download_spectrogram("sel-1")
        self.assertEqual(filename, "spec.png")
        self.assertEqual(data, png)

    def test_download_spectrogram_not_found(self):
        from artwarp.oceans.api import OceansAPIError
        client = self._client_with_token()
        resp = _mock_response(404)
        resp.text = "Not found"
        with patch.object(client._session, "get", return_value=resp):
            with self.assertRaises(OceansAPIError):
                client.download_spectrogram("sel-missing")


# ---------------------------------------------------------------------------
# contours.py tests
# ---------------------------------------------------------------------------

class TestExtractContourFromWavBytes(unittest.TestCase):
    def test_basic_extraction_returns_array(self):
        from artwarp.oceans.contours import extract_contour_from_wav_bytes
        wav = _make_wav_bytes(num_samples=1024)
        contour = extract_contour_from_wav_bytes(wav, nperseg=64, peak_quantile=0.0)
        self.assertIsInstance(contour, np.ndarray)
        self.assertEqual(contour.ndim, 1)
        self.assertGreater(len(contour), 0)

    def test_contour_values_are_positive_frequencies(self):
        from artwarp.oceans.contours import extract_contour_from_wav_bytes
        wav = _make_wav_bytes(num_samples=1024)
        contour = extract_contour_from_wav_bytes(wav, nperseg=64, peak_quantile=0.0)
        self.assertTrue(np.all(contour >= 0))

    def test_denoising_quantile_applied(self):
        """With peak_quantile=0.9, output should be same dtype and valid."""
        from artwarp.oceans.contours import extract_contour_from_wav_bytes
        wav = _make_wav_bytes(num_samples=2048)
        c0 = extract_contour_from_wav_bytes(wav, nperseg=128, peak_quantile=0.0)
        c9 = extract_contour_from_wav_bytes(wav, nperseg=128, peak_quantile=0.9)
        self.assertEqual(len(c0), len(c9))
        self.assertEqual(c0.dtype, np.float64)

    def test_stereo_wav_converted_to_mono(self):
        """Stereo WAV should be converted to mono without error."""
        from artwarp.oceans.contours import extract_contour_from_wav_bytes

        buf = io.BytesIO()
        n = 512
        sr = 44100
        sig = (np.random.randn(n, 2) * 10000).astype(np.int16)
        data_bytes = sig.tobytes()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + len(data_bytes)))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<H", 1))  # PCM
        buf.write(struct.pack("<H", 2))  # stereo
        buf.write(struct.pack("<I", sr))
        buf.write(struct.pack("<I", sr * 4))
        buf.write(struct.pack("<H", 4))
        buf.write(struct.pack("<H", 16))
        buf.write(b"data")
        buf.write(struct.pack("<I", len(data_bytes)))
        buf.write(data_bytes)
        contour = extract_contour_from_wav_bytes(buf.getvalue(), nperseg=64, peak_quantile=0.0)
        self.assertGreater(len(contour), 0)

    def test_invalid_bytes_raise(self):
        from artwarp.oceans.contours import extract_contour_from_wav_bytes
        with self.assertRaises(ValueError):
            extract_contour_from_wav_bytes(b"not a wav file at all", peak_quantile=0.0)


class TestFetchContoursToDir(unittest.TestCase):
    """Test fetch_contours_to_dir with a fully mocked OceansClient."""

    def _make_mock_client(self) -> MagicMock:
        client = MagicMock()
        client.get_encounters.return_value = [{"id": "enc-1"}]
        client.get_recordings.return_value = [{"id": "rec-1"}]
        client.get_selections.return_value = [
            {"id": "sel-1", "selection_file_id": "fid-1"},
            {"id": "sel-2", "selection_file_id": "fid-2"},
        ]
        wav = _make_wav_bytes()
        client.download_wav.return_value = ("test.wav", wav)
        return client

    def test_writes_csvs(self):
        import tempfile
        from artwarp.oceans.contours import fetch_contours_to_dir
        client = self._make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            n = fetch_contours_to_dir(
                tmpdir,
                species_ids=["sp-1"],
                client=client,
                verbose=False,
            )
        self.assertEqual(n, 2)

    def test_max_per_species_respected(self):
        import tempfile
        from artwarp.oceans.contours import fetch_contours_to_dir
        client = self._make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            n = fetch_contours_to_dir(
                tmpdir,
                species_ids=["sp-1"],
                max_per_species=1,
                client=client,
                verbose=False,
            )
        self.assertEqual(n, 1)

    def test_skips_selection_without_file_id(self):
        import tempfile
        from artwarp.oceans.contours import fetch_contours_to_dir
        client = self._make_mock_client()
        client.get_selections.return_value = [
            {"id": "sel-no-file"},  # no selection_file_id
            {"id": "sel-ok", "selection_file_id": "fid-ok"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            n = fetch_contours_to_dir(
                tmpdir,
                species_ids=["sp-1"],
                client=client,
                verbose=False,
            )
        self.assertEqual(n, 1)

    def test_handles_api_error_gracefully(self):
        import tempfile
        from artwarp.oceans.api import OceansAPIError
        from artwarp.oceans.contours import fetch_contours_to_dir
        client = MagicMock()
        client.get_encounters.side_effect = OceansAPIError("Connection refused")
        with tempfile.TemporaryDirectory() as tmpdir:
            n = fetch_contours_to_dir(
                tmpdir, species_ids=["sp-fail"], client=client, verbose=False
            )
        self.assertEqual(n, 0)

    def test_creates_output_dir(self):
        import tempfile
        from artwarp.oceans.contours import fetch_contours_to_dir
        client = self._make_mock_client()
        client.get_selections.return_value = []
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = os.path.join(tmpdir, "new_subdir")
            fetch_contours_to_dir(
                outdir, species_ids=["sp-1"], client=client, verbose=False
            )
            # check inside the with-block -> tmpdir is deleted on exit
            self.assertTrue(os.path.isdir(outdir))

    def test_uses_default_species_when_none_given(self):
        import tempfile
        from artwarp.oceans.api import DEFAULT_SPECIES_IDS
        from artwarp.oceans.contours import fetch_contours_to_dir
        client = self._make_mock_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            fetch_contours_to_dir(tmpdir, client=client, verbose=False)
        # should have been called once per default species ID
        self.assertEqual(client.get_encounters.call_count, len(DEFAULT_SPECIES_IDS))


# ---------------------------------------------------------------------------
# cli.py tests — argument parser
# ---------------------------------------------------------------------------

class TestAddOceansParser(unittest.TestCase):
    def _root_parser(self):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        return parser, sub

    def test_parser_registered(self):
        from artwarp.oceans.cli import add_oceans_parser
        parser, sub = self._root_parser()
        add_oceans_parser(sub)
        args = parser.parse_args(["oceans", "fetch", "-o", "/tmp/out"])
        self.assertEqual(args.command, "oceans")
        self.assertEqual(args.oceans_command, "fetch")
        self.assertEqual(args.output_dir, "/tmp/out")

    def test_fetch_defaults(self):
        from artwarp.oceans.cli import add_oceans_parser
        parser, sub = self._root_parser()
        add_oceans_parser(sub)
        args = parser.parse_args(["oceans", "fetch", "-o", "/tmp/out"])
        self.assertIsNone(args.species_ids)
        self.assertIsNone(args.max_per_species)
        self.assertIsNone(args.nperseg)  # None = auto
        self.assertAlmostEqual(args.peak_quantile, 0.9)
        self.assertIsNone(args.freq_low)
        self.assertIsNone(args.freq_high)
        self.assertFalse(args.quiet)

    def test_fetch_with_all_options(self):
        from artwarp.oceans.cli import add_oceans_parser
        parser, sub = self._root_parser()
        add_oceans_parser(sub)
        args = parser.parse_args([
            "oceans", "fetch",
            "-o", "/tmp/out",
            "--species-id", "uuid-1",
            "--species-id", "uuid-2",
            "--max-per-species", "5",
            "--nperseg", "512",
            "--peak-quantile", "0.8",
            "--quiet",
        ])
        self.assertEqual(args.species_ids, ["uuid-1", "uuid-2"])
        self.assertEqual(args.max_per_species, 5)
        self.assertEqual(args.nperseg, 512)
        self.assertAlmostEqual(args.peak_quantile, 0.8)
        self.assertTrue(args.quiet)

    def test_count_subcommand_registered(self):
        from artwarp.oceans.cli import add_oceans_parser
        parser, sub = self._root_parser()
        add_oceans_parser(sub)
        args = parser.parse_args(["oceans", "count"])
        self.assertEqual(args.oceans_command, "count")
        self.assertIsNone(args.species_ids)


class TestCommandOceansFetch(unittest.TestCase):
    def _args(self, **kwargs):
        import argparse
        defaults = dict(
            oceans_command="fetch",
            output_dir="/tmp/test_out",
            species_ids=None,
            max_per_species=None,
            nperseg=None,
            peak_quantile=0.9,
            freq_low=None,
            freq_high=None,
            quiet=True,
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_calls_fetch_and_prints_summary(self, capsys=None):
        from artwarp.oceans.cli import command_oceans_fetch
        with patch("artwarp.oceans.contours.fetch_contours_to_dir", return_value=5) as mock_f:
            with patch("builtins.print"):  # suppress output
                command_oceans_fetch(self._args())
            mock_f.assert_called_once()

    def test_exits_1_on_zero_writes(self):
        from artwarp.oceans.cli import command_oceans_fetch
        with patch("artwarp.oceans.contours.fetch_contours_to_dir", return_value=0):
            with self.assertRaises(SystemExit) as cm:
                command_oceans_fetch(self._args())
            self.assertEqual(cm.exception.code, 1)

    def test_exits_1_on_value_error(self):
        from artwarp.oceans.cli import command_oceans_fetch
        with patch(
            "artwarp.oceans.contours.fetch_contours_to_dir",
            side_effect=ValueError("bad creds"),
        ):
            with self.assertRaises(SystemExit) as cm:
                command_oceans_fetch(self._args())
            self.assertEqual(cm.exception.code, 1)


class TestCommandOceansCount(unittest.TestCase):
    def _args(self, **kwargs):
        import argparse
        defaults = dict(oceans_command="count", species_ids=None)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_calls_count_and_prints(self):
        from artwarp.oceans.cli import command_oceans_count
        mock_result = {"total": 42}
        with patch(
            "artwarp.oceans.contours.count_available_selections",
            return_value=mock_result,
        ):
            with patch("builtins.print"):
                command_oceans_count(self._args())


class TestCommandOceansDispatch(unittest.TestCase):
    def test_no_subcommand_exits_1(self):
        import argparse
        from artwarp.oceans.cli import command_oceans
        args = argparse.Namespace(oceans_command=None)
        with self.assertRaises(SystemExit) as cm:
            command_oceans(args)
        self.assertEqual(cm.exception.code, 1)

    def test_unknown_subcommand_exits_1(self):
        import argparse
        from artwarp.oceans.cli import command_oceans
        args = argparse.Namespace(oceans_command="nonexistent")
        with self.assertRaises(SystemExit) as cm:
            command_oceans(args)
        self.assertEqual(cm.exception.code, 1)


# ---------------------------------------------------------------------------
# __init__.py / public API surface
# ---------------------------------------------------------------------------

class TestOceansPublicAPI(unittest.TestCase):
    def test_public_symbols_importable(self):
        from artwarp.oceans import (
            DEFAULT_SPECIES_IDS,
            OceansAPIError,
            OceansAuthError,
            OceansClient,
            fetch_contours_to_dir,
        )
        self.assertIsNotNone(OceansClient)
        self.assertIsNotNone(fetch_contours_to_dir)
        self.assertIsInstance(DEFAULT_SPECIES_IDS, list)
        self.assertGreater(len(DEFAULT_SPECIES_IDS), 0)

    def test_default_species_ids_are_uuids(self):
        from artwarp.oceans import DEFAULT_SPECIES_IDS
        import re
        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        for sid in DEFAULT_SPECIES_IDS:
            self.assertRegex(sid, uuid_re, f"{sid!r} is not a valid UUID")


if __name__ == "__main__":
    unittest.main()
