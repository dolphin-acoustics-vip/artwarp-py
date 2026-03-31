"""
Credential management for the OCEANS API.

Credentials must NEVER be stored in source code or committed to a repository.
Use environment variables exclusively, or let this module prompt interactively
when running in an interactive terminal.

Supported environment variables
--------------------------------
OCEAN_BASE_URL       Override the API base URL (default: production server).
                     Set to https://rescomp-test-2.st-andrews.ac.uk/ocean/api
                     to use the test server, which does not require API
                     privileges on your account.
OCEAN_ACCESS_TOKEN   A pre-obtained JWT bearer token. Takes priority over
                     username/password. Tokens expire after ~30 minutes; re-run
                     your script or re-export the variable when that happens.
OCEAN_USERNAME       Your OCEANS account e-mail address.
OCEAN_PASSWORD       Your OCEANS account password.

Example .env file (add to .gitignore — never commit):
    OCEAN_USERNAME=your@email.ac.uk
    OCEAN_PASSWORD=your_password

To request API access, contact the OCEANS administrators!

@author: Pedro Gronda Garrigues
"""

import getpass
import os
import sys
from typing import Optional, Tuple

# default production base URL; override with OCEAN_BASE_URL env var
DEFAULT_BASE_URL = "https://research.st-andrews.ac.uk/ocean/api"

# test server base URL (no API-privileges gate on accounts, but not sure how extensive the test server is...):
TEST_BASE_URL = "https://rescomp-test-2.st-andrews.ac.uk/ocean/api"


def get_base_url() -> str:
    """Return the OCEANS API base URL, stripping any trailing slash."""
    return os.environ.get("OCEAN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def get_token_from_env() -> Optional[str]:
    """Return OCEAN_ACCESS_TOKEN if set and non-empty, otherwise None."""
    token = os.environ.get("OCEAN_ACCESS_TOKEN", "").strip()
    return token if token else None


def get_credentials_from_env() -> Tuple[Optional[str], Optional[str]]:
    """Return (username, password) from environment variables, or (None, None)."""
    u = os.environ.get("OCEAN_USERNAME", "").strip() or None
    p = os.environ.get("OCEAN_PASSWORD", "").strip() or None
    return u, p


def prompt_credentials() -> Tuple[str, str]:
    """
    Interactively prompt for OCEANS username and password.

    Only called when stdin is a TTY (i.e. a real interactive terminal).
    The password is read with echo suppressed via getpass.

    Returns:
        (username, password) both as non-empty strings.

    Raises:
        ValueError: If stdin is not a TTY (non-interactive environment).
    """
    if not sys.stdin.isatty():
        raise ValueError(
            "OCEANS credentials not found in environment and stdin is not a TTY.\n"
            "Set environment variables before running:\n"
            "  export OCEAN_USERNAME='your@email.ac.uk'\n"
            "  export OCEAN_PASSWORD='your_password'\n"
            "Or set OCEAN_ACCESS_TOKEN with a pre-obtained bearer token.\n"
            "To use the test server (no API-privileges required):\n"
            f"  export OCEAN_BASE_URL='{TEST_BASE_URL}'"
        )

    print()
    print("  OCEANS credentials required.")
    print("  To avoid this prompt, set environment variables:")
    print("    export OCEAN_USERNAME='your@email.ac.uk'")
    print("    export OCEAN_PASSWORD='your_password'")
    print()
    username = input("  OCEANS username (e-mail): ").strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    password = getpass.getpass("  OCEANS password: ")
    if not password:
        raise ValueError("Password cannot be empty.")
    return username, password


def resolve_auth(
    username: Optional[str] = None,
    password: Optional[str] = None,
    access_token: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolve OCEANS authentication credentials from all available sources.

    Priority order (highest to lowest):
    1. ``access_token`` argument
    2. ``OCEAN_ACCESS_TOKEN`` environment variable
    3. ``username`` / ``password`` arguments
    4. ``OCEAN_USERNAME`` / ``OCEAN_PASSWORD`` environment variables
    5. Interactive prompt (only if stdin is a TTY)

    Returns:
        Tuple ``(token, username, password)`` where either *token* is set
        (and username/password are None) or *username* and *password* are set
        (and token is None).

    Raises:
        ValueError: If no credentials can be resolved.
    """
    # explicit token argument takes top priority
    if access_token and access_token.strip():
        return access_token.strip(), None, None

    # OCEAN_ACCESS_TOKEN env var next
    env_token = get_token_from_env()
    if env_token:
        return env_token, None, None

    # explicit username/password arguments
    if username and password:
        return None, username.strip(), password

    # env-var username/password
    env_u, env_p = get_credentials_from_env()
    if env_u and env_p:
        return None, env_u, env_p

    # last resort: interactive prompt
    prompted_u, prompted_p = prompt_credentials()
    return None, prompted_u, prompted_p
