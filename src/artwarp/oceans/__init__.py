"""
OCEANS integration for artwarp-py.

OCEANS (Odontocete Call Environment and Archival Network) is the University of
St Andrews dolphin acoustics database, developed by James Sullivan:
  https://github.com/dolphin-acoustics-vip/database-management-system

This subpackage bridges OCEANS and artwarp-py:
  - Authenticate with the OCEANS REST API
  - Browse encounters, recordings, and selections by species
  - Download selection WAV files and extract frequency contours
  - Save contours as CSV files ready for ARTwarp training

Quick start::

    from artwarp.oceans import OceansClient, fetch_contours_to_dir

    # credentials from OCEAN_USERNAME / OCEAN_PASSWORD env vars
    client = OceansClient()
    n = fetch_contours_to_dir("./contours_ocean", max_per_species=20)
    print(f"Wrote {n} contour CSV files — ready for artwarp-py train")

See docs/user/OCEANS.md for the full guide.

Credentials — never commit to source control:
    export OCEAN_USERNAME="your@email.ac.uk"
    export OCEAN_PASSWORD="your_password"
    # or, with a pre-obtained token:
    export OCEAN_ACCESS_TOKEN="eyJ..."

@author: Pedro Gronda Garrigues
"""

from artwarp.oceans.api import (
    DEFAULT_SPECIES_IDS,
    OceansClient,
    OceansAuthError,
    OceansAPIError,
)
from artwarp.oceans.contours import fetch_contours_to_dir

__all__ = [
    "OceansClient",
    "OceansAuthError",
    "OceansAPIError",
    "DEFAULT_SPECIES_IDS",
    "fetch_contours_to_dir",
]
