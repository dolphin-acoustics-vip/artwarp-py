"""
CLI helpers for the OCEANS integration.

These functions are called by ``artwarp.cli.main`` when the user runs::

    artwarp-py oceans fetch ...
    artwarp-py oceans count ...

They are kept in this module so the rest of the OCEANS package does not need
to import argparse, and so the CLI module stays focused on args parsing.

@author: Pedro Gronda Garrigues
"""

import argparse
import sys

from artwarp.oceans.api import DEFAULT_SPECIES_IDS


def add_oceans_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:  # type: ignore[type-arg]
    """
    Register the ``oceans`` command group with an existing subparsers action.

    Subcommands added:
    - ``oceans fetch``  — download selections and extract contours to CSV
    - ``oceans count``  — count available selections without downloading

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers action from the root ``artwarp-py`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The ``oceans`` sub-parser (for reference or testing).
    """
    oceans_parser = subparsers.add_parser(
        "oceans",
        help="OCEANS data pipeline — fetch dolphin call data for training",
        description=(
            "Download selection WAV files from the OCEANS API, extract frequency\n"
            "contours, and save them as CSV files ready for 'artwarp-py train'.\n\n"
            "Credentials — set environment variables (never commit to source control):\n"
            "  export OCEAN_USERNAME='your@email.ac.uk'\n"
            "  export OCEAN_PASSWORD='your_password'\n"
            "  # or, with a pre-obtained token:\n"
            "  export OCEAN_ACCESS_TOKEN='eyJ...'\n\n"
            "Test server (no API-privileges required on your account):\n"
            "  export OCEAN_BASE_URL='https://rescomp-test-2.st-andrews.ac.uk/ocean/api'\n\n"
            "OCEANS is developed by James Sullivan:\n"
            "  https://github.com/dolphin-acoustics-vip/database-management-system"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    oceans_sub = oceans_parser.add_subparsers(dest="oceans_command", metavar="subcommand")

    # ---- oceans fetch ----
    fetch = oceans_sub.add_parser(
        "fetch",
        help="Download OCEANS selections and extract contours to CSV",
        description=(
            "Download selection WAV files from OCEANS, extract a frequency\n"
            "contour per WAV (spectrogram peak tracking), and write one CSV\n"
            "file per selection to OUTPUT_DIR.\n\n"
            "The output directory can then be passed directly to training:\n"
            "  artwarp-py train --input-dir OUTPUT_DIR --format csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fetch.add_argument(
        "-o",
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory to write contour CSV files (created if it does not exist).",
    )
    fetch.add_argument(
        "--species-id",
        action="append",
        dest="species_ids",
        metavar="UUID",
        help=(
            "OCEANS species UUID to fetch.  May be repeated for multiple species.\n"
            f"Default: {DEFAULT_SPECIES_IDS[0]!r} (and one more bottlenose dolphin ID)."
        ),
    )
    fetch.add_argument(
        "--max-per-species",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of contour CSVs to write per species (default: all).",
    )
    fetch.add_argument(
        "--nperseg",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Spectrogram segment length for contour extraction.\n"
            "Default: auto (targets ≈50 Hz/bin; recommended for OCEANS 500 kHz WAVs).\n"
            "Larger values give finer frequency resolution, coarser time resolution."
        ),
    )
    fetch.add_argument(
        "--peak-quantile",
        type=float,
        default=0.9,
        metavar="Q",
        help=(
            "Noise-suppression quantile [0.0–1.0] (default: 0.9).\n"
            "Frames with peak power below this quantile are replaced by the\n"
            "median peak frequency.  Set to 0.0 to disable."
        ),
    )
    fetch.add_argument(
        "--freq-low",
        type=float,
        default=None,
        metavar="HZ",
        help=(
            "Global lower frequency bound (Hz) for peak tracking.\n"
            "Default: use each selection's annotated low_frequency from OCEANS."
        ),
    )
    fetch.add_argument(
        "--freq-high",
        type=float,
        default=None,
        metavar="HZ",
        help=(
            "Global upper frequency bound (Hz) for peak tracking.\n"
            "Default: use each selection's annotated high_frequency from OCEANS."
        ),
    )
    fetch.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    # ---- oceans count ----
    count = oceans_sub.add_parser(
        "count",
        help="Count available OCEANS selections without downloading",
        description=(
            "Traverse the OCEANS hierarchy (species → encounters → recordings\n"
            "→ selections) and report how many selections with WAV files exist.\n"
            "Useful for estimating dataset size before running 'oceans fetch'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    count.add_argument(
        "--species-id",
        action="append",
        dest="species_ids",
        metavar="UUID",
        help="OCEANS species UUID to count.  May be repeated.  Default: built-in species IDs.",
    )

    return oceans_parser


def command_oceans_fetch(args: argparse.Namespace) -> None:
    """Execute ``artwarp-py oceans fetch``."""
    try:
        from artwarp.oceans.contours import fetch_contours_to_dir
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    species_ids = args.species_ids or None  # None → use DEFAULT_SPECIES_IDS inside function
    verbose = not args.quiet

    if verbose:
        print("OCEANS → artwarp-py contour fetch")
        print("===================================")
        print(f"Output directory : {args.output_dir}")
        if species_ids:
            for sid in species_ids:
                print(f"Species          : {sid}")
        else:
            print("Species          : default (bottlenose dolphin)")
        if args.max_per_species:
            print(f"Max per species  : {args.max_per_species}")
        nperseg_display = args.nperseg if args.nperseg is not None else "auto (≈50 Hz/bin)"
        print(f"Spectrogram nperseg : {nperseg_display}")
        print(f"Peak quantile       : {args.peak_quantile}")
        freq_lo_display = f"{args.freq_low} Hz" if args.freq_low is not None else "from OCEANS selection table"
        freq_hi_display = f"{args.freq_high} Hz" if args.freq_high is not None else "from OCEANS selection table"
        print(f"Freq low            : {freq_lo_display}")
        print(f"Freq high           : {freq_hi_display}")
        print()
        print("Credentials from environment (OCEAN_USERNAME / OCEAN_PASSWORD")
        print("or OCEAN_ACCESS_TOKEN).  Never stored by artwarp-py.")
        print()

    try:
        n = fetch_contours_to_dir(
            output_dir=args.output_dir,
            species_ids=species_ids,
            max_per_species=args.max_per_species,
            nperseg=args.nperseg,
            peak_quantile=args.peak_quantile,
            freq_low=args.freq_low,
            freq_high=args.freq_high,
            verbose=verbose,
        )
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nFetch cancelled by user.")
        sys.exit(1)

    if n == 0:
        print(
            "Warning: no contour CSV files were written. Check credentials, "
            "species IDs, and that selections have WAV files attached.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDone. Wrote {n} contour CSV(s) to '{args.output_dir}'.")
    print(f"Ready for training:")
    print(f"  artwarp-py train --input-dir {args.output_dir} --format csv")


def command_oceans_count(args: argparse.Namespace) -> None:
    """Execute ``artwarp-py oceans count``."""
    try:
        from artwarp.oceans.contours import count_available_selections
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    species_ids = args.species_ids or None

    print("OCEANS selection count")
    print("======================")
    if species_ids:
        for sid in species_ids:
            print(f"Species: {sid}")
    else:
        print("Species: default (bottlenose dolphin)")
    print()

    try:
        result = count_available_selections(species_ids=species_ids, verbose=True)
    except (ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)

    print(f"\nTotal selections with WAV files: {result.get('total', 0)}")


def command_oceans(args: argparse.Namespace) -> None:
    """Dispatch ``artwarp-py oceans <subcommand>``."""
    if not args.oceans_command:
        # no subcommand given -> print help
        print(
            "Usage: artwarp-py oceans <subcommand>\n\n"
            "Subcommands:\n"
            "  fetch   Download OCEANS selections and extract contours to CSV\n"
            "  count   Count available OCEANS selections without downloading\n\n"
            "Run 'artwarp-py oceans <subcommand> --help' for details."
        )
        sys.exit(1)

    if args.oceans_command == "fetch":
        command_oceans_fetch(args)
    elif args.oceans_command == "count":
        command_oceans_count(args)
    else:
        print(f"Unknown oceans subcommand: {args.oceans_command!r}", file=sys.stderr)
        sys.exit(1)
