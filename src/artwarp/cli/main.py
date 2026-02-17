"""
Command-line interface for ARTwarp-py.

Provides commands for:
- Training networks on contour data
- Predicting categories for new data
- Exporting results and visualizations
- Generating visualization reports

@author: Pedro Gronda Garrigues
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np

from artwarp import ARTwarp, load_contours
from artwarp.core.network import TrainingResults
from artwarp.io.exporters import (
    export_results,
    export_reference_contours,
    export_category_assignments,
    load_results,
)
from artwarp.visualization import create_results_report
from artwarp.utils.resample import resample_contours


def command_train(args: argparse.Namespace) -> None:
    """Execute the train command."""
    print(f"Loading contours from: {args.input_dir}")

    need_tempres = getattr(args, "resample", False)
    try:
        result = load_contours(
            args.input_dir,
            file_format=args.format,
            frequency_column=args.freq_column,
            return_tempres=need_tempres,
        )
        if need_tempres:
            contours, names, tempres_list = result
            default_tr = getattr(args, "tempres", 0.01)
            tempres_list = [t if t is not None else default_tr for t in tempres_list]
        else:
            contours, names = result
    except Exception as e:
        print(f"Error loading contours: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(contours)} contours")

    if getattr(args, "resample", False):
        sample_interval = getattr(args, "sample_interval", 0.02)
        contours = resample_contours(contours, tempres_list, sample_interval)
        print(f"Resampled contours to {sample_interval}s interval")

    # create network
    network = ARTwarp(
        vigilance=args.vigilance,
        learning_rate=args.learning_rate,
        bias=args.bias,
        max_categories=args.max_categories,
        max_iterations=args.max_iterations,
        warp_factor_level=args.warp_factor,
        random_seed=args.seed,
        verbose=not args.quiet,
    )

    # train
    print("\nTraining network...")
    results = network.fit(contours, names)

    # save results
    if args.output:
        print(f"\nSaving results to: {args.output}")
        export_results(results, args.output, contour_names=names)

    # export ref contours if requested
    if args.export_refs:
        ref_dir = Path(args.output).parent / "reference_contours"
        print(f"Exporting reference contours to: {ref_dir}")
        export_reference_contours(results.weight_matrix, str(ref_dir))

    # export category assignments if requested
    if args.export_categories:
        cat_file = Path(args.output).parent / "category_assignments.csv"
        print(f"Exporting category assignments to: {cat_file}")
        export_category_assignments(results.categories, results.matches, names, str(cat_file))

    # print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total contours:     {len(contours)}")
    print(f"Categories created: {results.num_categories}")
    print(f"Iterations:         {results.num_iterations}")
    print(f"Converged:          {'Yes' if results.converged else 'No'}")
    print(f"Training time:      {results.training_time:.2f} seconds")
    print(f"Uncategorized:      {results.get_uncategorized_count()}")
    print("\nCategory sizes:")
    for cat, size in sorted(results.get_category_sizes().items()):
        print(f"  Category {cat}: {size} contours")


def command_predict(args: argparse.Namespace) -> None:
    """Execute the predict command."""
    # load trained model
    print(f"Loading trained model from: {args.model}")
    try:
        model_data = load_results(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # load contours to predict
    print(f"Loading contours from: {args.input_dir}")
    try:
        contours, names = load_contours(
            args.input_dir, file_format=args.format, frequency_column=args.freq_column
        )
    except Exception as e:
        print(f"Error loading contours: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(contours)} contours")

    # create network + set weights
    network = ARTwarp(verbose=not args.quiet)
    network.weight_matrix = model_data["weight_matrix"]
    network.num_categories = model_data["num_categories"]
    network.max_features = model_data["weight_matrix"].shape[0]

    # predict
    print("\nPredicting categories...")
    categories, matches = network.predict(contours)

    # export results
    if args.output:
        print(f"Saving predictions to: {args.output}")
        export_category_assignments(categories, matches, names, args.output)

    # print summary
    print("\nPrediction Summary:")
    print(f"Predicted: {len(contours)} contours")
    print(f"Categorized: {len(categories[~np.isnan(categories)])}")
    print(f"Uncategorized: {int(np.sum(np.isnan(categories)))}")


def command_export(args: argparse.Namespace) -> None:
    """Execute the export command."""
    # load results
    print(f"Loading results from: {args.results}")
    try:
        data = load_results(args.results)
    except Exception as e:
        print(f"Error loading results: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # export ref contours
    if args.export_type in ["all", "references"]:
        ref_dir = output_dir / "reference_contours"
        print(f"Exporting reference contours to: {ref_dir}")
        export_reference_contours(data["weight_matrix"], str(ref_dir))

    # export category assignments
    if args.export_type in ["all", "categories"]:
        if "contour_names" in data:
            cat_file = output_dir / "category_assignments.csv"
            print(f"Exporting category assignments to: {cat_file}")
            export_category_assignments(
                data["categories"], data["matches"], data["contour_names"], str(cat_file)
            )
        else:
            print("Warning: No contour names in results, skipping category export")

    print("Export complete!")


def command_plot(args: argparse.Namespace) -> None:
    """Execute the plot command: generate visualization report from results."""
    print(f"Loading results from: {args.results}")
    try:
        data = load_results(args.results)
    except Exception as e:
        print(f"Error loading results: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading contours from: {args.input_dir}")
    try:
        contours, names = load_contours(
            args.input_dir, file_format=args.format, frequency_column=args.freq_column
        )
    except Exception as e:
        print(f"Error loading contours: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(contours)} contours")

    results = TrainingResults(
        categories=data["categories"],
        matches=data["matches"],
        weight_matrix=data["weight_matrix"],
        num_categories=data["num_categories"],
        num_iterations=data["num_iterations"],
        converged=data["converged"],
        iteration_history=data["iteration_history"],
        training_time=data["training_time"],
    )

    output_dir = args.output_dir
    print(f"\nGenerating visualization report in: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = create_results_report(
        results, contours, contour_names=names, output_dir=output_dir, dpi=args.dpi
    )
    print(f"Report generated: {len(paths)} figures saved to {output_dir}")
    for name, path in sorted(paths.items()):
        print(f"  {name}: {path}")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="ARTwarp-py: High-performance bioacoustic signal categorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train command
    train_parser = subparsers.add_parser("train", help="Train ARTwarp network")
    train_parser.add_argument(
        "-i", "--input-dir", required=True, help="Directory containing contour files"
    )
    train_parser.add_argument(
        "-o", "--output", required=True, help="Output file for trained model (.pkl)"
    )
    train_parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "ctr", "csv", "txt"],
        help="Input file format (default: auto)",
    )
    train_parser.add_argument(
        "--freq-column",
        type=int,
        default=0,
        help="Frequency column index for CSV/TXT files (default: 0)",
    )
    train_parser.add_argument(
        "--vigilance", type=float, default=85.0, help="Vigilance threshold (1-99, default: 85)"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate (0-1, default: 0.1)"
    )
    train_parser.add_argument(
        "--bias", type=float, default=0.0, help="Activation bias (0-1, default: 0.0)"
    )
    train_parser.add_argument(
        "--max-categories", type=int, default=50, help="Maximum number of categories (default: 50)"
    )
    train_parser.add_argument(
        "--max-iterations", type=int, default=50, help="Maximum number of iterations (default: 50)"
    )
    train_parser.add_argument(
        "--warp-factor", type=int, default=3, help="Maximum DTW warping factor (default: 3)"
    )
    train_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    train_parser.add_argument(
        "--export-refs", action="store_true", help="Export reference contours to CSV files"
    )
    train_parser.add_argument(
        "--export-categories", action="store_true", help="Export category assignments to CSV"
    )
    train_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    train_parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample contours to a uniform temporal resolution before training (like MATLAB resample option)",
    )
    train_parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.02,
        metavar="SEC",
        help="Target sampling interval in seconds when --resample (default: 0.02)",
    )
    train_parser.add_argument(
        "--tempres",
        type=float,
        default=0.01,
        metavar="SEC",
        help="Default temporal resolution (sec/point) for contours that do not provide it, when --resample (default: 0.01)",
    )

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Predict categories for new data")
    predict_parser.add_argument("-m", "--model", required=True, help="Path to trained model (.pkl)")
    predict_parser.add_argument(
        "-i", "--input-dir", required=True, help="Directory containing contour files to predict"
    )
    predict_parser.add_argument(
        "-o", "--output", required=True, help="Output CSV file for predictions"
    )
    predict_parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "ctr", "csv", "txt"],
        help="Input file format (default: auto)",
    )
    predict_parser.add_argument(
        "--freq-column",
        type=int,
        default=0,
        help="Frequency column index for CSV/TXT files (default: 0)",
    )
    predict_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress output"
    )

    # export command
    export_parser = subparsers.add_parser("export", help="Export results to various formats")
    export_parser.add_argument("-r", "--results", required=True, help="Path to results file (.pkl)")
    export_parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for exported files"
    )
    export_parser.add_argument(
        "--export-type",
        default="all",
        choices=["all", "references", "categories"],
        help="What to export (default: all)",
    )

    # plot command
    plot_parser = subparsers.add_parser(
        "plot", help="Generate visualization report from training results"
    )
    plot_parser.add_argument(
        "-r", "--results", required=True, help="Path to results file (.pkl from train)"
    )
    plot_parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Directory containing contour files (same as used for train)",
    )
    plot_parser.add_argument(
        "-o",
        "--output-dir",
        default="./report",
        help="Output directory for figures (default: ./report)",
    )
    plot_parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "ctr", "csv", "txt"],
        help="Input file format (default: auto)",
    )
    plot_parser.add_argument(
        "--freq-column",
        type=int,
        default=0,
        help="Frequency column index for CSV/TXT files (default: 0)",
    )
    plot_parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for saved figures (default: 300)"
    )

    return parser


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "train":
            command_train(args)
        elif args.command == "predict":
            command_predict(args)
        elif args.command == "export":
            command_export(args)
        elif args.command == "plot":
            command_plot(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
