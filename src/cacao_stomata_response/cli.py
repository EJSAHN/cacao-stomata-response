from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cacao-stomata-response",
        description="Run the cacao stomatal response analysis pipeline and export results to Excel.",
    )
    parser.add_argument(
        "--stomata-file",
        required=True,
        help="Path to the stimulus-response workbook.",
    )
    parser.add_argument(
        "--leaf-file",
        required=False,
        default=None,
        help="Optional path to the developmental leaf workbook.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path for the output Excel workbook.",
    )
    parser.add_argument(
        "--control-label",
        default="Control",
        help="Exact strain label used for control cells in the stimulus workbook.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Covariance regularization used in Gaussian Wasserstein calculations.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_pipeline(
        stomata_file=Path(args.stomata_file),
        leaf_file=Path(args.leaf_file) if args.leaf_file else None,
        output_file=Path(args.output_file),
        control_label=args.control_label,
        regularization=args.regularization,
    )


if __name__ == "__main__":
    main()
