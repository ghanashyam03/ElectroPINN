"""Unified CLI entry point for battery-pinn."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Dispatch to training, evaluation, or data generation subcommands."""
    parser = argparse.ArgumentParser(description="Battery PINN research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-data", help="Generate PyBaMM simulation dataset")
    sub.add_parser("train-baseline", help="Train baseline MLP")
    sub.add_parser("train-pinn", help="Train PINN model")
    sub.add_parser("evaluate", help="Evaluate and plot results")
    sub.add_parser("benchmark", help="Run inference benchmark")

    args = parser.parse_args()
    if args.command == "generate-data":
        from data.generate_dataset import main as run

        run()
    elif args.command == "train-baseline":
        from training.train_baseline import main as run

        run()
    elif args.command == "train-pinn":
        from training.train_pinn import main as run

        run()
    elif args.command == "evaluate":
        from evaluation.evaluate import main as run

        run()
    elif args.command == "benchmark":
        from evaluation.benchmark import main as run

        run()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
