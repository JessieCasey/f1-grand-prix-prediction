import argparse
import logging
from pathlib import Path

from src.core.app import PipelineConfig, PredictorPipeline
from src.core.models import RaceIdentifier

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEASON = 2025
DEFAULT_ROUND = 25

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the F1 predictor and generate predictions for a specific season/round."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help=f"Season to build inputs for (default: {DEFAULT_SEASON}).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=DEFAULT_ROUND,
        help=f"Round to build inputs for (default: {DEFAULT_ROUND}).",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    configure_logging()
    args = parse_args()
    LOGGER.info("Running F1 predictor pipeline for season %s round %s", args.season, args.round)
    config = PipelineConfig.from_project_root(
        RaceIdentifier(season=args.season, round=args.round),
        PROJECT_ROOT,
    )
    predictor = PredictorPipeline.with_default_predictor(config)
    predictor.run()


if __name__ == "__main__":
    main()
