import argparse
from pathlib import Path

from src.core.feature_engineering import FeatureEngineer
from src.core.jolpica import JolpicaClient
from src.core.predictor import F1GrandPrixPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the F1 predictor and generate predictions for a specific season/round."
    )
    parser.add_argument("--season", type=int, default=2025, help="Season to build inputs for (default: 2025).")
    parser.add_argument("--round", type=int, default=24, help="Round to build inputs for (default: 24).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictor = F1GrandPrixPredictor(JolpicaClient(), FeatureEngineer())
    predictor.download_kaggle_data()
    predictor.load_data()
    predictor.preprocess()
    predictor.verify_jolpica_availability(args.season, args.round)
    predictor.train()

    race_inputs_path = RESULTS_DIR / "race_inputs.csv"
    predictor.build_inputs_csv_via_jolpica(season=args.season, rnd=args.round, output_csv=str(race_inputs_path))

    prediction_path = RESULTS_DIR / "prediction_results.csv"
    results = predictor.predict_from_csv(str(race_inputs_path), output_path=str(prediction_path))
    print(results.head())


if __name__ == "__main__":
    main()
