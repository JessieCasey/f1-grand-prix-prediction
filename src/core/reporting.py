from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .models import PredictorPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PredictorPaths.from_project_root(PROJECT_ROOT).prediction_path
DEFAULT_TOP_N = 5

REQUIRED_COLUMNS = {
    "season",
    "round",
    "raceName",
    "driverRef",
    "predicted_position",
    "predicted_rank",
    "predicted_top5",
    "win_probability",
}

DEFAULT_DISPLAY_COLUMNS = [
    "driverRef",
    "predicted_rank",
    "win_probability",
    "constructorRef",
]


@dataclass(frozen=True)
class ReportConfig:
    csv_path: Path = DEFAULT_RESULTS_PATH
    top_n: int = DEFAULT_TOP_N


class PredictionReporter:
    def __init__(self, config: ReportConfig) -> None:
        self.config = config

    def load_predictions(self) -> pd.DataFrame:
        path = Path(self.config.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Prediction CSV '{path}' not found.")
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(
                f"CSV '{path}' is missing required columns: {sorted(missing)}. "
                "Generate it with the latest predictor before running the reporter."
            )
        return df

    def summarize_predictions(self, frame: pd.DataFrame) -> str:
        df = frame.copy()

        races: list[str] = []
        show_cols = DEFAULT_DISPLAY_COLUMNS

        group_keys = ["season", "round", "raceName"]
        for (season, rnd, name), group in df.groupby(group_keys, sort=True):
            subset = group.nsmallest(self.config.top_n, "predicted_rank").sort_values("predicted_rank")
            available_cols = [col for col in show_cols if col in subset.columns]
            if not available_cols:
                continue
            formatted_subset = subset[available_cols].copy()
            if "predicted_position" in formatted_subset.columns:
                formatted_subset["predicted_position"] = formatted_subset["predicted_position"].round(2)
            header = f"{season} Round {rnd}: {name}"
            races.append(self._format_table(header, formatted_subset))

        if not races:
            return "No prediction rows available."

        summary_lines = [
            f"Total races summarised: {df['raceName'].nunique()}",
            f"Unique drivers: {df['driverRef'].nunique()}",
            "",
        ]
        summary_lines.extend(races)
        return "\n".join(summary_lines)

    def run(self) -> str:
        frame = self.load_predictions()
        return self.summarize_predictions(frame)

    @staticmethod
    def _format_table(title: str, table: pd.DataFrame) -> str:
        formatted = table.to_string(index=False)
        return f"{title}\n{formatted}\n"
