from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .abstract.interfaces import FeatureEngineerBase
from .models import KaggleDataSources, TrainingData

DEFAULT_FEATURE_COLUMNS = [
    "driver_encoded",
    "constructor_encoded",
    "grid",
    "circuit_encoded",
    "prev_points",
    "prev_wins",
    "driver_prev_podiums",
    "driver_prev_avg_finish",
    "constructor_prev_points",
    "constructor_prev_wins",
    "year",
    "round",
]

LOGGER = logging.getLogger(__name__)

PREDICTION_INPUT_COLUMNS = {
    "driverRef",
    "constructorRef",
    "circuitRef",
    "grid",
    "prev_points",
    "prev_wins",
    "driver_prev_podiums",
    "driver_prev_avg_finish",
    "constructor_prev_points",
    "constructor_prev_wins",
    "year",
    "round",
}


class FeatureEngineer(FeatureEngineerBase):
    """Handle feature engineering, encoding, and scaling for race prediction."""

    def __init__(self, feature_columns: Sequence[str] | None = None) -> None:
        self.feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
        self.driver_encoder = LabelEncoder()
        self.constructor_encoder = LabelEncoder()
        self.circuit_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def build_training_data(self, sources: KaggleDataSources) -> TrainingData:
        df = self._merge_sources(sources)
        df = self._engineer_features(df)
        df = self._encode_categoricals(df)
        df = self._finalize_numeric(df)

        df_model = df[self.feature_columns + ["target", "finish_position", "raceName"]].reset_index(drop=True)
        X = df_model[self.feature_columns].to_numpy()
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        y_win = df_model["target"].astype(int).to_numpy()
        y_position = df_model["finish_position"].astype(float).to_numpy()
        race_names = df_model["raceName"].astype(str).to_numpy()

        return TrainingData(
            frame=df_model,
            X_scaled=X_scaled,
            y_win=y_win,
            y_position=y_position,
            race_names=race_names,
        )

    def transform_prediction_inputs(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        encoded = frame.copy()
        encoded["driver_encoded"] = self._encode_with_new(self.driver_encoder, encoded["driverRef"], "driver")
        encoded["constructor_encoded"] = self._encode_with_new(
            self.constructor_encoder, encoded["constructorRef"], "constructor"
        )
        encoded["circuit_encoded"] = self._encode_with_new(self.circuit_encoder, encoded["circuitRef"], "circuit")

        numeric_cols = [
            col
            for col in self.feature_columns
            if col not in {"driver_encoded", "constructor_encoded", "circuit_encoded"}
        ]
        encoded[numeric_cols] = encoded[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        avg_zero_mask = encoded["driver_prev_avg_finish"] <= 0
        encoded.loc[avg_zero_mask, "driver_prev_avg_finish"] = encoded.loc[avg_zero_mask, "grid"].clip(lower=1)

        X = encoded[self.feature_columns].to_numpy()
        X_scaled = self.scaler.transform(X)
        return encoded, X_scaled

    def _merge_sources(self, sources: KaggleDataSources) -> pd.DataFrame:
        df = sources.results.merge(sources.races, on="raceId", suffixes=("", "_race"))
        df = df.merge(sources.drivers, on="driverId")
        df = df.merge(sources.constructors, on="constructorId")
        df = df.merge(sources.circuits, on="circuitId")
        if "raceName" not in df.columns and "name" in df.columns:
            df["raceName"] = df["name"]
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df.sort_values(by=["driverId", "year", "round"], inplace=True)
        df["finish_position"] = df["positionOrder"].astype(float)
        df["target"] = (df["finish_position"] == 1).astype(int)

        driver_group = df.groupby(["driverId", "year"], sort=False)
        df["prev_points"] = driver_group["points"].cumsum() - df["points"]

        win_flag = (df["positionOrder"] == 1).astype(int)
        df["prev_wins"] = win_flag.groupby([df["driverId"], df["year"]]).cumsum() - win_flag

        podium_flag = (df["positionOrder"] <= 3).astype(int)
        df["driver_prev_podiums"] = podium_flag.groupby([df["driverId"], df["year"]]).cumsum() - podium_flag

        driver_finish_cumsum = driver_group["finish_position"].cumsum() - df["finish_position"]
        driver_counts = driver_group.cumcount()
        df["driver_prev_avg_finish"] = driver_finish_cumsum / driver_counts.replace(0, np.nan)
        df.loc[driver_counts == 0, "driver_prev_avg_finish"] = df.loc[driver_counts == 0, "grid"]
        df["driver_prev_avg_finish"] = df["driver_prev_avg_finish"].fillna(df["grid"])

        constructor_points = (
            df.groupby(["constructorId", "year", "round"], sort=False)["points"]
            .sum()
            .rename("constructor_points")
        )
        constructor_wins = (
            win_flag.groupby([df["constructorId"], df["year"], df["round"]])
            .sum()
            .rename("constructor_wins")
        )
        constructor_race = pd.concat([constructor_points, constructor_wins], axis=1).reset_index()
        constructor_race.sort_values(by=["constructorId", "year", "round"], inplace=True)
        constructor_group = constructor_race.groupby(["constructorId", "year"], sort=False)
        constructor_race["constructor_prev_points"] = (
            constructor_group["constructor_points"].cumsum() - constructor_race["constructor_points"]
        )
        constructor_race["constructor_prev_wins"] = (
            constructor_group["constructor_wins"].cumsum() - constructor_race["constructor_wins"]
        )
        df = df.merge(
            constructor_race[
                ["constructorId", "year", "round", "constructor_prev_points", "constructor_prev_wins"]
            ],
            on=["constructorId", "year", "round"],
            how="left",
        )

        df["year"] = df["year"].astype(int)
        df["round"] = df["round"].astype(int)
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["driverRef"] = df["driverRef"].fillna("unknown")
        df["constructorRef"] = df["constructorRef"].fillna("unknown")
        df["circuitRef"] = df["circuitRef"].fillna("unknown")

        df["driver_encoded"] = self.driver_encoder.fit_transform(df["driverRef"])
        df["constructor_encoded"] = self.constructor_encoder.fit_transform(df["constructorRef"])
        df["circuit_encoded"] = self.circuit_encoder.fit_transform(df["circuitRef"])
        return df

    def _finalize_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            "grid",
            "prev_points",
            "prev_wins",
            "driver_prev_podiums",
            "driver_prev_avg_finish",
            "constructor_prev_points",
            "constructor_prev_wins",
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        avg_zero_mask = df["driver_prev_avg_finish"] <= 0
        df.loc[avg_zero_mask, "driver_prev_avg_finish"] = df.loc[avg_zero_mask, "grid"].clip(lower=1)
        return df

    @staticmethod
    def _encode_with_new(encoder: LabelEncoder, series: pd.Series, kind: str) -> np.ndarray:
        values = series.fillna("unknown").astype(str)
        missing_mask = ~np.isin(values, encoder.classes_)
        if missing_mask.any():
            new_classes = np.unique(values[missing_mask])
            encoder.classes_ = np.unique(np.concatenate([encoder.classes_, new_classes]))
            LOGGER.warning("Added new %s labels: %s", kind, ", ".join(new_classes))
        return encoder.transform(values)
