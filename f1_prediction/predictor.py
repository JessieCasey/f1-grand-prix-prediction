from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .feature_engineering import (
    FeatureEngineer,
    PREDICTION_INPUT_COLUMNS,
    TrainingData,
)
from .jolpica import JolpicaClient


class F1GrandPrixPredictor:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        jolpica: JolpicaClient | None = None,
        feature_engineer: FeatureEngineer | None = None,
    ) -> None:
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.jolpica = jolpica or JolpicaClient()

        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data" / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.global_win_model: LogisticRegression | None = None
        self.global_position_model: RandomForestRegressor | None = None
        self.win_models: Dict[str, LogisticRegression] = {}
        self.position_models: Dict[str, RandomForestRegressor] = {}

        self.training_data: TrainingData | None = None
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train_win: np.ndarray | None = None
        self.y_test_win: np.ndarray | None = None
        self.y_train_position: np.ndarray | None = None
        self.y_test_position: np.ndarray | None = None
        self.train_race_names: np.ndarray | None = None
        self.test_race_names: np.ndarray | None = None

    def download_kaggle_data(self, kaggle_dataset: str = "rohanrao/formula-1-world-championship-1950-2020") -> None:
        required = {
            name: self.data_dir / name
            for name in ("results.csv", "races.csv", "drivers.csv", "constructors.csv", "circuits.csv")
        }
        if all(path.exists() for path in required.values()):
            return
        try:
            import subprocess

            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    kaggle_dataset,
                    "--unzip",
                    "-p",
                    str(self.data_dir),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.data_dir),
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Kaggle CLI not available. Install it with `pip install kaggle`.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Kaggle download failed ({exc.returncode}): {exc.stderr.strip()}") from exc

    def load_data(self) -> None:
        self.results = pd.read_csv(self.data_dir / "results.csv")
        self.races = pd.read_csv(self.data_dir / "races.csv")
        self.drivers = pd.read_csv(self.data_dir / "drivers.csv").drop(columns=["url"], errors="ignore")
        self.constructors = pd.read_csv(self.data_dir / "constructors.csv").drop(columns=["url"], errors="ignore")
        self.circuits = pd.read_csv(self.data_dir / "circuits.csv").drop(columns=["url"], errors="ignore")

    def preprocess(self) -> None:
        training_data = self.feature_engineer.build_training_data(
            self.results,
            self.races,
            self.drivers,
            self.constructors,
            self.circuits,
        )
        self.training_data = training_data

        y_win = training_data.y_win
        stratify = y_win if np.unique(y_win).size > 1 and np.min(np.bincount(y_win)) >= 2 else None

        (
            self.X_train,
            self.X_test,
            self.y_train_win,
            self.y_test_win,
            self.y_train_position,
            self.y_test_position,
            self.train_race_names,
            self.test_race_names,
        ) = train_test_split(
            training_data.X_scaled,
            training_data.y_win,
            training_data.y_position,
            training_data.race_names,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

    def train(self, max_iter: int = 1000) -> None:
        if any(
            attr is None
            for attr in (
                self.training_data,
                self.X_train,
                self.X_test,
                self.y_train_win,
                self.y_test_win,
                self.y_train_position,
                self.y_test_position,
            )
        ):
            raise RuntimeError("Call preprocess() before train().")

        classes = np.array([0, 1])
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train_win)
        sample_weight = np.where(self.y_train_win == 0, class_weights[0], class_weights[1])

        self.global_win_model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            max_depth=None,
            l2_regularization=1e-2,
            early_stopping=True,
            random_state=42,
        )
        self.global_win_model.fit(self.X_train, self.y_train_win, sample_weight=sample_weight)
        win_probs = self.global_win_model.predict_proba(self.X_test)[:, 1]
        win_pred = (win_probs >= 0.5).astype(int)
        print("Global win accuracy:", accuracy_score(self.y_test_win, win_pred))
        print(classification_report(self.y_test_win, win_pred))

        self.global_position_model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=63,
            max_depth=None,
            l2_regularization=1e-2,
            early_stopping=True,
            random_state=42,
        )
        self.global_position_model.fit(self.X_train, self.y_train_position)
        pos_pred = self.global_position_model.predict(self.X_test)
        print(f"Global position MAE: {mean_absolute_error(self.y_test_position, pos_pred):.2f}")

        self._train_race_specific_models(max_iter)

    def _train_race_specific_models(self, max_iter: int) -> None:
        assert self.training_data is not None
        frame = self.training_data.frame
        X_full = self.training_data.X_scaled

        self.win_models.clear()
        self.position_models.clear()

        min_samples = 30
        for race_name, subset in frame.groupby("raceName"):
            idx = subset.index.to_numpy()
            X_race = X_full[idx]
            y_win_race = subset["target"].to_numpy()
            y_pos_race = subset["finish_position"].to_numpy()

            if len(y_win_race) >= min_samples and np.unique(y_win_race).size > 1:
                win_model = LogisticRegression(max_iter=max_iter, class_weight="balanced")
                win_model.fit(X_race, y_win_race)
                self.win_models[race_name] = win_model

            if len(y_pos_race) >= min_samples:
                pos_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                pos_model.fit(X_race, y_pos_race)
                self.position_models[race_name] = pos_model

        print(
            f"Race-specific models trained: wins={len(self.win_models)} "
            f"positions={len(self.position_models)}"
        )

    def predict_from_csv(self, csv_path: str | Path, output_path: str = "prediction_results.csv") -> pd.DataFrame:
        if self.global_win_model is None or self.global_position_model is None:
            raise RuntimeError("Models are not trained. Call train() first.")

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Input CSV '{csv_path}' not found. Generate it via Jolpica:\n"
                "  predictor.build_inputs_csv_via_jolpica(..., output_csv='race_inputs.csv')"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Input CSV '{csv_path}' is empty.")

        df = pd.read_csv(path)
        missing = PREDICTION_INPUT_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in '{csv_path}': {sorted(missing)}. "
                "Regenerate inputs with build_inputs_csv_via_jolpica."
            )
        if "raceName" not in df.columns:
            raise ValueError("Prediction CSV must contain 'raceName' to select race-specific models.")

        race_summary = " ".join(f"({idx}, '{name}')" for idx, name in enumerate(df['raceName'].tolist()))
        print(f"[predict] raceName rows: {race_summary}")

        encoded_df, X_scaled = self.feature_engineer.transform_prediction_inputs(df)

        raw_win_prob = np.zeros(len(encoded_df))
        predicted_positions = np.zeros(len(encoded_df))

        for race_name, group in encoded_df.groupby("raceName"):
            idx = group.index.to_numpy()
            X_group = X_scaled[idx]

            if race_name in self.win_models:
                win_model = self.win_models[race_name]
                win_source = "race"
            else:
                win_model = self.global_win_model
                win_source = "global"
            probs = win_model.predict_proba(X_group)[:, 1]
            raw_win_prob[idx] = probs

            if race_name in self.position_models:
                pos_model = self.position_models[race_name]
                pos_source = "race"
            else:
                pos_model = self.global_position_model
                pos_source = "global"
            predicted_positions[idx] = pos_model.predict(X_group)

            print(f"[predict][{race_name}] win_model={win_source} position_model={pos_source} samples={len(idx)}")

        encoded_df["raw_win_probability"] = raw_win_prob
        encoded_df["predicted_position"] = predicted_positions

        encoded_df["win_probability"] = 0.0
        for race_name, group in encoded_df.groupby("raceName"):
            idx = group.index.to_numpy()
            probs = raw_win_prob[idx]
            total = probs.sum()
            if total > 0:
                encoded_df.loc[idx, "win_probability"] = probs / total
            else:
                encoded_df.loc[idx, "win_probability"] = 1.0 / len(idx)

        encoded_df["predicted_rank"] = encoded_df.groupby("raceName")["predicted_position"].rank(method="first").astype(int)
        encoded_df["predicted_top5"] = encoded_df["predicted_rank"] <= 5

        ordered_cols = [
            "season",
            "year",
            "round",
            "raceName",
            "driverRef",
            "driverId",
            "constructorRef",
            "constructorId",
            "circuitRef",
            "circuitId",
            "grid",
            "prev_points",
            "prev_wins",
            "driver_prev_podiums",
            "driver_prev_avg_finish",
            "constructor_prev_points",
            "constructor_prev_wins",
            "driver_age",
            "predicted_position",
            "predicted_rank",
            "predicted_top5",
            "win_probability",
            "raw_win_probability",
        ]
        available_cols = [col for col in ordered_cols if col in encoded_df.columns]

        results = encoded_df[available_cols].sort_values(["raceName", "predicted_rank", "predicted_position"])
        results.to_csv(output_path, index=False)

        for race_name, group in results.groupby("raceName"):
            top = group.nsmallest(5, "predicted_rank")
            print(f"[predict][{race_name}] Top 5 predicted finishers:")
            print(top[["driverRef", "predicted_position", "win_probability"]].to_string(index=False))

        return results

    # ---------------------------- Jolpica helper ----------------------------
    def build_inputs_csv_via_jolpica(self, season: int, rnd: int, output_csv: str) -> pd.DataFrame:
        try:
            final_df = self.jolpica.build_inputs_for_round(
                season,
                rnd,
                kaggle_drivers=self.drivers,
                kaggle_constructors=self.constructors,
                kaggle_races=self.races,
                kaggle_circuits=self.circuits,
            )
            if final_df.empty:
                raise RuntimeError("No participants returned for requested round.")
        except RuntimeError as exc:
            print(
                f"[jolpica] build_inputs_for_round failed for season {season} round {rnd}: {exc}. "
                "Falling back to next race candidates."
            )
            final_df = self.jolpica.get_next_race_candidates(
                season,
                kaggle_drivers=self.drivers,
                kaggle_constructors=self.constructors,
                kaggle_races=self.races,
                kaggle_circuits=self.circuits,
            )
            if final_df.empty:
                raise
            if int(final_df["round"].iloc[0]) != rnd:
                print(
                    f"[jolpica] Warning: candidate data corresponds to round {int(final_df['round'].iloc[0])}, "
                    f"not requested round {rnd}."
                )

        final_df.to_csv(output_csv, index=False)
        race_label = final_df["raceName"].iloc[0] if not final_df.empty else "unknown"
        print(f"[jolpica] Saved inputs for season {season} round {rnd} ({race_label}) -> {output_csv}")
        return final_df

    def get_next_race_candidates(
        self,
        season: int | None = None,
        *,
        output_csv: str | None = None,
    ) -> pd.DataFrame:
        """
        Build a candidate grid for the next upcoming race in the given season.

        If ``season`` is omitted, the latest season present in the Kaggle calendar is used.
        Optionally writes the result to ``output_csv``.
        """
        if season is None:
            if not hasattr(self, "races"):
                raise RuntimeError("Call load_data() before invoking get_next_race_candidates().")
            season = int(self.races["year"].max())

        candidates = self.jolpica.get_next_race_candidates(
            season,
            kaggle_drivers=getattr(self, "drivers", None),
            kaggle_constructors=getattr(self, "constructors", None),
            kaggle_races=getattr(self, "races", None),
            kaggle_circuits=getattr(self, "circuits", None),
        )

        if output_csv is not None:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            candidates.to_csv(output_csv, index=False)
            race_label = candidates["raceName"].iloc[0] if not candidates.empty else "unknown"
            print(f"[jolpica] Saved next-race candidates for season {season} ({race_label}) -> {output_csv}")

        return candidates
