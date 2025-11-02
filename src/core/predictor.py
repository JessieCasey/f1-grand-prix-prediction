from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .feature_engineering import (
    PREDICTION_INPUT_COLUMNS,
    FeatureEngineer,
    TrainingData,
)
from .jolpica import JolpicaClient

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "historical"
DEFAULT_RESULTS_DIR = DEFAULT_DATA_DIR.parent / "results"
DEFAULT_PREDICTION_PATH = DEFAULT_RESULTS_DIR / "prediction_results.csv"

DEFAULT_GLOBAL_WIN_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "l2_regularization": 1e-2,
    "early_stopping": True,
}

DEFAULT_GLOBAL_POSITION_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_leaf_nodes": 63,
    "max_depth": None,
    "l2_regularization": 1e-2,
    "early_stopping": True,
}

DEFAULT_RACE_WIN_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
}

DEFAULT_RACE_POSITION_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}

GLOBAL_WIN_PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "max_iter": [200, 300, 400, 500, 600, 800],
    "max_leaf_nodes": [15, 31, 63, 95, 127],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [10, 20, 30, 40],
    "l2_regularization": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    "early_stopping": [True, False],
}

GLOBAL_POSITION_PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "max_iter": [300, 400, 500, 650, 800],
    "max_leaf_nodes": [31, 63, 95, 127, 191],
    "max_depth": [5, 7, 9, None],
    "min_samples_leaf": [5, 10, 20, 35],
    "l2_regularization": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    "early_stopping": [True, False],
}


class F1GrandPrixPredictor:
    def __init__(
        self,
        jolpica: JolpicaClient,
        feature_engineer: FeatureEngineer,
    ) -> None:
        self.feature_engineer = feature_engineer
        self.jolpica = jolpica

        self.data_dir = DEFAULT_DATA_DIR
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

    def verify_jolpica_availability(self, season: int, rnd: int | None = None, *, fail_fast: bool = False) -> bool:
        try:
            self.jolpica.verify_availability(season, rnd)
            print(f"[jolpica] Connectivity check succeeded for season {season}{f' round {rnd}' if rnd else ''}.")
            return True
        except RuntimeError as exc:
            print(
                f"[jolpica] Warning: unable to reach API for season {season}"
                f"{f' round {rnd}' if rnd else ''}: {exc}"
            )
            if fail_fast:
                raise
            return False

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

    def train(
        self,
        max_iter: int = 1000,
        *,
        global_win_params: Dict[str, Any] | None = None,
        global_position_params: Dict[str, Any] | None = None,
        race_win_params: Dict[str, Any] | None = None,
        race_position_params: Dict[str, Any] | None = None,
        tune_global: bool = False,
        tuning_iter: int = 20,
        cv_folds: int = 3,
        race_min_samples: int = 30,
        random_state: int = 42,
    ) -> None:
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
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train_win)  # type: ignore[arg-type]
        sample_weight = np.where(self.y_train_win == 0, class_weights[0], class_weights[1])

        win_params = DEFAULT_GLOBAL_WIN_PARAMS.copy()
        win_params.setdefault("random_state", random_state)
        if global_win_params:
            win_params.update(global_win_params)

        self.global_win_model = HistGradientBoostingClassifier(**win_params)

        if tune_global:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            self.global_win_model = self._random_search_hist_gradient(
                model=self.global_win_model,
                param_distributions=GLOBAL_WIN_PARAM_DISTRIBUTIONS,
                X=self.X_train,
                y=self.y_train_win,
                scoring="roc_auc",
                cv=cv,
                n_iter=tuning_iter,
                model_name="global_win",
                random_state=random_state,
                sample_weight=sample_weight,
            )

        self.global_win_model.fit(self.X_train, self.y_train_win, sample_weight=sample_weight)
        train_win_probs = self.global_win_model.predict_proba(self.X_train)[:, 1]
        train_win_pred = (train_win_probs >= 0.5).astype(int)
        print(f"Global win train accuracy: {accuracy_score(self.y_train_win, train_win_pred):.3f}")
        try:
            train_auc = roc_auc_score(self.y_train_win, train_win_probs)
            print(f"Global win train ROC-AUC: {train_auc:.3f}")
        except ValueError:
            pass

        win_probs = self.global_win_model.predict_proba(self.X_test)[:, 1]
        win_pred = (win_probs >= 0.5).astype(int)
        print("Global win accuracy:", accuracy_score(self.y_test_win, win_pred))
        try:
            test_auc = roc_auc_score(self.y_test_win, win_probs)
            print(f"Global win ROC-AUC: {test_auc:.3f}")
        except ValueError:
            pass
        print(classification_report(self.y_test_win, win_pred))

        pos_params = DEFAULT_GLOBAL_POSITION_PARAMS.copy()
        pos_params.setdefault("random_state", random_state)
        if global_position_params:
            pos_params.update(global_position_params)

        self.global_position_model = HistGradientBoostingRegressor(**pos_params)

        if tune_global:
            cv_reg = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            self.global_position_model = self._random_search_hist_gradient(
                model=self.global_position_model,
                param_distributions=GLOBAL_POSITION_PARAM_DISTRIBUTIONS,
                X=self.X_train,
                y=self.y_train_position,
                scoring="neg_mean_absolute_error",
                cv=cv_reg,
                n_iter=tuning_iter,
                model_name="global_position",
                random_state=random_state,
            )

        self.global_position_model.fit(self.X_train, self.y_train_position)
        pos_pred = self.global_position_model.predict(self.X_test)
        print(f"Global position MAE: {mean_absolute_error(self.y_test_position, pos_pred):.2f}")

        race_win_kwargs = DEFAULT_RACE_WIN_PARAMS.copy()
        race_win_kwargs["max_iter"] = max_iter
        if race_win_params:
            race_win_kwargs.update(race_win_params)

        race_pos_kwargs = DEFAULT_RACE_POSITION_PARAMS.copy()
        if race_position_params:
            race_pos_kwargs.update(race_position_params)

        self._train_race_specific_models(
            race_win_kwargs,
            race_pos_kwargs,
            min_samples=race_min_samples,
        )

    def _random_search_hist_gradient(
        self,
        *,
        model: HistGradientBoostingClassifier | HistGradientBoostingRegressor,
        param_distributions: Mapping[str, Sequence[Any]],
        X: np.ndarray,
        y: np.ndarray,
        scoring: str,
        cv,
        n_iter: int,
        model_name: str,
        random_state: int,
        sample_weight: np.ndarray | None = None,
    ) -> HistGradientBoostingClassifier | HistGradientBoostingRegressor:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        search.fit(X, y, **fit_kwargs)
        print(f"[tune][{model_name}] best_score={search.best_score_:.4f} params={search.best_params_}")
        return search.best_estimator_

    def _train_race_specific_models(
        self,
        win_params: Mapping[str, Any],
        position_params: Mapping[str, Any],
        *,
        min_samples: int,
    ) -> None:
        assert self.training_data is not None
        frame = self.training_data.frame
        X_full = self.training_data.X_scaled

        self.win_models.clear()
        self.position_models.clear()

        for race_name, subset in frame.groupby("raceName"):
            idx = subset.index.to_numpy()
            X_race = X_full[idx]
            y_win_race = subset["target"].to_numpy()
            y_pos_race = subset["finish_position"].to_numpy()

            if len(y_win_race) >= min_samples and np.unique(y_win_race).size > 1:
                win_model = LogisticRegression(**dict(win_params))
                win_model.fit(X_race, y_win_race)
                self.win_models[race_name] = win_model

            if len(y_pos_race) >= min_samples:
                pos_model = RandomForestRegressor(**dict(position_params))
                pos_model.fit(X_race, y_pos_race)
                self.position_models[race_name] = pos_model

        print(
            f"Race-specific models trained: wins={len(self.win_models)} "
            f"positions={len(self.position_models)}"
        )

    def predict_from_csv(
        self,
        csv_path: str | Path,
        output_path: str | Path = DEFAULT_PREDICTION_PATH,
    ) -> pd.DataFrame:
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
        for _, group in encoded_df.groupby("raceName"):
            idx = group.index.to_numpy()
            probs = raw_win_prob[idx]
            total = probs.sum()
            if total > 0:
                encoded_df.loc[idx, "win_probability"] = probs / total
            else:
                encoded_df.loc[idx, "win_probability"] = 1.0 / len(idx)

        encoded_df["predicted_rank"] = encoded_df.groupby("raceName")["predicted_position"].rank(method="first").astype(
            int
        )
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
            "predicted_position",
            "predicted_rank",
            "predicted_top5",
            "win_probability",
            "raw_win_probability",
        ]
        available_cols = [col for col in ordered_cols if col in encoded_df.columns]

        results = encoded_df[available_cols].sort_values(["raceName", "predicted_rank", "predicted_position"])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

        for race_name, group in results.groupby("raceName"):
            top = group.nsmallest(5, "predicted_rank")
            print(f"[predict][{race_name}] Top 5 predicted finishers:")
            print(top[["driverRef", "predicted_position", "win_probability"]].to_string(index=False))

        return results

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
        if season is None:
            if not hasattr(self, "races"):
                raise RuntimeError("Call load_data() before invoking get_next_race_candidates().")
            season = int(self.races["year"].max())

        candidates = self.jolpica.get_next_race_candidates(
            season,
            kaggle_drivers=self.drivers,
            kaggle_constructors=self.constructors,
            kaggle_races=self.races,
            kaggle_circuits=self.circuits,
        )

        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            candidates.to_csv(output_path, index=False)
            race_label = candidates["raceName"].iloc[0] if not candidates.empty else "unknown"
            print(f"[jolpica] Saved next-race candidates for season {season} ({race_label}) -> {output_csv}")

        return candidates
