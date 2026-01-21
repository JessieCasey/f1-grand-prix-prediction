from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .feature_engineering import PREDICTION_INPUT_COLUMNS
from .abstract.interfaces import FeatureEngineerBase, RaceDataClient
from .models import (
    GlobalPositionModelConfig,
    GlobalPositionTuningSpace,
    GlobalWinModelConfig,
    GlobalWinTuningSpace,
    JolpicaNextRaceRequest,
    JolpicaRoundRequest,
    KaggleDataSources,
    KaggleDownloadConfig,
    PredictorPaths,
    RaceIdentifier,
    RacePositionModelConfig,
    RaceWinModelConfig,
    TuningConfig,
    TrainingConfig,
    TrainingData,
    TrainTestSplit,
)

LOGGER = logging.getLogger(__name__)

WinModel = LogisticRegression | HistGradientBoostingClassifier
PositionModel = RandomForestRegressor | HistGradientBoostingRegressor


class F1GrandPrixPredictor:
    def __init__(
        self,
        jolpica: RaceDataClient,
        feature_engineer: FeatureEngineerBase,
        *,
        paths: PredictorPaths | None = None,
    ) -> None:
        self.feature_engineer = feature_engineer
        self.jolpica = jolpica

        project_root = Path(__file__).resolve().parents[2]
        self.paths = paths or PredictorPaths.from_project_root(project_root)
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)

        self.data_sources: KaggleDataSources | None = None

        self.global_win_model: HistGradientBoostingClassifier | None = None
        self.global_position_model: HistGradientBoostingRegressor | None = None
        self.win_models: Dict[str, LogisticRegression] = {}
        self.position_models: Dict[str, RandomForestRegressor] = {}

        self.training_data: TrainingData | None = None
        self.split: TrainTestSplit | None = None

    def download_kaggle_data(self, config: KaggleDownloadConfig | None = None) -> None:
        config = config or KaggleDownloadConfig()
        required_paths = config.required_paths(self.paths.data_dir)
        if all(path.exists() for path in required_paths):
            return
        try:
            import subprocess

            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    config.dataset,
                    "--unzip",
                    "-p",
                    str(self.paths.data_dir),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.paths.data_dir),
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Kaggle CLI not available. Install it with `pip install kaggle`.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Kaggle download failed ({exc.returncode}): {exc.stderr.strip()}") from exc

    def load_data(self) -> KaggleDataSources:
        sources = KaggleDataSources(
            results=pd.read_csv(self.paths.data_dir / "results.csv"),
            races=pd.read_csv(self.paths.data_dir / "races.csv"),
            drivers=pd.read_csv(self.paths.data_dir / "drivers.csv").drop(columns=["url"], errors="ignore"),
            constructors=pd.read_csv(self.paths.data_dir / "constructors.csv").drop(columns=["url"], errors="ignore"),
            circuits=pd.read_csv(self.paths.data_dir / "circuits.csv").drop(columns=["url"], errors="ignore"),
        )
        self.data_sources = sources
        return sources

    def _ensure_data_loaded(self) -> None:
        if self.data_sources is None:
            raise RuntimeError("Call load_data() before using this method.")

    def _get_data_sources(self) -> KaggleDataSources:
        self._ensure_data_loaded()
        assert self.data_sources is not None
        return self.data_sources

    def _get_training_data(self) -> TrainingData:
        if self.training_data is None:
            raise RuntimeError("Call preprocess() before using training data.")
        return self.training_data

    def _ensure_models_trained(self) -> None:
        if self.global_win_model is None or self.global_position_model is None:
            raise RuntimeError("Models are not trained. Call train() first.")

    def _ensure_prediction_ready(self) -> None:
        self._ensure_models_trained()
        self._get_training_data()

    def verify_jolpica_availability(self, season: int, rnd: int | None = None, *, fail_fast: bool = False) -> bool:
        try:
            self.jolpica.verify_availability(season, rnd)
            LOGGER.info(
                "[jolpica] Connectivity check succeeded for season %s%s.",
                season,
                f" round {rnd}" if rnd else "",
            )
            return True
        except RuntimeError as exc:
            LOGGER.warning(
                "[jolpica] Unable to reach API for season %s%s: %s",
                season,
                f" round {rnd}" if rnd else "",
                exc,
            )
            if fail_fast:
                raise
            return False

    def preprocess(self) -> None:
        sources = self._get_data_sources()
        training_data = self.feature_engineer.build_training_data(sources)
        self.training_data = training_data
        self.split = self._build_train_test_split(training_data)

    def _ensure_training_ready(self) -> None:
        if self.training_data is None or self.split is None:
            raise RuntimeError("Call preprocess() before train().")

    def _get_split(self) -> TrainTestSplit:
        self._ensure_training_ready()
        assert self.split is not None
        return self.split

    @staticmethod
    def _build_train_test_split(training_data: TrainingData) -> TrainTestSplit:
        y_win = training_data.y_win
        stratify = y_win if np.unique(y_win).size > 1 and np.min(np.bincount(y_win)) >= 2 else None

        (
            X_train,
            X_test,
            y_train_win,
            y_test_win,
            y_train_position,
            y_test_position,
            train_race_names,
            test_race_names,
        ) = train_test_split(
            training_data.X_scaled,
            training_data.y_win,
            training_data.y_position,
            training_data.race_names,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
        return TrainTestSplit(
            X_train=X_train,
            X_test=X_test,
            y_train_win=y_train_win,
            y_test_win=y_test_win,
            y_train_position=y_train_position,
            y_test_position=y_test_position,
            train_race_names=train_race_names,
            test_race_names=test_race_names,
        )

    def _build_win_sample_weights(self) -> np.ndarray:
        split = self._get_split()
        classes = np.array([0, 1])
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=split.y_train_win)
        return np.where(split.y_train_win == 0, class_weights[0], class_weights[1])

    def train(self, config: TrainingConfig | None = None) -> None:
        self._ensure_training_ready()

        config = config or TrainingConfig()
        sample_weight = self._build_win_sample_weights()
        self.global_win_model = self._fit_global_win_model(
            model_config=config.global_win,
            tuning=config.tuning,
            tuning_space=config.global_win_space,
            sample_weight=sample_weight,
        )
        self._report_win_metrics(self.global_win_model)

        self.global_position_model = self._fit_global_position_model(
            model_config=config.global_position,
            tuning=config.tuning,
            tuning_space=config.global_position_space,
        )
        self._report_position_metrics(self.global_position_model)

        self._train_race_specific_models(
            win_config=config.race_win,
            position_config=config.race_position,
            min_samples=config.race_min_samples,
        )

    def _fit_global_win_model(
        self,
        *,
        model_config: GlobalWinModelConfig,
        tuning: TuningConfig,
        tuning_space: GlobalWinTuningSpace,
        sample_weight: np.ndarray,
    ) -> HistGradientBoostingClassifier:
        split = self._get_split()

        model = HistGradientBoostingClassifier(**model_config.to_model_kwargs())

        if tuning.enable:
            cv = StratifiedKFold(n_splits=tuning.cv_folds, shuffle=True, random_state=tuning.random_state)
            model = self._random_search_hist_gradient(
                model=model,
                param_distributions=tuning_space.to_param_distributions(),
                X=split.X_train,
                y=split.y_train_win,
                scoring="roc_auc",
                cv=cv,
                n_iter=tuning.n_iter,
                model_name="global_win",
                random_state=tuning.random_state,
                sample_weight=sample_weight,
            )

        model.fit(split.X_train, split.y_train_win, sample_weight=sample_weight)
        return model

    def _fit_global_position_model(
        self,
        *,
        model_config: GlobalPositionModelConfig,
        tuning: TuningConfig,
        tuning_space: GlobalPositionTuningSpace,
    ) -> HistGradientBoostingRegressor:
        split = self._get_split()

        model = HistGradientBoostingRegressor(**model_config.to_model_kwargs())

        if tuning.enable:
            cv_reg = KFold(n_splits=tuning.cv_folds, shuffle=True, random_state=tuning.random_state)
            model = self._random_search_hist_gradient(
                model=model,
                param_distributions=tuning_space.to_param_distributions(),
                X=split.X_train,
                y=split.y_train_position,
                scoring="neg_mean_absolute_error",
                cv=cv_reg,
                n_iter=tuning.n_iter,
                model_name="global_position",
                random_state=tuning.random_state,
            )

        model.fit(split.X_train, split.y_train_position)
        return model

    def _report_win_metrics(self, model: HistGradientBoostingClassifier) -> None:
        split = self._get_split()

        train_win_probs = model.predict_proba(split.X_train)[:, 1]
        train_win_pred = (train_win_probs >= 0.5).astype(int)
        LOGGER.info("Global win train accuracy: %.3f", accuracy_score(split.y_train_win, train_win_pred))
        try:
            train_auc = roc_auc_score(split.y_train_win, train_win_probs)
            LOGGER.info("Global win train ROC-AUC: %.3f", train_auc)
        except ValueError:
            LOGGER.info("Global win train ROC-AUC: unavailable (single class)")

        win_probs = model.predict_proba(split.X_test)[:, 1]
        win_pred = (win_probs >= 0.5).astype(int)
        LOGGER.info("Global win accuracy: %.3f", accuracy_score(split.y_test_win, win_pred))
        try:
            test_auc = roc_auc_score(split.y_test_win, win_probs)
            LOGGER.info("Global win ROC-AUC: %.3f", test_auc)
        except ValueError:
            LOGGER.info("Global win ROC-AUC: unavailable (single class)")
        LOGGER.info("Global win classification report:\n%s", classification_report(split.y_test_win, win_pred))

    def _report_position_metrics(self, model: HistGradientBoostingRegressor) -> None:
        split = self._get_split()
        pos_pred = model.predict(split.X_test)
        LOGGER.info("Global position MAE: %.2f", mean_absolute_error(split.y_test_position, pos_pred))

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
        LOGGER.info(
            "[tune][%s] best_score=%.4f params=%s",
            model_name,
            search.best_score_,
            search.best_params_,
        )
        return search.best_estimator_

    def _train_race_specific_models(
        self,
        win_config: RaceWinModelConfig,
        position_config: RacePositionModelConfig,
        *,
        min_samples: int,
    ) -> None:
        training_data = self._get_training_data()
        frame = training_data.frame
        X_full = training_data.X_scaled

        self.win_models.clear()
        self.position_models.clear()

        for race_name, subset in frame.groupby("raceName"):
            idx = subset.index.to_numpy()
            X_race = X_full[idx]
            y_win_race = subset["target"].to_numpy()
            y_pos_race = subset["finish_position"].to_numpy()

            if len(y_win_race) >= min_samples and np.unique(y_win_race).size > 1:
                win_model = LogisticRegression(**win_config.to_model_kwargs())
                win_model.fit(X_race, y_win_race)
                self.win_models[race_name] = win_model

            if len(y_pos_race) >= min_samples:
                pos_model = RandomForestRegressor(**position_config.to_model_kwargs())
                pos_model.fit(X_race, y_pos_race)
                self.position_models[race_name] = pos_model

        LOGGER.info(
            "Race-specific models trained: wins=%s positions=%s",
            len(self.win_models),
            len(self.position_models),
        )

    def _select_win_model(self, race_name: str) -> tuple[WinModel, str]:
        if race_name in self.win_models:
            return self.win_models[race_name], "race"
        assert self.global_win_model is not None
        return self.global_win_model, "global"

    def _select_position_model(self, race_name: str) -> tuple[PositionModel, str]:
        if race_name in self.position_models:
            return self.position_models[race_name], "race"
        assert self.global_position_model is not None
        return self.global_position_model, "global"

    def _compute_predictions(self, encoded_df: pd.DataFrame, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_win_prob = np.zeros(len(encoded_df))
        predicted_positions = np.zeros(len(encoded_df))

        for race_name, group in encoded_df.groupby("raceName"):
            idx = group.index.to_numpy()
            X_group = X_scaled[idx]

            win_model, win_source = self._select_win_model(race_name)
            probs = win_model.predict_proba(X_group)[:, 1]
            raw_win_prob[idx] = probs

            pos_model, pos_source = self._select_position_model(race_name)
            predicted_positions[idx] = pos_model.predict(X_group)

            LOGGER.info(
                "[predict][%s] win_model=%s position_model=%s samples=%s",
                race_name,
                win_source,
                pos_source,
                len(idx),
            )

        return raw_win_prob, predicted_positions

    @staticmethod
    def _normalize_win_probabilities(frame: pd.DataFrame, raw_win_prob: np.ndarray) -> None:
        frame["win_probability"] = 0.0
        for _, group in frame.groupby("raceName"):
            idx = group.index.to_numpy()
            probs = raw_win_prob[idx]
            total = probs.sum()
            if total > 0:
                frame.loc[idx, "win_probability"] = probs / total
            else:
                frame.loc[idx, "win_probability"] = 1.0 / len(idx)

    @staticmethod
    def _apply_rankings(frame: pd.DataFrame) -> None:
        frame["predicted_rank"] = frame.groupby("raceName")["predicted_position"].rank(method="first").astype(int)
        frame["predicted_top5"] = frame["predicted_rank"] <= 5

    @staticmethod
    def _log_top_predictions(results: pd.DataFrame) -> None:
        for race_name, group in results.groupby("raceName"):
            top = group.nsmallest(5, "predicted_rank")
            LOGGER.info("[predict][%s] Top 5 predicted finishers:", race_name)
            LOGGER.info(
                "\n%s",
                top[["driverRef", "predicted_position", "win_probability"]].to_string(index=False),
            )

    def predict_from_csv(
        self,
        csv_path: str | Path,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        self._ensure_prediction_ready()

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

        raw_win_prob, predicted_positions = self._compute_predictions(encoded_df, X_scaled)

        encoded_df["raw_win_probability"] = raw_win_prob
        encoded_df["predicted_position"] = predicted_positions

        self._normalize_win_probabilities(encoded_df, raw_win_prob)
        self._apply_rankings(encoded_df)

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
        output_path = Path(output_path or self.paths.prediction_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

        self._log_top_predictions(results)

        return results

    def build_inputs_csv_via_jolpica(
        self,
        season: int,
        rnd: int,
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        sources = self._get_data_sources()
        race = RaceIdentifier(season=season, round=rnd)
        try:
            final_df = self.jolpica.build_inputs_for_round(
                JolpicaRoundRequest(
                    race=race,
                    sources=sources,
                )
            )
            if final_df.empty:
                raise RuntimeError("No participants returned for requested round.")
        except RuntimeError as exc:
            LOGGER.warning(
                "[jolpica] build_inputs_for_round failed for season %s round %s: %s. Falling back to next race candidates.",
                season,
                rnd,
                exc,
            )
            final_df = self.jolpica.get_next_race_candidates(
                JolpicaNextRaceRequest(
                    season=season,
                    sources=sources,
                )
            )
            if final_df.empty:
                raise
            if int(final_df["round"].iloc[0]) != rnd:
                LOGGER.warning(
                    "[jolpica] Candidate data corresponds to round %s, not requested round %s.",
                    int(final_df["round"].iloc[0]),
                    rnd,
                )

        output_path = Path(output_csv or self.paths.race_inputs_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        race_label = final_df["raceName"].iloc[0] if not final_df.empty else "unknown"
        LOGGER.info(
            "[jolpica] Saved inputs for season %s round %s (%s) -> %s",
            season,
            rnd,
            race_label,
            output_path,
        )
        return final_df

    def get_next_race_candidates(
        self,
        season: int | None = None,
        *,
        output_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        sources = self._get_data_sources()
        if season is None:
            season = int(sources.races["year"].max())

        candidates = self.jolpica.get_next_race_candidates(
            JolpicaNextRaceRequest(
                season=season,
                sources=sources,
            )
        )

        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            candidates.to_csv(output_path, index=False)
            race_label = candidates["raceName"].iloc[0] if not candidates.empty else "unknown"
            LOGGER.info(
                "[jolpica] Saved next-race candidates for season %s (%s) -> %s",
                season,
                race_label,
                output_path,
            )

        return candidates
