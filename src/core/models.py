from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictorPaths:
    data_dir: Path
    results_dir: Path
    prediction_path: Path
    race_inputs_path: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PredictorPaths":
        data_dir = project_root / "data" / "historical"
        results_dir = project_root / "data" / "results"
        return cls(
            data_dir=data_dir,
            results_dir=results_dir,
            prediction_path=results_dir / "prediction_results.csv",
            race_inputs_path=results_dir / "race_inputs.csv",
        )


@dataclass(frozen=True)
class KaggleDownloadConfig:
    dataset: str = "rohanrao/formula-1-world-championship-1950-2020"
    required_files: tuple[str, ...] = (
        "results.csv",
        "races.csv",
        "drivers.csv",
        "constructors.csv",
        "circuits.csv",
    )

    def required_paths(self, data_dir: Path) -> tuple[Path, ...]:
        return tuple(data_dir / name for name in self.required_files)


@dataclass(frozen=True)
class KaggleDataSources:
    results: pd.DataFrame
    races: pd.DataFrame
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    circuits: pd.DataFrame


@dataclass(frozen=True)
class TrainingData:
    frame: pd.DataFrame
    X_scaled: np.ndarray
    y_win: np.ndarray
    y_position: np.ndarray
    race_names: np.ndarray


@dataclass(frozen=True)
class TrainTestSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train_win: np.ndarray
    y_test_win: np.ndarray
    y_train_position: np.ndarray
    y_test_position: np.ndarray
    train_race_names: np.ndarray
    test_race_names: np.ndarray


@dataclass(frozen=True)
class RaceIdentifier:
    season: int
    round: int


@dataclass(frozen=True)
class JolpicaRoundRequest:
    race: RaceIdentifier
    sources: KaggleDataSources


@dataclass(frozen=True)
class JolpicaNextRaceRequest:
    season: int
    sources: KaggleDataSources


@dataclass(frozen=True)
class GlobalWinModelConfig:
    learning_rate: float = 0.05
    max_iter: int = 400
    max_leaf_nodes: int = 31
    max_depth: int | None = None
    l2_regularization: float = 1e-2
    early_stopping: bool = True
    random_state: int = 42

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "l2_regularization": self.l2_regularization,
            "early_stopping": self.early_stopping,
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class GlobalPositionModelConfig:
    learning_rate: float = 0.05
    max_iter: int = 400
    max_leaf_nodes: int = 63
    max_depth: int | None = None
    l2_regularization: float = 1e-2
    early_stopping: bool = True
    random_state: int = 42

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "l2_regularization": self.l2_regularization,
            "early_stopping": self.early_stopping,
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class RaceWinModelConfig:
    max_iter: int = 1000
    class_weight: str = "balanced"
    solver: str = "lbfgs"

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "solver": self.solver,
        }


@dataclass(frozen=True)
class RacePositionModelConfig:
    n_estimators: int = 300
    random_state: int = 42
    n_jobs: int = -1

    def to_model_kwargs(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass(frozen=True)
class TuningConfig:
    enable: bool = False
    n_iter: int = 20
    cv_folds: int = 3
    random_state: int = 42


@dataclass(frozen=True)
class GlobalWinTuningSpace:
    learning_rate: Sequence[float] = (0.01, 0.02, 0.03, 0.05, 0.08, 0.1)
    max_iter: Sequence[int] = (200, 300, 400, 500, 600, 800)
    max_leaf_nodes: Sequence[int] = (15, 31, 63, 95, 127)
    max_depth: Sequence[int | None] = (3, 5, 7, None)
    min_samples_leaf: Sequence[int] = (10, 20, 30, 40)
    l2_regularization: Sequence[float] = (1e-3, 5e-3, 1e-2, 5e-2, 1e-1)
    early_stopping: Sequence[bool] = (True, False)

    def to_param_distributions(self) -> dict[str, Sequence[Any]]:
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "early_stopping": self.early_stopping,
        }


@dataclass(frozen=True)
class GlobalPositionTuningSpace:
    learning_rate: Sequence[float] = (0.03, 0.05, 0.08, 0.1)
    max_iter: Sequence[int] = (300, 400, 500, 650, 800)
    max_leaf_nodes: Sequence[int] = (31, 63, 95, 127, 191)
    max_depth: Sequence[int | None] = (5, 7, 9, None)
    min_samples_leaf: Sequence[int] = (5, 10, 20, 35)
    l2_regularization: Sequence[float] = (1e-3, 5e-3, 1e-2, 5e-2, 1e-1)
    early_stopping: Sequence[bool] = (True, False)

    def to_param_distributions(self) -> dict[str, Sequence[Any]]:
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "early_stopping": self.early_stopping,
        }


@dataclass(frozen=True)
class TrainingConfig:
    global_win: GlobalWinModelConfig = field(default_factory=GlobalWinModelConfig)
    global_position: GlobalPositionModelConfig = field(default_factory=GlobalPositionModelConfig)
    race_win: RaceWinModelConfig = field(default_factory=RaceWinModelConfig)
    race_position: RacePositionModelConfig = field(default_factory=RacePositionModelConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    global_win_space: GlobalWinTuningSpace = field(default_factory=GlobalWinTuningSpace)
    global_position_space: GlobalPositionTuningSpace = field(default_factory=GlobalPositionTuningSpace)
    race_min_samples: int = 30
