from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .feature_engineering import FeatureEngineer
from .jolpica import JolpicaClient
from .models import KaggleDownloadConfig, PredictorPaths, RaceIdentifier, TrainingConfig
from .predictor import F1GrandPrixPredictor

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    race: RaceIdentifier
    paths: PredictorPaths
    kaggle: KaggleDownloadConfig = field(default_factory=KaggleDownloadConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_project_root(cls, race: RaceIdentifier, project_root: Path) -> "PipelineConfig":
        return cls(race=race, paths=PredictorPaths.from_project_root(project_root))


class PredictorPipeline:
    def __init__(self, predictor: F1GrandPrixPredictor, config: PipelineConfig) -> None:
        self.predictor = predictor
        self.config = config

    @classmethod
    def with_default_predictor(cls, config: PipelineConfig) -> "PredictorPipeline":
        predictor = F1GrandPrixPredictor(
            JolpicaClient(),
            FeatureEngineer(),
            paths=config.paths,
        )
        return cls(predictor, config)

    def run(self) -> None:
        self.config.paths.results_dir.mkdir(parents=True, exist_ok=True)

        self.predictor.download_kaggle_data(self.config.kaggle)
        self.predictor.load_data()
        self.predictor.preprocess()
        self.predictor.verify_jolpica_availability(
            self.config.race.season,
            self.config.race.round,
        )
        self.predictor.train(self.config.training)

        self.predictor.build_inputs_csv_via_jolpica(
            season=self.config.race.season,
            rnd=self.config.race.round,
            output_csv=self.config.paths.race_inputs_path,
        )

        results = self.predictor.predict_from_csv(
            self.config.paths.race_inputs_path,
            output_path=self.config.paths.prediction_path,
        )
        print(results.head())
