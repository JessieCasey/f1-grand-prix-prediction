from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.core.models import (
    JolpicaNextRaceRequest,
    JolpicaRoundRequest,
    KaggleDataSources,
    TrainingData,
)


class FeatureEngineerBase(ABC):
    @abstractmethod
    def build_training_data(self, sources: KaggleDataSources) -> TrainingData:
        raise NotImplementedError

    @abstractmethod
    def transform_prediction_inputs(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        raise NotImplementedError


class RaceDataClient(ABC):
    @abstractmethod
    def verify_availability(self, season: int, rnd: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_inputs_for_round(self, request: JolpicaRoundRequest) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_next_race_candidates(self, request: JolpicaNextRaceRequest) -> pd.DataFrame:
        raise NotImplementedError
