from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from .abstract.interfaces import RaceDataClient
from .models import JolpicaNextRaceRequest, JolpicaRoundRequest, RaceIdentifier

ROUND_OUTPUT_COLUMNS = [
    "season",
    "year",
    "round",
    "raceName",
    "driverRef",
    "constructorRef",
    "circuitRef",
    "driverId",
    "constructorId",
    "circuitId",
    "grid",
    "prev_points",
    "prev_wins",
    "driver_prev_podiums",
    "driver_prev_avg_finish",
    "constructor_prev_points",
    "constructor_prev_wins",
]

NUMERIC_FEATURE_COLUMNS = [
    "grid",
    "prev_points",
    "prev_wins",
    "driver_prev_podiums",
    "driver_prev_avg_finish",
    "constructor_prev_points",
    "constructor_prev_wins",
]

LOGGER = logging.getLogger(__name__)


@dataclass
class RaceCalendarEntry:
    season: int
    round: int
    race_name: str
    circuit_ref: Optional[str]
    date: pd.Timestamp


class JolpicaClient(RaceDataClient):
    def __init__(self, base_url: str = "https://api.jolpi.ca/ergast/f1"):
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        path: str,
        params: Optional[Dict[str, str]] = None,
        *,
        limit: int = 1000,
        retries: int = 3,
        timeout: int = 30,
    ) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        query = dict(params or {})
        query.setdefault("limit", str(limit))

        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                response = requests.get(url, params=query, timeout=timeout)
                response.raise_for_status()
                return response.json().get("MRData", {})
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Jolpica GET failed after {retries} attempts: {last_err}")

    def fetch_season_results(self, season: int) -> pd.DataFrame:
        payload = self._request(f"{season}/results.json")
        races = payload.get("RaceTable", {}).get("Races", [])
        rows: list[dict] = []
        for race in races:
            circuit = race.get("Circuit") or {}
            for result in race.get("Results", []):
                driver = result.get("Driver") or {}
                constructor = result.get("Constructor") or {}
                rows.append(
                    {
                        "season": int(race.get("season")),
                        "round": int(race.get("round")),
                        "raceName": race.get("raceName"),
                        "circuitRef": circuit.get("circuitId"),
                        "driverRef": driver.get("driverId"),
                        "constructorRef": constructor.get("constructorId"),
                        "grid": int(result.get("grid") or 0),
                        "position": int(result.get("position") or 0),
                        "points": float(result.get("points") or 0.0),
                    }
                )
        return pd.DataFrame(rows)

    def verify_availability(self, season: int, rnd: Optional[int] = None) -> None:
        try:
            self._fetch_calendar(season)
            if rnd is not None:
                self.fetch_round_qualifying_entries(season, rnd)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to reach Jolpica API: {exc}") from exc

    def fetch_round_qualifying_entries(self, season: int, rnd: int) -> tuple[
        pd.DataFrame, Optional[str], Optional[str]]:
        payload = self._request(f"{season}/{rnd}/qualifying.json")
        races = payload.get("RaceTable", {}).get("Races", [])
        if not races:
            return pd.DataFrame(), None, None

        race = races[0]
        circuit = (race.get("Circuit") or {}).get("circuitId")
        race_name = race.get("raceName")
        rows = []
        for result in race.get("QualifyingResults", []):
            driver = result.get("Driver") or {}
            constructor = result.get("Constructor") or {}
            rows.append(
                {
                    "driverRef": driver.get("driverId"),
                    "constructorRef": constructor.get("constructorId"),
                    "grid": int(result.get("position") or 0),
                    "season": season,
                    "round": rnd,
                    "raceName": race_name or "",
                }
            )
        return pd.DataFrame(rows), circuit, race_name

    def build_inputs_for_round(self, request: JolpicaRoundRequest) -> pd.DataFrame:
        race = request.race
        sources = request.sources
        season_results = self.fetch_season_results(race.season)
        if season_results.empty:
            raise RuntimeError(f"No race results available for season {race.season}.")

        entries = self._extract_round_entries(season_results, race)
        stats = self._compute_historical_stats(season_results, race)
        enriched = self._attach_stats(entries, stats, race)
        return self._attach_identifiers(
            enriched,
            sources.drivers,
            sources.constructors,
            sources.circuits,
        )

    def get_next_race_candidates(self, request: JolpicaNextRaceRequest) -> pd.DataFrame:
        season = request.season
        sources = request.sources
        calendar = self._fetch_calendar(season)
        next_race = self._determine_next_race(calendar)

        try:
            return self.build_inputs_for_round(
                JolpicaRoundRequest(
                    race=RaceIdentifier(season=season, round=next_race.round),
                    sources=sources,
                )
            )
        except RuntimeError:
            LOGGER.info(
                "[jolpica] Falling back to historical participant list because Jolpica "
                "does not yet expose the upcoming round."
            )
            return self._approximate_candidates_from_history(
                season,
                next_race,
                sources.drivers,
                sources.constructors,
                sources.circuits,
            )

    def _fetch_calendar(self, season: int) -> pd.DataFrame:
        payload = self._request(f"{season}/races.json")
        races = payload.get("RaceTable", {}).get("Races", [])
        if not races:
            raise RuntimeError(f"No races found for season {season}.")

        df = pd.DataFrame(
            {
                "season": int(race.get("season")),
                "round": int(race.get("round")),
                "raceName": race.get("raceName"),
                "circuitRef": (race.get("Circuit") or {}).get("circuitId"),
                "date": race.get("date"),
            }
            for race in races
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["date"] = df["date"].map(_to_naive_timestamp)
        return df.sort_values("round")

    def _determine_next_race(self, calendar: pd.DataFrame) -> RaceCalendarEntry:
        now = pd.Timestamp.utcnow().replace(tzinfo=None)
        future = calendar[calendar["date"] > now]
        if not future.empty:
            race_row = future.sort_values("date").iloc[0]
        else:
            race_row = calendar.iloc[-1]
        return RaceCalendarEntry(
            season=int(race_row["season"]),
            round=int(race_row["round"]),
            race_name=str(race_row["raceName"]),
            circuit_ref=race_row.get("circuitRef"),
            date=race_row["date"],
        )

    def _extract_round_entries(self, season_results: pd.DataFrame, race: RaceIdentifier) -> pd.DataFrame:
        round_entries = (
            season_results[season_results["round"] == race.round][
                ["driverRef", "constructorRef", "circuitRef", "grid", "season", "round", "raceName"]
            ]
            .drop_duplicates(subset=["driverRef", "constructorRef"])
            .copy()
        )
        if not round_entries.empty:
            return round_entries

        qualifying_df, circuit_ref, race_name = self.fetch_round_qualifying_entries(race.season, race.round)
        if qualifying_df.empty:
            raise RuntimeError(f"Could not find entries for season {race.season} round {race.round}.")

        qualifying_df["circuitRef"] = circuit_ref
        qualifying_df["raceName"] = race_name or ""
        return qualifying_df

    def _compute_historical_stats(self, season_results: pd.DataFrame, race: RaceIdentifier) -> Dict[str, pd.Series]:
        history = season_results[season_results["round"] < race.round].copy()
        if history.empty:
            empty_series = pd.Series(dtype=float)
            return {
                "driver_points": empty_series,
                "driver_wins": empty_series,
                "driver_podiums": empty_series,
                "driver_avg_finish": empty_series,
                "constructor_points": empty_series,
                "constructor_wins": empty_series,
            }

        driver_points = history.groupby("driverRef")["points"].sum()
        driver_wins = (history["position"] == 1).astype(int).groupby(history["driverRef"]).sum()
        driver_podiums = (history["position"] <= 3).astype(int).groupby(history["driverRef"]).sum()

        finish_sum = history.groupby("driverRef")["position"].sum()
        finish_count = history.groupby("driverRef")["position"].count()
        driver_avg_finish = finish_sum / finish_count.replace(0, np.nan)

        constructor_points = history.groupby("constructorRef")["points"].sum()
        constructor_wins = (history["position"] == 1).astype(int).groupby(history["constructorRef"]).sum()

        return {
            "driver_points": driver_points,
            "driver_wins": driver_wins,
            "driver_podiums": driver_podiums,
            "driver_avg_finish": driver_avg_finish,
            "constructor_points": constructor_points,
            "constructor_wins": constructor_wins,
        }

    def _attach_stats(
        self,
        entries: pd.DataFrame,
        stats: Dict[str, pd.Series],
        race: RaceIdentifier,
    ) -> pd.DataFrame:
        enriched = entries.copy()
        enriched["prev_points"] = enriched["driverRef"].map(stats["driver_points"]).fillna(0.0)
        enriched["prev_wins"] = enriched["driverRef"].map(stats["driver_wins"]).fillna(0)
        enriched["driver_prev_podiums"] = enriched["driverRef"].map(stats["driver_podiums"]).fillna(0)
        enriched["driver_prev_avg_finish"] = enriched["driverRef"].map(stats["driver_avg_finish"]).fillna(
            enriched["grid"]
        )
        enriched["constructor_prev_points"] = enriched["constructorRef"].map(stats["constructor_points"]).fillna(0.0)
        enriched["constructor_prev_wins"] = enriched["constructorRef"].map(stats["constructor_wins"]).fillna(0)
        enriched["year"] = race.season
        enriched["round"] = enriched["round"].astype(int)

        for col in NUMERIC_FEATURE_COLUMNS:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0.0)
        avg_zero_mask = enriched["driver_prev_avg_finish"] <= 0
        enriched.loc[avg_zero_mask, "driver_prev_avg_finish"] = enriched.loc[avg_zero_mask, "grid"].clip(lower=1)
        return enriched

    def _attach_identifiers(
        self,
        entries: pd.DataFrame,
        kaggle_drivers: pd.DataFrame,
        kaggle_constructors: pd.DataFrame,
        kaggle_circuits: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        result = entries.copy()
        driver_map = dict(
            zip(
                kaggle_drivers["driverRef"],
                kaggle_drivers["driverId"],
                strict=False,
            )
        )
        constructor_map = dict(
            zip(
                kaggle_constructors["constructorRef"],
                kaggle_constructors["constructorId"],
                strict=False,
            )
        )
        result["driverId"] = result["driverRef"].map(driver_map)
        result["constructorId"] = result["constructorRef"].map(constructor_map)

        if kaggle_circuits is not None and not kaggle_circuits.empty:
            circuit_map = dict(
                zip(
                    kaggle_circuits["circuitRef"],
                    kaggle_circuits["circuitId"],
                    strict=False,
                )
            )
            result["circuitId"] = result["circuitRef"].map(circuit_map)
        else:
            result["circuitId"] = None

        LOGGER.info(
            "[jolpica] Prepared %s entries for season %s round %s (%s) with %s constructors.",
            len(result),
            result["season"].iloc[0],
            result["round"].iloc[0],
            result["raceName"].iloc[0],
            result["constructorRef"].nunique(),
        )
        return result[ROUND_OUTPUT_COLUMNS].copy()

    def _approximate_candidates_from_history(
        self,
        season: int,
        next_race: RaceCalendarEntry,
        kaggle_drivers: pd.DataFrame,
        kaggle_constructors: pd.DataFrame,
        kaggle_circuits: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        season_results = self.fetch_season_results(season)
        history = season_results[season_results["round"] < next_race.round]
        if history.empty:
            raise RuntimeError("No historical data available to approximate next race candidates.")

        last_round = history["round"].max()
        last_entries = history[history["round"] == last_round][["driverRef", "constructorRef"]].drop_duplicates()
        stats = self._compute_historical_stats(
            season_results,
            RaceIdentifier(season=season, round=next_race.round),
        )

        approx = last_entries.copy()
        approx["season"] = season
        approx["year"] = season
        approx["round"] = next_race.round
        approx["raceName"] = next_race.race_name
        approx["circuitRef"] = next_race.circuit_ref
        approx["grid"] = 0

        approx = self._attach_stats(
            approx,
            stats,
            race=RaceIdentifier(season=season, round=next_race.round),
        )
        return self._attach_identifiers(approx, kaggle_drivers, kaggle_constructors, kaggle_circuits)


def _to_naive_timestamp(value: pd.Timestamp | pd.NaT) -> pd.Timestamp:
    return value.to_pydatetime().replace(tzinfo=None)
