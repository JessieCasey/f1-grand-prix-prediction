from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pandas as pd

from .feature_engineering import PREDICTION_INPUT_COLUMNS


class JolpicaClient:
    """Thin client for the Jolpica API."""

    def __init__(self, base_url: str = "https://api.jolpi.ca/ergast/f1"):
        self.base_url = base_url.rstrip("/")

    def _get(
        self,
        path: str,
        params: dict | None = None,
        *,
        limit: int = 1000,
        retries: int = 3,
        timeout: int = 30,
    ) -> dict:
        import requests

        url = f"{self.base_url}/{path.lstrip('/')}"
        p = dict(params or {})
        p.setdefault("limit", str(limit))
        last_err = None
        for attempt in range(retries):
            try:
                response = requests.get(url, params=p, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                return data.get("MRData", {})
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Jolpica GET failed after {retries} attempts: {last_err}")

    def fetch_season_results(self, season: int) -> pd.DataFrame:
        """Return flattened DataFrame of all race results for a season."""
        mr = self._get(f"{season}/results.json", params={"limit": "2000"})
        races = mr.get("RaceTable", {}).get("Races", [])
        rows: list[dict] = []
        for race in races:
            s = int(race.get("season"))
            rnd = int(race.get("round"))
            race_name = race.get("raceName")
            circuit_ref = (race.get("Circuit") or {}).get("circuitId")
            for res in race.get("Results", []):
                drv = res.get("Driver", {})
                cons = res.get("Constructor", {})
                rows.append(
                    {
                        "season": s,
                        "round": rnd,
                        "raceName": race_name,
                        "circuitRef": circuit_ref,
                        "driverRef": drv.get("driverId"),
                        "constructorRef": cons.get("constructorId"),
                        "grid": int(res.get("grid") or 0),
                        "position": int(res.get("position") or 0),
                        "points": float(res.get("points") or 0.0),
                    }
                )
        return pd.DataFrame(rows)

    def fetch_round_qualifying_entries(self, season: int, rnd: int) -> Tuple[pd.DataFrame, str | None, str | None]:
        """Return qualifying entries for a given round along with circuit and race metadata."""
        mr = self._get(f"{season}/{rnd}/qualifying.json")
        races = mr.get("RaceTable", {}).get("Races", [])
        if not races:
            return pd.DataFrame(), None, None
        race_info = races[0]
        results = race_info.get("QualifyingResults", [])
        circuit_ref = (race_info.get("Circuit") or {}).get("circuitId")
        race_name = race_info.get("raceName")
        rows = []
        for res in results:
            drv = res.get("Driver", {})
            cons = res.get("Constructor", {})
            rows.append(
                {
                    "driverRef": drv.get("driverId"),
                    "constructorRef": cons.get("constructorId"),
                    "grid": int(res.get("position") or 0),
                    "season": season,
                    "round": rnd,
                    "raceName": race_name or "",
                }
            )
        return pd.DataFrame(rows), circuit_ref, race_name

    def build_inputs_for_round(
        self,
        season: int,
        rnd: int,
        kaggle_drivers: pd.DataFrame,
        kaggle_constructors: pd.DataFrame,
        kaggle_races: pd.DataFrame,
        kaggle_circuits: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Create an inputs DataFrame with engineered features required by the predictor."""
        df_season = self.fetch_season_results(season)
        if df_season.empty:
            raise RuntimeError(f"No data from Jolpica for season {season}")

        prev = df_season[df_season["round"] < rnd].copy()
        prev_points = prev.groupby("driverRef")["points"].sum() if not prev.empty else pd.Series(dtype=float)
        prev_wins = (
            (prev["position"] == 1).astype(int).groupby(prev["driverRef"]).sum() if not prev.empty else pd.Series(dtype=int)
        )
        prev_podiums = (
            (prev["position"] <= 3).astype(int).groupby(prev["driverRef"]).sum()
            if not prev.empty
            else pd.Series(dtype=int)
        )
        if not prev.empty:
            driver_finish_sum = prev.groupby("driverRef")["position"].sum()
            driver_finish_count = prev.groupby("driverRef")["position"].count()
            driver_prev_avg_finish = driver_finish_sum / driver_finish_count.replace(0, np.nan)
        else:
            driver_prev_avg_finish = pd.Series(dtype=float)

        constructor_prev_points = (
            prev.groupby("constructorRef")["points"].sum() if not prev.empty else pd.Series(dtype=float)
        )
        constructor_prev_wins = (
            (prev["position"] == 1).astype(int).groupby(prev["constructorRef"]).sum()
            if not prev.empty
            else pd.Series(dtype=int)
        )

        entries = (
            df_season[df_season["round"] == rnd][
                ["driverRef", "constructorRef", "circuitRef", "grid", "season", "round", "raceName"]
            ]
            .drop_duplicates(subset=["driverRef", "constructorRef"])
            .copy()
        )
        if entries.empty:
            quali_df, circuit_ref, race_name = self.fetch_round_qualifying_entries(season, rnd)
            if quali_df.empty:
                raise RuntimeError(f"Could not find entries for season {season} round {rnd}")
            entries = quali_df
            entries["circuitRef"] = circuit_ref
            entries["raceName"] = race_name or ""

        entries["prev_points"] = entries["driverRef"].map(prev_points).fillna(0.0)
        entries["prev_wins"] = entries["driverRef"].map(prev_wins).fillna(0)
        entries["driver_prev_podiums"] = entries["driverRef"].map(prev_podiums).fillna(0)
        entries["driver_prev_avg_finish"] = entries["driverRef"].map(driver_prev_avg_finish).fillna(entries["grid"])
        entries["constructor_prev_points"] = entries["constructorRef"].map(constructor_prev_points).fillna(0.0)
        entries["constructor_prev_wins"] = entries["constructorRef"].map(constructor_prev_wins).fillna(0)
        entries["year"] = season
        entries["round"] = entries["round"].astype(int)

        race_row = kaggle_races[(kaggle_races["year"] == season) & (kaggle_races["round"] == rnd)]
        race_date = pd.to_datetime(race_row["date"].iloc[0], errors="coerce") if not race_row.empty else pd.NaT

        drivers_dob = pd.to_datetime(kaggle_drivers.set_index("driverRef")["dob"], errors="coerce")
        if pd.isna(race_date):
            entries["driver_age"] = np.nan
        else:
            entries["driver_age"] = (race_date - entries["driverRef"].map(drivers_dob)).dt.days / 365.25
        if entries["driver_age"].notna().any():
            entries["driver_age"] = entries["driver_age"].fillna(entries["driver_age"].median())
        else:
            entries["driver_age"] = 30.0

        for col in PREDICTION_INPUT_COLUMNS:
            if col in ("driverRef", "constructorRef", "circuitRef"):
                continue
            entries[col] = pd.to_numeric(entries[col], errors="coerce").fillna(0.0)
        avg_zero_mask = entries["driver_prev_avg_finish"] <= 0
        entries.loc[avg_zero_mask, "driver_prev_avg_finish"] = entries.loc[avg_zero_mask, "grid"].clip(lower=1)
        if entries["driver_age"].eq(0).all():
            entries["driver_age"] = 30.0
        else:
            median_age = entries.loc[entries["driver_age"] > 0, "driver_age"].median()
            entries.loc[entries["driver_age"] == 0, "driver_age"] = median_age if not np.isnan(median_age) else 30.0

        dmap = dict(zip(kaggle_drivers["driverRef"], kaggle_drivers["driverId"]))
        cmap = dict(zip(kaggle_constructors["constructorRef"], kaggle_constructors["constructorId"]))
        entries["driverId"] = entries["driverRef"].map(dmap)
        entries["constructorId"] = entries["constructorRef"].map(cmap)

        if kaggle_circuits is not None:
            cimap = dict(zip(kaggle_circuits["circuitRef"], kaggle_circuits["circuitId"]))
            entries["circuitId"] = entries["circuitRef"].map(cimap)
        else:
            entries["circuitId"] = None

        ordered_cols = [
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
            "driver_age",
        ]
        final_df = entries[ordered_cols].copy()
        race_label = final_df["raceName"].iloc[0] if not final_df.empty else "unknown"
        print(
            f"[jolpica] Prepared {len(final_df)} entries for season {season} round {rnd} "
            f"({race_label}) with {final_df['constructorRef'].nunique()} teams."
        )
        return final_df


    def get_next_race_candidates(
        self,
        season: int,
        kaggle_drivers: pd.DataFrame | None = None,
        kaggle_constructors: pd.DataFrame | None = None,
        kaggle_races: pd.DataFrame | None = None,
        kaggle_circuits: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        mr = self._get(f"{season}/races.json", params={"limit": "2000"})
        races = mr.get("RaceTable", {}).get("Races", [])
        if not races:
            raise RuntimeError(f"No races found for season {season}")

        df_calendar = pd.DataFrame([
            {
                "season": int(race.get("season")),
                "round": int(race.get("round")),
                "raceName": race.get("raceName"),
                "circuitRef": (race.get("Circuit") or {}).get("circuitId"),
                "date": race.get("date"),
            }
            for race in races
        ])

        df_calendar["date"] = pd.to_datetime(df_calendar["date"], errors="coerce", utc=True).dt.tz_localize(None)
        now = pd.Timestamp.utcnow().tz_localize(None)
        future = df_calendar[df_calendar["date"] > now]
        if not future.empty:
            next_race = future.sort_values("date").iloc[0]
        else:
            next_race = df_calendar.sort_values("round", ascending=False).iloc[0]
        next_round = int(next_race["round"])
        print(f"[jolpica] Next race in season {season} is round {next_round}: {next_race['raceName']}")

        try:
            inputs_df = self.build_inputs_for_round(
                season,
                next_round,
                kaggle_drivers=kaggle_drivers,
                kaggle_constructors=kaggle_constructors,
                kaggle_races=kaggle_races,
                kaggle_circuits=kaggle_circuits,
            )
            if not inputs_df.empty:
                return inputs_df
        except RuntimeError:
            pass

        print(
            "[jolpica] Falling back to historical participants for the next race "
            "because current round data is unavailable."
        )
        df_season = self.fetch_season_results(season)
        prev = df_season[df_season["round"] < next_round].copy()
        if prev.empty:
            raise RuntimeError("No historical data available to approximate next race candidates.")

        prev_points = prev.groupby("driverRef")["points"].sum()
        prev_wins = (prev["position"] == 1).astype(int).groupby(prev["driverRef"]).sum()
        prev_podiums = (prev["position"] <= 3).astype(int).groupby(prev["driverRef"]).sum()
        driver_finish_sum = prev.groupby("driverRef")["position"].sum()
        driver_finish_count = prev.groupby("driverRef")["position"].count()
        driver_prev_avg_finish = driver_finish_sum / driver_finish_count.replace(0, np.nan)

        constructor_prev_points = prev.groupby("constructorRef")["points"].sum()
        constructor_prev_wins = (prev["position"] == 1).astype(int).groupby(prev["constructorRef"]).sum()

        last_round = prev["round"].max()
        last_entries = prev[prev["round"] == last_round][["driverRef", "constructorRef"]].drop_duplicates()

        entries = last_entries.copy()
        entries["season"] = season
        entries["year"] = season
        entries["round"] = next_round
        entries["raceName"] = next_race["raceName"]
        entries["circuitRef"] = next_race["circuitRef"]
        entries["grid"] = 0  # Unknown grid so far.

        entries["prev_points"] = entries["driverRef"].map(prev_points).fillna(0.0)
        entries["prev_wins"] = entries["driverRef"].map(prev_wins).fillna(0)
        entries["driver_prev_podiums"] = entries["driverRef"].map(prev_podiums).fillna(0)
        entries["driver_prev_avg_finish"] = entries["driverRef"].map(driver_prev_avg_finish).fillna(10.0)
        entries["constructor_prev_points"] = entries["constructorRef"].map(constructor_prev_points).fillna(0.0)
        entries["constructor_prev_wins"] = entries["constructorRef"].map(constructor_prev_wins).fillna(0)

        race_date = next_race["date"]
        race_date = pd.to_datetime(race_date, errors="coerce")
        if kaggle_drivers is not None:
            drivers_dob = pd.to_datetime(kaggle_drivers.set_index("driverRef")["dob"], errors="coerce")
            entries["driver_age"] = (
                (race_date - entries["driverRef"].map(drivers_dob)).dt.days / 365.25
                if pd.notna(race_date)
                else np.nan
            )
        else:
            entries["driver_age"] = np.nan
        if entries["driver_age"].notna().any():
            entries["driver_age"] = entries["driver_age"].fillna(entries["driver_age"].median())
        else:
            entries["driver_age"] = 30.0

        for col in PREDICTION_INPUT_COLUMNS:
            if col in ("driverRef", "constructorRef", "circuitRef", "year", "round"):
                continue
            entries[col] = pd.to_numeric(entries[col], errors="coerce").fillna(0.0)
        avg_zero_mask = entries["driver_prev_avg_finish"] <= 0
        entries.loc[avg_zero_mask, "driver_prev_avg_finish"] = entries.loc[avg_zero_mask, "grid"].clip(lower=1)

        if kaggle_drivers is not None:
            dmap = dict(zip(kaggle_drivers["driverRef"], kaggle_drivers["driverId"]))
            entries["driverId"] = entries["driverRef"].map(dmap)
        else:
            entries["driverId"] = None
        if kaggle_constructors is not None:
            cmap = dict(zip(kaggle_constructors["constructorRef"], kaggle_constructors["constructorId"]))
            entries["constructorId"] = entries["constructorRef"].map(cmap)
        else:
            entries["constructorId"] = None

        if kaggle_circuits is not None:
            cimap = dict(zip(kaggle_circuits["circuitRef"], kaggle_circuits["circuitId"]))
            entries["circuitId"] = entries["circuitRef"].map(cimap)
        else:
            entries["circuitId"] = None

        ordered_cols = [
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
            "driver_age",
        ]
        return entries[ordered_cols]
