import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------- Jolpica (Ergast-compatible) client ----------------------------
class JolpicaClient:
    """Thin client for Jolpica (Ergast-compatible) API to fetch entries/results."""

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
        """Return qualifying entries for a given round, with circuit and race metadata."""
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

        for col in (
            "grid",
            "prev_points",
            "prev_wins",
            "driver_prev_podiums",
            "driver_prev_avg_finish",
            "constructor_prev_points",
            "constructor_prev_wins",
            "driver_age",
        ):
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

        final_df = entries[
            [
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
        ].copy()
        race_label = final_df["raceName"].iloc[0] if not final_df.empty else "unknown"
        print(
            f"[jolpica] Prepared {len(final_df)} entries for season {season} round {rnd} "
            f"({race_label}) with {final_df['constructorRef'].nunique()} teams."
        )
        return final_df


# ---------------------------- Predictor ----------------------------
class F1GrandPrixPredictor:
    def __init__(self, data_dir: str | Path | None = None, jolpica: JolpicaClient | None = None):
        self.global_win_model: LogisticRegression | None = None
        self.global_position_model: RandomForestRegressor | None = None
        self.win_models: Dict[str, LogisticRegression] = {}
        self.position_models: Dict[str, RandomForestRegressor] = {}

        self.driver_encoder = LabelEncoder()
        self.constructor_encoder = LabelEncoder()
        self.circuit_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        base_path = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else base_path / "data" / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.jolpica = jolpica or JolpicaClient()
        self.training_frame: pd.DataFrame | None = None
        self.X_scaled_full: np.ndarray | None = None

        self.features = [
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
            "driver_age",
        ]

    # ---------------------------- Kaggle data ----------------------------
    def download_kaggle_data(self, kaggle_dataset: str = "rohanrao/formula-1-world-championship-1950-2020") -> None:
        required = {
            name: self.data_dir / name
            for name in ("results.csv", "races.csv", "drivers.csv", "constructors.csv", "circuits.csv")
        }
        if all(path.exists() for path in required.values()):
            return
        try:
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
            raise RuntimeError("Kaggle CLI not available. Install with `pip install kaggle`.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Kaggle download failed ({exc.returncode}): {exc.stderr.strip()}") from exc

    def load_data(self) -> None:
        self.results = pd.read_csv(self.data_dir / "results.csv")
        self.races = pd.read_csv(self.data_dir / "races.csv")
        self.drivers = pd.read_csv(self.data_dir / "drivers.csv").drop(columns=["url"], errors="ignore")
        self.constructors = pd.read_csv(self.data_dir / "constructors.csv").drop(columns=["url"], errors="ignore")
        self.circuits = pd.read_csv(self.data_dir / "circuits.csv").drop(columns=["url"], errors="ignore")

    # ---------------------------- Feature engineering ----------------------------
    def preprocess(self) -> None:
        df = self.results.merge(self.races, on="raceId", suffixes=("", "_race"))
        df = df.merge(self.drivers, on="driverId")
        df = df.merge(self.constructors, on="constructorId")
        df = df.merge(self.circuits, on="circuitId")

        if "name" in df.columns and "raceName" not in df.columns:
            df["raceName"] = df["name"]

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

        df.sort_values(by=["driverId", "year", "round"], inplace=True)
        df["finish_position"] = df["positionOrder"].astype(float)
        df["target"] = (df["finish_position"] == 1).astype(int)

        df["prev_points"] = df.groupby("driverId")["points"].transform(lambda s: s.cumsum() - s)
        win_flag = (df["positionOrder"] == 1).astype(int)
        df["prev_wins"] = win_flag.groupby(df["driverId"]).cumsum() - win_flag

        podium_flag = (df["positionOrder"] <= 3).astype(int)
        df["driver_prev_podiums"] = podium_flag.groupby(df["driverId"]).cumsum() - podium_flag

        driver_finish_cumsum = df.groupby("driverId")["finish_position"].cumsum() - df["finish_position"]
        driver_counts = df.groupby("driverId").cumcount()
        df["driver_prev_avg_finish"] = driver_finish_cumsum / driver_counts.replace(0, np.nan)
        df.loc[driver_counts == 0, "driver_prev_avg_finish"] = df.loc[driver_counts == 0, "grid"]
        df["driver_prev_avg_finish"] = df["driver_prev_avg_finish"].fillna(df["grid"])

        df["constructor_prev_points"] = df.groupby("constructorId")["points"].cumsum() - df["points"]
        df["constructor_prev_wins"] = win_flag.groupby(df["constructorId"]).cumsum() - win_flag

        df["driverRef"] = df["driverRef"].fillna("unknown")
        df["constructorRef"] = df["constructorRef"].fillna("unknown")
        df["circuitRef"] = df["circuitRef"].fillna("unknown")
        df["driver_encoded"] = self.driver_encoder.fit_transform(df["driverRef"])
        df["constructor_encoded"] = self.constructor_encoder.fit_transform(df["constructorRef"])
        df["circuit_encoded"] = self.circuit_encoder.fit_transform(df["circuitRef"])

        df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25
        df["driver_age"] = df["driver_age"].fillna(df["driver_age"].median())

        df["year"] = df["year"].astype(int)
        df["round"] = df["round"].astype(int)

        feature_cols = self.features + ["target", "finish_position", "raceName"]
        df_model = df[feature_cols].copy().reset_index(drop=True)
        df_model[self.features] = df_model[self.features].fillna(0.0)

        X = df_model[self.features].values
        y_win = df_model["target"].astype(int).values
        y_pos = df_model["finish_position"].astype(float).values
        race_names = df_model["raceName"].astype(str).values

        self.X_scaled_full = self.scaler.fit_transform(X)
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
            self.X_scaled_full,
            y_win,
            y_pos,
            race_names,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        self.training_frame = df_model

    # ---------------------------- Training ----------------------------
    def train(self, max_iter: int = 1000) -> None:
        if self.training_frame is None or self.X_scaled_full is None:
            raise RuntimeError("Call preprocess() before train().")

        self.global_win_model = LogisticRegression(max_iter=max_iter, class_weight="balanced")
        self.global_win_model.fit(self.X_train, self.y_train_win)
        win_probs = self.global_win_model.predict_proba(self.X_test)[:, 1]
        win_pred = (win_probs >= 0.5).astype(int)
        print("Global win accuracy:", accuracy_score(self.y_test_win, win_pred))
        print(classification_report(self.y_test_win, win_pred))

        self.global_position_model = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        )
        self.global_position_model.fit(self.X_train, self.y_train_position)
        pos_pred = self.global_position_model.predict(self.X_test)
        print(f"Global position MAE: {mean_absolute_error(self.y_test_position, pos_pred):.2f}")

        self.win_models.clear()
        self.position_models.clear()
        if self.training_frame is None:
            return

        min_samples = 30
        for race_name, subset in self.training_frame.groupby("raceName"):
            idx = subset.index.to_numpy()
            X_race = self.X_scaled_full[idx]
            y_win_race = subset["target"].to_numpy()
            y_pos_race = subset["finish_position"].to_numpy()

            if len(y_win_race) >= min_samples and np.unique(y_win_race).size > 1:
                model = LogisticRegression(max_iter=max_iter, class_weight="balanced")
                model.fit(X_race, y_win_race)
                self.win_models[race_name] = model

            if len(y_pos_race) >= min_samples:
                pos_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                pos_model.fit(X_race, y_pos_race)
                self.position_models[race_name] = pos_model

        print(
            f"Race-specific models trained: wins={len(self.win_models)} "
            f"positions={len(self.position_models)}"
        )

    # ---------------------------- Inference ----------------------------
    def _encode_with_new(self, encoder: LabelEncoder, series: pd.Series, kind: str) -> np.ndarray:
        values = series.fillna("unknown").astype(str)
        missing_mask = ~np.isin(values, encoder.classes_)
        if missing_mask.any():
            new_classes = np.unique(values[missing_mask])
            encoder.classes_ = np.unique(np.concatenate([encoder.classes_, new_classes]))
            print(f"[warn] Added new {kind} labels: {', '.join(new_classes)}")
        return encoder.transform(values)

    def predict_from_csv(self, csv_path: str | Path, output_path: str = "prediction_results.csv") -> pd.DataFrame:
        if self.global_win_model is None or self.global_position_model is None:
            raise RuntimeError("Models are not trained. Call train() first.")

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Input CSV '{csv_path}' not found. Build it via Jolpica:\n"
                "  predictor.build_inputs_csv_via_jolpica(..., output_csv='race_inputs.csv')"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Input CSV '{csv_path}' is empty.")

        df = pd.read_csv(path)
        required = {
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
            "driver_age",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in '{csv_path}': {sorted(missing)}. "
                "Regenerate inputs with build_inputs_csv_via_jolpica."
            )

        if "raceName" not in df.columns:
            raise ValueError("Prediction CSV must contain 'raceName' to select race-specific models.")

        race_summary = " ".join(f"({idx}, '{name}')" for idx, name in enumerate(df["raceName"].tolist()))
        print(f"[predict] raceName rows: {race_summary}")

        df["driver_encoded"] = self._encode_with_new(self.driver_encoder, df["driverRef"], "driver")
        df["constructor_encoded"] = self._encode_with_new(self.constructor_encoder, df["constructorRef"], "constructor")
        df["circuit_encoded"] = self._encode_with_new(self.circuit_encoder, df["circuitRef"], "circuit")

        numeric_cols = [col for col in self.features if col not in ("driver_encoded", "constructor_encoded", "circuit_encoded")]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        zero_mask = df["driver_prev_avg_finish"] <= 0
        df.loc[zero_mask, "driver_prev_avg_finish"] = df.loc[zero_mask, "grid"].clip(lower=1)
        if df["driver_age"].eq(0).all():
            df["driver_age"] = 30.0
        else:
            median_age = df.loc[df["driver_age"] > 0, "driver_age"].median()
            df.loc[df["driver_age"] == 0, "driver_age"] = median_age if not np.isnan(median_age) else 30.0

        X = df[self.features].values
        X_scaled = self.scaler.transform(X)

        raw_win_prob = np.zeros(len(df))
        predicted_positions = np.zeros(len(df))

        for race_name, group in df.groupby("raceName"):
            idx = group.index.to_numpy()
            X_group = X_scaled[idx]

            if race_name in self.win_models:
                win_model = self.win_models[race_name]
                source_win = "race"
            else:
                win_model = self.global_win_model
                source_win = "global"
            probs = win_model.predict_proba(X_group)[:, 1]
            raw_win_prob[idx] = probs

            if race_name in self.position_models:
                pos_model = self.position_models[race_name]
                source_pos = "race"
            else:
                pos_model = self.global_position_model
                source_pos = "global"
            predicted_positions[idx] = pos_model.predict(X_group)

            print(f"[predict][{race_name}] win_model={source_win} position_model={source_pos} samples={len(idx)}")

        df["raw_win_probability"] = raw_win_prob
        df["predicted_position"] = predicted_positions

        df["win_probability"] = 0.0
        for race_name, group in df.groupby("raceName"):
            idx = group.index.to_numpy()
            probs = raw_win_prob[idx]
            total = probs.sum()
            if total > 0:
                df.loc[idx, "win_probability"] = probs / total
            else:
                df.loc[idx, "win_probability"] = 1.0 / len(idx)

        df["predicted_rank"] = df.groupby("raceName")["predicted_position"].rank(method="first").astype(int)
        df["predicted_top5"] = df["predicted_rank"] <= 5

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
        available_cols = [col for col in ordered_cols if col in df.columns]

        sort_cols = ["raceName", "predicted_rank", "predicted_position"]
        results = df[available_cols].sort_values(sort_cols)
        results.to_csv(output_path, index=False)

        for race_name, group in results.groupby("raceName"):
            top = group.nsmallest(5, "predicted_rank")
            print(f"[predict][{race_name}] Top 5 predicted finishers:")
            print(top[["driverRef", "predicted_position", "win_probability"]].to_string(index=False))

        return results

    # ---------------------------- Jolpica helper ----------------------------
    def build_inputs_csv_via_jolpica(self, season: int, rnd: int, output_csv: str) -> pd.DataFrame:
        final_df = self.jolpica.build_inputs_for_round(
            season,
            rnd,
            kaggle_drivers=self.drivers,
            kaggle_constructors=self.constructors,
            kaggle_races=self.races,
            kaggle_circuits=self.circuits,
        )
        final_df.to_csv(output_csv, index=False)
        race_label = final_df["raceName"].iloc[0] if not final_df.empty else "unknown"
        print(f"[jolpica] Saved inputs for season {season} round {rnd} ({race_label}) -> {output_csv}")
        return final_df


if __name__ == "__main__":
    predictor = F1GrandPrixPredictor()
    predictor.download_kaggle_data()
    predictor.load_data()
    predictor.preprocess()
    predictor.train()
    result_df = predictor.predict_from_csv("race_inputs.csv")
    print(result_df)
