import time
import subprocess
from pathlib import Path

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# ---------------------------- Jolpica (Ergast-compatible) client ----------------------------
class JolpicaClient:
    """Thin client for Jolpica (Ergast-compatible) API to fetch entries/results
    and prepare race input frames compatible with F1GrandPrixPredictor.predict_from_csv().
    """
    def __init__(self, base_url: str = "https://api.jolpi.ca/ergast/f1"):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, params: dict | None = None, *, limit: int = 1000, retries: int = 3, timeout: int = 30) -> dict:
        import requests
        url = f"{self.base_url}/{path.lstrip('/')}"
        p = dict(params or {})
        p.setdefault("limit", str(limit))
        last_err = None
        for i in range(retries):
            try:
                r = requests.get(url, params=p, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("MRData", {})
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * (i + 1))
        raise RuntimeError(f"Jolpica GET failed after {retries} attempts: {last_err}")

    def fetch_season_results(self, season: int) -> pd.DataFrame:
        """Return flattened DataFrame of all race results for a season.
        Columns: season, round, raceName, circuitRef, driverRef, constructorRef, grid, position, points, status
        """
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
                        "status": res.get("status"),
                    }
                )
        return pd.DataFrame(rows)

    def fetch_round_qualifying_entries(self, season: int, rnd: int) -> tuple[pd.DataFrame, str | None]:
        """Return DataFrame with driverRef, constructorRef, grid from Quali; also circuitRef if present."""
        mr = self._get(f"{season}/{rnd}/qualifying.json")
        races = mr.get("RaceTable", {}).get("Races", [])
        if not races:
            return pd.DataFrame(), None
        results = races[0].get("QualifyingResults", [])
        circuit_ref = (races[0].get("Circuit") or {}).get("circuitId")
        rows = []
        for res in results:
            drv = res.get("Driver", {})
            cons = res.get("Constructor", {})
            rows.append(
                {
                    "driverRef": drv.get("driverId"),
                    "constructorRef": cons.get("constructorId"),
                    "grid": int(res.get("position") or 0),
                }
            )
        return pd.DataFrame(rows), circuit_ref

    def build_inputs_for_round(
        self,
        season: int,
        rnd: int,
        kaggle_drivers: pd.DataFrame,
        kaggle_constructors: pd.DataFrame,
        kaggle_circuits: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Create a race input DataFrame with columns required by the predictor.
        Returns columns: driverRef, constructorRef, circuitRef, driverId, constructorId, circuitId, grid, prev_points, prev_wins.
        """
        df_season = self.fetch_season_results(season)
        if df_season.empty:
            raise RuntimeError(f"No data from Jolpica for season {season}")

        # cumulative features up to previous round
        prev = df_season[df_season["round"] < rnd].copy()
        prev_points = prev.groupby("driverRef")["points"].sum() if not prev.empty else pd.Series(dtype=float)
        prev_wins = (
            prev.assign(win=(prev["position"] == 1).astype(int)).groupby("driverRef")["win"].sum()
            if not prev.empty
            else pd.Series(dtype=int)
        )

        # entries for target round
        entries = df_season[df_season["round"] == rnd][["driverRef", "constructorRef", "circuitRef", "grid"]].drop_duplicates()
        if entries.empty:
            quali_df, circuit_ref = self.fetch_round_qualifying_entries(season, rnd)
            if quali_df.empty:
                raise RuntimeError(f"Could not find entries for season {season} round {rnd}")
            entries = quali_df
            entries["circuitRef"] = circuit_ref

        # attach prev features
        entries["prev_points"] = entries["driverRef"].map(prev_points).fillna(0.0)
        entries["prev_wins"] = entries["driverRef"].map(prev_wins).fillna(0)

        # Map refs to Kaggle numeric ids
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
                "driverRef",
                "constructorRef",
                "circuitRef",
                "driverId",
                "constructorId",
                "circuitId",
                "grid",
                "prev_points",
                "prev_wins",
            ]
        ].copy()
        return final_df


class F1GrandPrixPredictor:
    def __init__(self, data_dir=None, jolpica: JolpicaClient | None = None):
        self.model = None
        self.driver_encoder = LabelEncoder()
        self.constructor_encoder = LabelEncoder()
        self.race_encoder = LabelEncoder()
        self.circuit_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        base_path = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else base_path / "data" / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.jolpica = jolpica or JolpicaClient()
        self.features = [
            "driver_encoded",
            "constructor_encoded",
            "grid",
            "circuit_encoded",
            "prev_points",
            "prev_wins",
        ]

    # ---------------------------- Kaggle data ----------------------------
    def download_kaggle_data(self, kaggle_dataset: str = "rohanrao/formula-1-world-championship-1950-2020"):
        required_files = {name: self.data_dir / name for name in ("results.csv", "races.csv", "drivers.csv", "constructors.csv", "circuits.csv")}
        if all(path.exists() for path in required_files.values()):
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
            raise RuntimeError(
                "Kaggle CLI is not installed or not available in PATH. Install it with `pip install kaggle`."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Kaggle download failed with exit code {exc.returncode}: {exc.stderr.strip()}") from exc

    def load_data(self):
        self.results = pd.read_csv(self.data_dir / "results.csv")
        self.races = pd.read_csv(self.data_dir / "races.csv")
        self.drivers = pd.read_csv(self.data_dir / "drivers.csv").drop(columns=["url"], errors="ignore")
        self.constructors = pd.read_csv(self.data_dir / "constructors.csv").drop(columns=["url"], errors="ignore")
        self.circuits = pd.read_csv(self.data_dir / "circuits.csv").drop(columns=["url"], errors="ignore")

    def preprocess(self):
        df = self.results.merge(self.races, on="raceId", suffixes=("", "_race"))
        df = df.merge(self.drivers, on="driverId")
        df = df.merge(self.constructors, on="constructorId")
        df = df.merge(self.circuits, on="circuitId")

        df["target"] = (df["positionOrder"] == 1).astype(int)
        df.sort_values(by=["driverId", "year", "round"], inplace=True)
        df["prev_points"] = df.groupby("driverId")["points"].transform(lambda s: s.cumsum() - s)
        win_ind = (df["positionOrder"] == 1).astype(int)
        df["prev_wins"] = win_ind.groupby(df["driverId"]).cumsum() - win_ind

        # use reference labels to remain consistent with Jolpica API identifiers
        df["driver_encoded"] = self.driver_encoder.fit_transform(df["driverRef"])
        df["constructor_encoded"] = self.constructor_encoder.fit_transform(df["constructorRef"])
        df["race_encoded"] = self.race_encoder.fit_transform(df["raceId"])
        df["circuit_encoded"] = self.circuit_encoder.fit_transform(df["circuitRef"])

        df_model = df[self.features + ["race_encoded", "target"]].dropna()
        X = df_model[self.features]
        y = df_model[["race_encoded", "target"]]

        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        self.X_train_raw = X.reset_index(drop=True)
        self.y_train_raw = y.reset_index(drop=True)

    # ---------------------------- Model ----------------------------
    def build_model(self):
        self.model = Sequential(
            [
                Input(shape=(len(self.features),)),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

    def train(self, epochs: int = 10, batch_size: int = 32):
        self.build_model()
        self.model.fit(
            self.X_train,
            self.y_train["target"],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
        )
        y_pred = (self.model.predict(self.X_test) > 0.5).astype("int32")
        print("Accuracy:", accuracy_score(self.y_test["target"], y_pred))
        print(classification_report(self.y_test["target"], y_pred))

    # ---------------------------- Inference ----------------------------
    def _encode_with_new(self, encoder: LabelEncoder, series: pd.Series, kind: str) -> np.ndarray:
        """Encode values, extending encoder with unseen labels if needed."""
        values = series.fillna("unknown").astype(str)
        unknown_mask = ~values.isin(encoder.classes_)
        if unknown_mask.any():
            new_classes = np.unique(values[unknown_mask])
            encoder.classes_ = np.unique(np.concatenate([encoder.classes_, new_classes]))
            print(f"[warn] Added new {kind} labels for inference: {', '.join(new_classes)}")
        return encoder.transform(values)

    def predict_from_csv(self, csv_path, output_path: str = "prediction_results.csv"):
        """CSV must contain columns: driverRef, constructorRef, grid, circuitRef, prev_points, prev_wins"""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Input CSV '{csv_path}' not found. Build one via Jolpica: "
                f"  predictor.build_inputs_csv_via_jolpica(season=2025, rnd=1, output_csv='{csv_path}')"
            )
        if path.stat().st_size == 0:
            raise ValueError(
                f"Input CSV '{csv_path}' is empty. Build it with Jolpica or fill required headers: "
                "driverRef,constructorRef,grid,circuitRef,prev_points,prev_wins"
            )

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError as e:
            raise ValueError(
                f"Input CSV '{csv_path}' has no columns. Ensure it has headers: "
                "driverRef,constructorRef,grid,circuitRef,prev_points,prev_wins"
            ) from e

        required = {"driverRef", "constructorRef", "grid", "circuitRef", "prev_points", "prev_wins"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in '{csv_path}': {sorted(missing)}. "
                "Regenerate inputs with build_inputs_csv_via_jolpica."
            )

        # Encode & scale (extend encoders with unseen Jolpica entries)
        df["driver_encoded"] = self._encode_with_new(self.driver_encoder, df["driverRef"], "driver")
        df["constructor_encoded"] = self._encode_with_new(self.constructor_encoder, df["constructorRef"], "constructor")
        df["circuit_encoded"] = self._encode_with_new(self.circuit_encoder, df["circuitRef"], "circuit")

        X = df[["driver_encoded", "constructor_encoded", "grid", "circuit_encoded", "prev_points", "prev_wins"]]
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled).flatten()
        # Convert independent sigmoids into a race-level distribution (via logits softmax)
        logits = (probs / (1 - probs + 1e-9))
        softmax_probs = tf.nn.softmax(tf.convert_to_tensor(logits)).numpy()
        df["win_probability"] = softmax_probs

        ordered_cols = ["driverRef"]
        if "driverId" in df.columns:
            ordered_cols.append("driverId")
        ordered_cols.append("win_probability")
        results = df[ordered_cols].sort_values(by="win_probability", ascending=False)
        results.to_csv(output_path, index=False)
        return results

    # ---------------------------- Jolpica helpers (composition) ----------------------------
    def build_inputs_csv_via_jolpica(self, season: int, rnd: int, output_csv: str) -> pd.DataFrame:
        """Use the JolpicaClient to assemble an inputs CSV for a season/round."""
        final_df = self.jolpica.build_inputs_for_round(
            season,
            rnd,
            kaggle_drivers=self.drivers,
            kaggle_constructors=self.constructors,
            kaggle_circuits=self.circuits,
        )
        final_df.to_csv(output_csv, index=False)
        return final_df


if __name__ == '__main__':
    predictor = F1GrandPrixPredictor()
    predictor.download_kaggle_data()
    predictor.load_data()
    predictor.preprocess()
    predictor.train()
    result_df = predictor.predict_from_csv('race_inputs.csv')
    print(result_df)
