import subprocess
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class F1GrandPrixPredictor:
    def __init__(self, data_dir=None):
        self.model = None
        self.driver_encoder = LabelEncoder()
        self.constructor_encoder = LabelEncoder()
        self.race_encoder = LabelEncoder()
        self.circuit_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        base_path = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else base_path / "data" / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.features = [
            'driver_encoded',
            'constructor_encoded',
            'grid',
            'circuit_encoded',
            'prev_points',
            'prev_wins'
        ]

    def download_kaggle_data(self, kaggle_dataset="rohanrao/formula-1-world-championship-1950-2020"):
        required_files = {
            name: self.data_dir / name
            for name in (
                'results.csv',
                'races.csv',
                'drivers.csv',
                'constructors.csv',
                'circuits.csv'
            )
        }

        if all(path.exists() for path in required_files.values()):
            return

        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", kaggle_dataset, "--unzip", "-p", str(self.data_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.data_dir)
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Kaggle CLI is not installed or not available in PATH. "
                "Install it with `pip install kaggle` inside your environment."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Kaggle download failed with exit code {exc.returncode}: {exc.stderr.strip()}"
            ) from exc

    def load_data(self):
        self.results = pd.read_csv(self.data_dir / 'results.csv')
        self.races = pd.read_csv(self.data_dir / 'races.csv')
        self.drivers = pd.read_csv(self.data_dir / 'drivers.csv').drop(columns=['url'], errors='ignore')
        self.constructors = pd.read_csv(self.data_dir / 'constructors.csv').drop(columns=['url'], errors='ignore')
        self.circuits = pd.read_csv(self.data_dir / 'circuits.csv').drop(columns=['url'], errors='ignore')

    def preprocess(self):
        df = self.results.merge(self.races, on='raceId', suffixes=('', '_race'))
        df = df.merge(self.drivers, on='driverId')
        df = df.merge(self.constructors, on='constructorId')
        df = df.merge(self.circuits, on='circuitId')

        df['target'] = (df['positionOrder'] == 1).astype(int)
        df.sort_values(by=['driverId', 'year', 'round'], inplace=True)
        df['prev_points'] = df.groupby('driverId')['points'].transform(lambda series: series.cumsum() - series)
        win_indicator = (df['positionOrder'] == 1).astype(int)
        df['prev_wins'] = win_indicator.groupby(df['driverId']).cumsum() - win_indicator

        df['driver_encoded'] = self.driver_encoder.fit_transform(df['driverId'])
        df['constructor_encoded'] = self.constructor_encoder.fit_transform(df['constructorId'])
        df['race_encoded'] = self.race_encoder.fit_transform(df['raceId'])
        df['circuit_encoded'] = self.circuit_encoder.fit_transform(df['circuitId'])

        df_model = df[self.features + ['race_encoded', 'target']].dropna()

        X = df_model[self.features]
        y = df_model[['race_encoded', 'target']]

        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.X_train_raw = X.reset_index(drop=True)
        self.y_train_raw = y.reset_index(drop=True)

    def build_model(self):
        self.model = Sequential(
            [
                Input(shape=(len(self.features),)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid'),
            ]
        )
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    def train(self, epochs=10, batch_size=32):
        self.build_model()
        self.model.fit(self.X_train, self.y_train['target'], epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
        y_pred = (self.model.predict(self.X_test) > 0.5).astype("int32")
        print("Accuracy:", accuracy_score(self.y_test['target'], y_pred))
        print(classification_report(self.y_test['target'], y_pred))

    def predict_from_csv(self, csv_path, output_path="prediction_results.csv"):
        """
        CSV must contain columns:
        driverId, constructorId, grid, circuitId, prev_points, prev_wins
        """
        df = pd.read_csv(csv_path)
        df['driver_encoded'] = self.driver_encoder.transform(df['driverId'])
        df['constructor_encoded'] = self.constructor_encoder.transform(df['constructorId'])
        df['circuit_encoded'] = self.circuit_encoder.transform(df['circuitId'])

        X = df[['driver_encoded', 'constructor_encoded', 'grid', 'circuit_encoded', 'prev_points', 'prev_wins']]
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled).flatten()
        softmax_probs = tf.nn.softmax(probs).numpy()
        df['win_probability'] = softmax_probs

        results = df[['driverId', 'win_probability']].sort_values(by='win_probability', ascending=False)
        results.to_csv(output_path, index=False)
        return results
