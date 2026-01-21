# Formula 1 Grand-prix Winner Prediction

Utilities for training gradient-boosted + race-specific models on Kaggle’s historical Formula 1 dataset and producing per-race winner probabilities.

> Thanks to the [jolpica/jolpica-f1](https://github.com/jolpica/jolpica-f1/tree/main) project for the Ergast-powered API that supplies up-to-date race entry data.

## Setup

```bash
poetry install
poetry run python main.py --season 2025 --round 19
```

The first run downloads the Kaggle dataset, trains the global and race-specific models, generates a CSV of inputs for the requested round via the Jolpica API, and writes predictions to `data/results/prediction_results.csv`.

### Feature Signals

Training uses engineered features derived from historical results:

- Driver/team identifiers (`driver_encoded`, `constructor_encoded`, `circuit_encoded`)
- Starting position on the grid (`grid`)
- Rolling driver stats: season-to-date points, wins, podiums, average finish
- Rolling constructor stats: season-to-date points, wins
- Season context: `year`, `round`

During inference, the same signals are prepared for the upcoming race to power the win-probability and finishing-position models.

## Inspect Results

```bash
poetry run python display_results.py --top 5
```

Shows the top finishers per race with win probabilities sourced from the latest prediction CSV.
Example output:

**2025 - Mexico City Grand Prix**

| driverRef | predicted_rank | predicted_position | win_probability | grid | constructorRef |
|-----------|----------------|--------------------|-----------------|------|----------------|
| norris    | 1              | 4.63               | 0.45694         | 3    | mclaren        |
| leclerc   | 2              | 6.20               | 0.31598         | 1    | ferrari        |
| russell   | 3              | 6.63               | 0.01630         | 2    | mercedes       |
| piastri   | 4              | 7.34               | 0.01435         | 4    | mclaren        |
| sainz     | 5              | 7.65               | 0.01044         | 5    | ferrari        |
