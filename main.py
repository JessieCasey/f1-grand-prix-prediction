from f1_prediction import F1GrandPrixPredictor

if __name__ == '__main__':
    p = F1GrandPrixPredictor()
    p.download_kaggle_data()
    p.load_data()
    p.preprocess()
    p.train()

    p.build_inputs_csv_via_jolpica(season=2025, rnd=21, output_csv="race_inputs.csv")

    res = p.predict_from_csv("race_inputs.csv")
    print(res.head())
