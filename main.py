from f1_grand_prix_predictor import F1GrandPrixPredictor

if __name__ == '__main__':
    p = F1GrandPrixPredictor()
    p.download_kaggle_data()
    p.load_data()
    p.preprocess()
    p.train()

    # зібрати CSV для сезону/раунду (приклад: 2025, раунд 1)
    p.build_inputs_csv_via_jolpica(season=2025, rnd=2, output_csv="race_inputs.csv")

    # прогноз
    res = p.predict_from_csv("race_inputs.csv")
    print(res.head())