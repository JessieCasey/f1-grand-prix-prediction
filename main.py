from f1_grand_prix_predictor import F1GrandPrixPredictor

if __name__ == '__main__':
    predictor = F1GrandPrixPredictor()
    predictor.download_kaggle_data()
    predictor.load_data()
    predictor.preprocess()
    predictor.train()
    result_df = predictor.predict_from_csv("race_inputs.csv")
    print(result_df)

