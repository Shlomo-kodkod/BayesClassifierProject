import pandas as pd
from src.models.naive_bayes import NaiveBayes


# Class for predicting the target value using a trained NaiveBayes model
class Predictor:
    def __init__(self ,model: NaiveBayes):
        self.__model = model

    # Predict the target value for a single row
    def predict(self, row: pd.Series) -> str | int:
        result = dict()
        for target in self.__model.target_value:
            temp = 1
            for col, val in row.items():
                temp *= self.__model.model_data[target][col].get(val, 1e-6)
            result[target] = temp * self.__model.target_value[target]
        prediction = max(result, key=result.get)
        return prediction