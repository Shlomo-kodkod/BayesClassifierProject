import pandas as pd
from src.models.naive_bayes import NaiveBayes
from src.models.predictor import Predictor


# Class for testing the model accuracy on new data
class TestModel:
    def __init__(self, data: pd.DataFrame, model: NaiveBayes):
        self.__test_data = data
        self.__model = model
        self.__predictor = Predictor(model)

    # Test the model accuracy on the test data
    def test_model(self) -> float:
        correct_cnt = 0
        for index, row in self.__test_data.iterrows():
            current_row = row.drop(self.__model.target_column)
            correct_value = row[self.__model.target_column]
            correct_cnt += (1 if self.__predictor.predict(current_row) == correct_value else 0)
        return (correct_cnt / len(self.__test_data)) * 100


         
        






