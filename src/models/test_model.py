import pandas as pd
from src.models.naive_bayes import NaiveBayes

# Class for testing the model and making predictions on new data
class TestModel:
    def __init__(self, data: pd.DataFrame, model: NaiveBayes):
        self.__test_data = data
        self.__model = model

    # Predict the target value for a single row
    def predict(self, row: pd.Series) -> str | int:
        result = dict()
        for target in self.__model.target_value:
            temp = 1
            for col, val in row.items():
                temp *= self.__model.model_data[target][col].get(val, 1e-6)
            result[target] = temp * self.__model.target_value[target]
        return max(result, key=result.get)
    
    # Test the model accuracy on the test data
    def test_model(self) -> float:
        correct_cnt = 0
        for index, row in self.__test_data.iterrows():
            current_row = row.drop(self.__model.target_column)
            correct_value = row[self.__model.target_column]
            correct_cnt += (1 if self.predict(current_row) == correct_value else 0)
        return (correct_cnt / len(self.__test_data)) * 100

         
        






