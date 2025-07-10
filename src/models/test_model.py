import pandas as pd
from src.models.naive_bayes import NaiveBayes

class TestModel:
    def __init__(self, data: pd.DataFrame, model: NaiveBayes):
        self.__train_data = data
        self.__model = model


    def predict(self, row: pd.Series) -> str | int:
        result = dict()
        for target in self.__model.target_value:
            temp = 1
            for col, val in row.items():
                    temp *= self.__model.model_data[target][col].get(val, 1e-6)
            result[target] = temp * self.__model.target_value[target]
        return max(result, key=result.get)
    
    def test_model(self) -> float:
        correct_cnt = 0
        for index, row in self.__train_data.iterrows():
            current_row = row.drop(self.__model.target_column)
            correct_value = row[self.__model.target_column]
            correct_cnt += (1 if self.predict(current_row) == correct_value else 0)
        return (correct_cnt / len(self.__train_data)) * 100

         
        






