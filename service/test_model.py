import pandas as pd
from service.naive_bayes import NaiveBayes

class TestModel:
    def __init__(self, data: pd.DataFrame, model: NaiveBayes, target_map: dict[any, dict[any, dict[any, float]]]):
        self.__train_data = data
        self.__data_model = model
        self.__target_values = target_map

    def predict(self, row: pd.Series) -> str | int:
        result = dict()
        for target in self.__target_values:
            temp = 1
            for col, val in row.items():
                    temp *= self.__data_model[target][col].get(val, 1e-6)
            result[target] = temp * self.__target_values[target]
        return max(result, key=result.get)
    
    def test_model(self) -> float:
        correct_cnt = 0
        for index, row in self.__train_data.iterrows():
            correct_cnt += (1 if self.predict(row.iloc[:-1]) == row.iloc[-1] else 0)
        return (correct_cnt / len(self.__train_data)) * 100

         
        






