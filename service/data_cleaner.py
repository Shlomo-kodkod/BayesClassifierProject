import pandas as pd

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.__index = int(len(self.__all_data) * 0.7)
        self.__all_data = data

    def clean_data(self):
        for col in self.__all_data.columns:
            self.__all_data[col].fillna(self.__all_data[col].value_counts().idxmax())

    def get_train_data(self):
        if self.__index:
            return self.__all_data[:self.__index]
            
    def get_test_data(self):
        if self.__index:
            return self.__all_data[:self.__index]
        